"""A layer that samples the next tokens from the model's outputs."""
from math import inf
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from wde.workflows.decoding.backends.sampling.sampling_metadata import \
    SamplingMetadata
from wde.workflows.decoding.backends.sampling.sampling_params import \
    SamplingType
from wde.workflows.decoding.schema.execute_io import (Logprob, Sample,
                                                      SamplerOutput)
from wde.workflows.decoding.schema.request import DecodingSchedulableRequest

SampleResultType = List[Tuple[List[int], List[int]]]


class Sampler(nn.Module):

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        assert logits is not None
        _, vocab_size = logits.shape

        # do_penalties = sampling_metadata.do_penalties
        do_top_p_top_k = sampling_metadata.do_top_p_top_k
        do_min_p = sampling_metadata.do_min_p

        # logits = _apply_min_tokens_penalty(logits, sampling_metadata)
        # Apply presence and frequency penalties.

        # Use float32 to apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits = logits.to(torch.float)
        logits.div_(sampling_metadata.temperatures.unsqueeze(dim=1))

        if do_top_p_top_k:
            logits = _apply_top_k_top_p(logits, sampling_metadata.top_ps,
                                        sampling_metadata.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_metadata.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        sampler_result = self._sample_with_torch(probs, logprobs,
                                                 sampling_metadata)

        return sampler_result

    @staticmethod
    def _sample_with_torch(
        probs: torch.Tensor,
        logprobs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        categorized_sample_indices = sampling_metadata.categorized_sample_indices

        categorized_requests = {}
        for request in sampling_metadata.sampling_requests:
            sampling_params = request.sampling_params
            sampling_type = sampling_params.sampling_type
            if sampling_type not in categorized_requests:
                categorized_requests[sampling_type] = []
            categorized_requests[sampling_type].append(request)

        multinomial_samples: Dict[SamplingType, torch.Tensor] = {}
        greedy_samples: Optional[torch.Tensor] = None

        for sampling_type in SamplingType:
            sample_indices = categorized_sample_indices[sampling_type]
            num_tokens = len(sample_indices)
            if num_tokens == 0:
                continue

            long_sample_indices = sample_indices.long()

            if sampling_type == SamplingType.GREEDY:
                greedy_samples = torch.argmax(logprobs[long_sample_indices],
                                              dim=-1)

            elif sampling_type in (SamplingType.RANDOM,
                                   SamplingType.RANDOM_SEED):
                max_n_in_batch = 1

                requests = (None if sampling_type == SamplingType.RANDOM else
                            categorized_requests[sampling_type])

                multinomial_samples[sampling_type] = _multinomial(
                    probs[long_sample_indices],
                    max_n_in_batch,
                    requests=requests)
            else:
                raise ValueError(f"Unsupported sampling type: {sampling_type}")

        return SamplerOutput(logprobs=logprobs,
                             categorized_requests=categorized_requests,
                             sampling_metadata=sampling_metadata,
                             multinomial_samples=multinomial_samples,
                             greedy_samples=greedy_samples)


def _apply_top_k_top_p(
    logits: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    # Apply top-k.
    top_k_mask = logits_sort.size(1) - k.to(torch.long)
    # Get all the top_k values.
    top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
    top_k_mask = logits_sort < top_k_mask
    logits_sort.masked_fill_(top_k_mask, -float("inf"))

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    # at least one
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = torch.empty_like(logits_sort).scatter_(dim=-1,
                                                    index=logits_idx,
                                                    src=logits_sort)
    return logits


def _apply_min_p(
    logits: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """
    Adapted from
    https://github.com/oobabooga/text-generation-webui/blob/3146124ec01f02c8fb1650a6517cf1b60b537aaf/modules/sampler_hijack.py#L16C17-L16C17
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    scaled_min_p = min_p.unsqueeze_(dim=1) * top_probs
    tokens_to_remove = probs < scaled_min_p
    logits = logits.masked_fill_(tokens_to_remove, -float("inf"))

    return logits


def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
    requests: Optional[List[DecodingSchedulableRequest]] = None,
) -> torch.Tensor:
    if num_samples > 1:
        probs = probs.repeat_interleave(num_samples, dim=0)
    q = torch.empty_like(probs)
    if requests is None:
        q.exponential_()
    else:
        sample_idx = 0
        for request in requests:
            stride = 1
            assert request.generator is not None
            q[sample_idx:sample_idx +
              stride].exponential_(generator=request.generator)
            sample_idx += stride
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


def get_sampler_output(execute_output: SamplerOutput):
    greedy_samples = execute_output.greedy_samples
    multinomial_samples = execute_output.multinomial_samples
    categorized_requests = execute_output.categorized_requests

    sample_results_dict = {}

    for sampling_type in SamplingType:
        if sampling_type not in categorized_requests:
            continue

        requests2sample = categorized_requests[sampling_type]

        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(requests2sample, greedy_samples)
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            if sampling_type not in multinomial_samples:
                continue

            sample_results = _random_sample(requests2sample,
                                            multinomial_samples[sampling_type])
        else:
            assert False

        sample_results_dict.update(sample_results)
    return sample_results_dict


def _greedy_sample(
    requests2sample: List[DecodingSchedulableRequest],
    samples: torch.Tensor,
):
    samples_lst = samples.tolist()
    sample_idx = 0
    results = []
    for request in requests2sample:
        if not request.c_do_sample:
            continue

        next_token_id = samples_lst[sample_idx]
        results.append((request.request_id, next_token_id))
        sample_idx += 1
    return results


def _random_sample(
    requests2sample: List[DecodingSchedulableRequest],
    random_samples: torch.Tensor,
):
    samples_lst = random_samples[:, 0].tolist()

    sample_idx = 0
    results = []
    for request in requests2sample:
        if not request.c_do_sample:
            continue

        next_token_id = samples_lst[sample_idx]
        results.append((request.request_id, next_token_id))
        sample_idx += 1
    return results


def get_sample(request: DecodingSchedulableRequest, output_token, logprobs):
    sampling_params = request.sampling_params
    if request.c_is_prompt and sampling_params.prompt_logprobs is not None:
        assert False, "prompt_logprobs not supported "

    if sampling_params.logprobs is None:
        sampled_logprobs_dict = {
            output_token: Logprob(logprob=inf, rank=None, decoded_token=None)
        }
        sample = Sample(output_token=output_token,
                        logprobs=sampled_logprobs_dict)

        return sample

    # get topk Logprob, slow test only
    num_logprobs = sampling_params.logprobs
    query_indices_gpu = torch.tensor(request.sample_logprobs_indices,
                                     device=logprobs.device)
    next_token_ids_gpu = torch.tensor([output_token], device=logprobs.device)

    selected_logprobs = logprobs[[
        query_indices_gpu,
        next_token_ids_gpu,
    ]]

    ranks = _get_ranks(
        logprobs[query_indices_gpu],
        next_token_ids_gpu,
    )

    top_logprobs, top_token_ids = torch.topk(logprobs[query_indices_gpu],
                                             num_logprobs,
                                             dim=-1)

    top_logprobs = top_logprobs.to('cpu')
    top_token_ids = top_token_ids.to('cpu')
    selected_logprobs = selected_logprobs.to('cpu')
    ranks = ranks.to('cpu')

    sampled_logprobs_dict = {
        output_token:
        Logprob(logprob=float(selected_logprobs[0]),
                rank=int(ranks[0]),
                decoded_token=None)
    }

    if num_logprobs is not None and num_logprobs > 0:
        top_ids = top_token_ids[0].tolist()
        top_probs = top_logprobs[0].tolist()
        top_ranks = range(1, num_logprobs + 1)

        sampled_logprobs_dict.update({
            top_id:
            Logprob(logprob=top_prob, rank=rank, decoded_token=None)
            for top_id, top_prob, rank in zip(top_ids, top_probs, top_ranks)
        })

    sample = Sample(output_token=output_token, logprobs=sampled_logprobs_dict)

    return sample


def _get_ranks(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    vals = x[torch.arange(0, len(x), device=x.device, dtype=indices.dtype),
             indices]
    result = (x > vals[:, None])
    del vals
    return result.sum(1).add_(1)
