from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from vllm.utils import is_pin_memory_available, make_tensor_with_pad

from .sampling_params import SamplingType

if TYPE_CHECKING:
    from wde.workflows.decoding.schema.request import \
        DecodingSchedulableRequest

pin_memory = is_pin_memory_available()
_SAMPLING_EPS = 1e-5
VLLM_TOKEN_ID_ARRAY_TYPE = "l"
VLLM_INVALID_TOKEN_ID = -1


@dataclass
class SamplingMetadata:
    sampling_requests: List["DecodingSchedulableRequest"]

    selected_token_indices: torch.Tensor
    categorized_sample_indices: Dict[SamplingType, torch.Tensor]

    # SamplingTensors
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor
    presence_penalties: torch.Tensor
    frequency_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    prompt_tokens: torch.Tensor
    output_tokens: torch.Tensor

    do_penalties: bool
    do_top_p_top_k: bool
    do_min_p: bool

    @staticmethod
    def prepare(
        scheduled_requests: List["DecodingSchedulableRequest"],
        vocab_size: int,
        dtype: torch.dtype,
        device: str = "cuda:0",
    ) -> "SamplingMetadata":
        (selected_token_indices, categorized_sample_indices,
         num_prompts) = SamplingMetadata._prepare_requests(
             scheduled_requests, device)

        selected_token_indices = torch.tensor(
            selected_token_indices,
            dtype=torch.long,
            device="cpu",
            pin_memory=pin_memory,
        )

        categorized_sample_indices = {
            t:
            torch.tensor(
                seq_ids,
                dtype=torch.int,
                device="cpu",
                pin_memory=pin_memory,
            )
            for t, seq_ids in categorized_sample_indices.items()
        }

        sampling_tensors, do_penalties, do_top_p_top_k, do_min_p = SamplingMetadata._prepare_sampling_tensors(
            scheduled_requests, vocab_size, dtype)

        sampling_metadata = SamplingMetadata(
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            sampling_requests=scheduled_requests,
            do_penalties=do_penalties,
            do_top_p_top_k=do_top_p_top_k,
            do_min_p=do_min_p,
            **sampling_tensors)
        return sampling_metadata

    @staticmethod
    def _prepare_requests(
        scheduled_requests: List["DecodingSchedulableRequest"],
        device,
    ):

        selected_token_indices: List[int] = []
        model_output_idx = 0

        categorized_sample_indices: Dict[SamplingType, List[int]] = {
            t: []
            for t in SamplingType
        }

        logit_idx = 0
        num_prompts = 0

        for i, request in enumerate(scheduled_requests):
            do_sample = request.c_do_sample
            is_prompt = request.c_is_prompt
            sampling_params = request.sampling_params

            if request.generator is None and request.sampling_params.seed is not None:
                request.generator = torch.Generator(device=device).manual_seed(
                    request.sampling_params.seed)

            if is_prompt:
                num_prompts += 1
                num_prefill_sample = 1
                assert num_prefill_sample == 1

                prompt_logprob_len = (request.c_query_len - num_prefill_sample
                                      if do_sample else request.c_query_len)
                sample_len = num_prefill_sample if do_sample else 0
            else:
                prompt_logprob_len = 0

                sample_len = request.c_query_len if do_sample else 0

            # Update indices to select from the model output.
            """
            This blocks computes selected_token_indices which is used in the
            following way.

            hidden_states = model(...)
            logits = hidden_states[selected_token_indices]
            """

            if sampling_params.prompt_logprobs is not None:
                selected_token_indices.extend(
                    range(model_output_idx,
                          model_output_idx + prompt_logprob_len))
            model_output_idx += prompt_logprob_len

            if do_sample:
                selected_token_indices.extend(
                    range(model_output_idx, model_output_idx + sample_len))
            model_output_idx += sample_len

            # We now find indices for logprob computation and sampling.
            """
            This block computes categorized_sample_indices which is used in the
            following way.

            hidden_states = model(...)
            logits = hidden_states[selected_token_indices]
            def sample(logits):
               # Use categorized_sample_indices for sampling.
               # prompt_logprobs_indices to find prompt logprob indices.
               # sample_indices to find sample indices.
            """

            if sampling_params.prompt_logprobs is not None:
                prompt_logprobs_indices = range(logit_idx,
                                                logit_idx + prompt_logprob_len)
                request.prompt_logprobs_indices = prompt_logprobs_indices

                logit_idx += prompt_logprob_len

            if do_sample:
                sample_logprobs_indices = range(logit_idx,
                                                logit_idx + sample_len)
                request.sample_logprobs_indices = sample_logprobs_indices

                categorized_sample_indices[
                    sampling_params.sampling_type].extend(
                        sample_logprobs_indices)

                logit_idx += sample_len

        return (selected_token_indices, categorized_sample_indices,
                num_prompts)

    @staticmethod
    def _prepare_sampling_tensors(
            scheduled_requests: List["DecodingSchedulableRequest"], vocab_size,
            dtype):
        prompt_tokens: List[array] = []
        output_tokens: List[array] = []
        top_ks: List[int] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        min_ps: List[float] = []
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        repetition_penalties: List[float] = []
        do_penalties = False
        do_top_p_top_k = False
        do_min_p = False

        for request in scheduled_requests:
            sampling_params = request.sampling_params
            temperature = sampling_params.temperature
            p = sampling_params.presence_penalty
            f = sampling_params.frequency_penalty
            r = sampling_params.repetition_penalty
            top_p = sampling_params.top_p
            min_p = sampling_params.min_p

            # k should not be greater than the vocab size.
            top_k = min(sampling_params.top_k, vocab_size)
            top_k = vocab_size if top_k == -1 else top_k
            if temperature < _SAMPLING_EPS:
                # NOTE: Zero temperature means deterministic sampling
                # (i.e., greedy sampling or beam search).
                # Set the temperature to 1 to avoid division by zero.
                temperature = 1.0
            if not do_top_p_top_k and (top_p < 1.0 - _SAMPLING_EPS
                                       or top_k != vocab_size):
                do_top_p_top_k = True
            if not do_min_p and min_p > _SAMPLING_EPS:
                do_min_p = True
            if not do_penalties and (abs(p) >= _SAMPLING_EPS
                                     or abs(f) >= _SAMPLING_EPS
                                     or abs(r - 1.0) >= _SAMPLING_EPS):
                do_penalties = True

            is_prompt = request.c_is_prompt

            if is_prompt and sampling_params.prompt_logprobs is not None:
                # For tokens in the prompt that we only need to get
                # their logprobs
                query_len = request.query_len
                assert query_len is not None
                prefill_len = len(request.prompt_logprobs_indices)
                temperatures += [temperature] * prefill_len
                top_ps += [top_p] * prefill_len
                top_ks += [top_k] * prefill_len
                min_ps += [min_p] * prefill_len
                presence_penalties += [0] * prefill_len
                frequency_penalties += [0] * prefill_len
                repetition_penalties += [1] * prefill_len

            if request.c_do_sample:
                sample_lens = len(request.sample_logprobs_indices)
                temperatures += [temperature] * sample_lens
                top_ps += [top_p] * sample_lens
                top_ks += [top_k] * sample_lens
                min_ps += [min_p] * sample_lens
                presence_penalties += [p] * sample_lens
                frequency_penalties += [f] * sample_lens
                repetition_penalties += [r] * sample_lens

        if do_penalties:
            for request in scheduled_requests:
                sampling_params = request.sampling_params

                if (request.c_is_prompt
                        and sampling_params.prompt_logprobs is not None):
                    prefill_len = len(request.prompt_logprobs_indices)
                    prompt_tokens.extend(
                        array(VLLM_TOKEN_ID_ARRAY_TYPE)
                        for _ in range(prefill_len))
                    output_tokens.extend(
                        array(VLLM_TOKEN_ID_ARRAY_TYPE)
                        for _ in range(prefill_len))

                if request.do_sample:
                    prompt_tokens.append(
                        array(VLLM_TOKEN_ID_ARRAY_TYPE,
                              request.prompt_token_ids))
                    output_tokens.append(
                        array(VLLM_TOKEN_ID_ARRAY_TYPE,
                              request.output_token_ids))

        sampling_tensors = SamplingMetadata._lists2tensers(
            temperatures, top_ps, top_ks, min_ps, presence_penalties,
            frequency_penalties, repetition_penalties, prompt_tokens,
            output_tokens, vocab_size, dtype)
        return sampling_tensors, do_penalties, do_top_p_top_k, do_min_p

    @staticmethod
    def _lists2tensers(
        temperatures: List[float],
        top_ps: List[float],
        top_ks: List[int],
        min_ps: List[float],
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
        prompt_tokens: List[array],
        output_tokens: List[array],
        vocab_size: int,
        dtype: torch.dtype,
    ):
        do_penalties = prompt_tokens or output_tokens

        if do_penalties:
            prompt_t = make_tensor_with_pad(
                prompt_tokens,
                vocab_size,
                device="cpu",
                dtype=torch.int64,
                pin_memory=pin_memory,
            )
            output_t = make_tensor_with_pad(
                output_tokens,
                vocab_size,
                device="cpu",
                dtype=torch.int64,
                pin_memory=pin_memory,
            )
        else:
            empty_tensor = torch.empty(0, device="cpu", dtype=torch.long)
            prompt_t = empty_tensor
            output_t = empty_tensor

        temperatures_t = torch.tensor(
            temperatures,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        top_ps_t = torch.tensor(
            top_ps,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        min_ps_t = torch.tensor(
            min_ps,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        presence_penalties_t = torch.tensor(
            presence_penalties,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        frequency_penalties_t = torch.tensor(
            frequency_penalties,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        repetition_penalties_t = torch.tensor(
            repetition_penalties,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        top_ks_t = torch.tensor(
            top_ks,
            device="cpu",
            dtype=torch.int,
            pin_memory=pin_memory,
        )

        sampling_tensors = dict(
            temperatures=temperatures_t,
            top_ps=top_ps_t,
            top_ks=top_ks_t,
            min_ps=min_ps_t,
            presence_penalties=presence_penalties_t,
            frequency_penalties=frequency_penalties_t,
            repetition_penalties=repetition_penalties_t,
            prompt_tokens=prompt_t,
            output_tokens=output_t,
        )
        return sampling_tensors

    def deferred_to(self, device, non_blocking=True):
        for k in self.__dict__:
            if not hasattr(self.__dict__[k], "to"):
                continue
            self.__dict__[k] = self.__dict__[k].to(device=device,
                                                   non_blocking=non_blocking)

        for k in self.categorized_sample_indices:
            self.categorized_sample_indices[
                k] = self.categorized_sample_indices[k].to(
                    device=device, non_blocking=non_blocking)

        return self
