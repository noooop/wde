import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from tests.tasks.utils import HfRunner, cleanup
from wde import LLM
from wde.tasks.decoding import SamplingParams


def check_logprobs_close(
    *,
    outputs_0_lst,
    outputs_1_lst,
    name_0: str,
    name_1: str,
    num_outputs_0_skip_tokens: int = 0,
    warn_on_mismatch: bool = True,
    always_check_logprobs: bool = False,
) -> None:
    assert len(outputs_0_lst) == len(outputs_1_lst)

    # Loop through responses to each prompt.
    for prompt_idx, (outputs_0,
                     outputs_1) in enumerate(zip(outputs_0_lst,
                                                 outputs_1_lst)):
        assert len(outputs_0) == len(outputs_1)
        if len(outputs_0) == 3:
            assert len(outputs_1) == 3
            # Break out tokens, text & sample logprobs
            # (prompt logprobs were not provided)
            output_ids_0, output_str_0, logprobs_0 = outputs_0
            output_ids_1, output_str_1, logprobs_1 = outputs_1
        elif len(outputs_0) == 4:
            assert len(outputs_1) == 4
            # Break out tokens, text, sample logprobs & prompt logprobs
            (
                output_ids_0,
                output_str_0,
                logprobs_0,
                prompt_logprobs_0,
            ) = outputs_0
            (
                output_ids_1,
                output_str_1,
                logprobs_1,
                prompt_logprobs_1,
            ) = outputs_1

            # Test prompt logprobs closeness
            if (prompt_logprobs_0 is not None
                    and prompt_logprobs_1 is not None):
                # Both sequences' prompt logprobs lists are not `None``
                # (although individual list elements may be `None`);
                # for each token's logprobs:
                for idx, (logprobs_elem_0, logprobs_elem_1) in enumerate(
                        zip(prompt_logprobs_0, prompt_logprobs_1)):
                    fail_msg = (
                        f"Prompt logprobs test:"
                        f"\n{name_0}:\tPrompt index {idx}\t{logprobs_elem_0}"
                        f"\n{name_1}:\tPrompt index {idx}\t{logprobs_elem_1}")

                    if logprobs_elem_0 is None:
                        # If the seq 0 token's logprobs are `None`,
                        # the seq 1 token's logprobs must be `None`
                        assert logprobs_elem_1 is None, fail_msg
                    else:
                        # If the seq 0 token's logprobs are not `None`,
                        # the seq 1 token's logprobs must not be `None`
                        assert logprobs_elem_1 is not None, fail_msg
                        # Logprobs check: top-k token choices must be the same
                        assert (set(logprobs_elem_0.keys()) == set(
                            logprobs_elem_1.keys())), fail_msg
            else:
                # Both sequence logprobs lists must be `None`
                fail_msg = (f"Prompt logprobs test:"
                            f"\n{name_0}:\tlogprobs\t{prompt_logprobs_0}"
                            f"\n{name_1}:\tlogprobs\t{prompt_logprobs_1}")

                assert (prompt_logprobs_0 is None
                        and prompt_logprobs_1 is None), fail_msg
        else:
            raise ValueError(f"Outputs tuple must have 3 or 4 elements but "
                             f"{len(outputs_0)} elements were provided: "
                             f"{outputs_0}")

        if logprobs_0 is None:
            logprobs_0 = [None] * len(output_ids_0)
        if logprobs_1 is None:
            logprobs_1 = [None] * len(output_ids_1)

        # Skip specified number of initial sequence #0 tokens
        # & logprobs, leaving output text as-is for simplicity
        # (text mismatches may generate warnings but do not
        # cause the test to fail.)
        if num_outputs_0_skip_tokens < 0:
            raise ValueError("num_outputs_0_skip_tokens must be non-negative")
        output_ids_0 = output_ids_0[num_outputs_0_skip_tokens:]
        logprobs_0 = logprobs_0[num_outputs_0_skip_tokens:]

        # Loop through generated tokens.
        for idx, (output_id_0,
                  output_id_1) in enumerate(zip(output_ids_0, output_ids_1)):

            is_tok_mismatch = output_id_0 != output_id_1

            # If generated tokens don't match
            # or it is desired to always check logprobs,
            # then
            if is_tok_mismatch or always_check_logprobs:
                logprobs_elem_0 = logprobs_0[idx]
                logprobs_elem_1 = logprobs_1[idx]

                # Each predicted token must be in top N logprobs of the other
                fail_msg = (
                    f"Test{prompt_idx}:"
                    f"\nMatched tokens:\t{output_ids_0[:idx]}"
                    f"\n{name_0}:\t{output_str_0!r}\t{logprobs_elem_0}"
                    f"\n{name_1}:\t{output_str_1!r}\t{logprobs_elem_1}")

                assert logprobs_elem_0 is not None, fail_msg
                assert logprobs_elem_1 is not None, fail_msg
                assert output_id_0 in logprobs_elem_1, fail_msg
                assert output_id_1 in logprobs_elem_0, fail_msg

                if warn_on_mismatch and is_tok_mismatch:
                    with warnings.catch_warnings():
                        # This ensures that repeated warnings are shown
                        # in the output, not just the first occurrence
                        warnings.simplefilter("always")

                        warnings.warn(fail_msg, stacklevel=2)

                # Break out since sequences will now diverge.
                break
        else:
            if output_str_0 != output_str_1 and warn_on_mismatch:
                # The token outputs exactly match,
                # so the text outputs should exactly match as well
                fail_msg = (f"Test{prompt_idx}:"
                            f"\n{name_0}:\t{output_str_0!r}"
                            f"\n{name_1}:\t{output_str_1!r}")

                with warnings.catch_warnings():
                    # This ensures that repeated warnings are shown
                    # in the output, not just the first occurrence
                    warnings.simplefilter("always")

                    warnings.warn(fail_msg, stacklevel=2)


class WDERunner:

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 1024,
        dtype: str = "auto",
        **kwargs,
    ) -> None:
        self.model = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.7,
            **kwargs,
        )

    def generate_w_logprobs(self, prompts, sampling_params: SamplingParams):

        self.model.engine.executor.worker.model_runner.sampler.include_gpu_probs_tensor = True

        req_outputs = self.model.generate(prompts,
                                          sampling_params=sampling_params)

        toks_str_logsprobs_prompt_logprobs = (
            self._final_steps_generate_w_logprobs(req_outputs))
        # Omit prompt logprobs if not required by sampling params
        return ([x[0:-1] for x in toks_str_logsprobs_prompt_logprobs]
                if sampling_params.prompt_logprobs is None else
                toks_str_logsprobs_prompt_logprobs)

    @staticmethod
    def _final_steps_generate_w_logprobs(req_outputs, ):
        outputs = []
        for req_output in req_outputs:
            assert len(req_output.outputs) > 0
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                output_logprobs = sample.logprobs
            outputs.append((output_ids, output_str, output_logprobs,
                            req_output.prompt_logprobs))
        return outputs

    def generate_greedy_logprobs(
        self,
        prompts,
        max_tokens: int,
        num_logprobs: int,
        num_prompt_logprobs=None,
        stop_token_ids=None,
    ):
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=num_prompt_logprobs,
            stop_token_ids=stop_token_ids)

        return self.generate_w_logprobs(prompts, greedy_logprobs_params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


class HfDecodingRunner(HfRunner):

    def generate_greedy_logprobs(
        self,
        prompts,
        max_tokens: int,
        num_logprobs: int,
        **kwargs,
    ):
        all_logprobs = []
        all_output_ids = []
        all_output_strs = []

        for inputs in prompts:
            model_inputs = self.tokenizer([inputs],
                                          return_tensors="pt").to("cuda")

            output = self.model.generate(
                **model_inputs,
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )

            (
                seq_logprobs_lst,
                output_len,
            ) = self._hidden_states_to_logprobs(output.hidden_states,
                                                num_logprobs)

            all_logprobs.append(seq_logprobs_lst)
            seq_ids = output.sequences[0]
            output_len = len(seq_logprobs_lst)
            output_ids = seq_ids[-output_len:]
            all_output_ids.append(output_ids.tolist())
            all_output_strs.append(self.tokenizer.decode(output_ids))

        outputs = zip(all_output_ids, all_output_strs, all_logprobs)
        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    def _hidden_states_to_logprobs(
        self,
        hidden_states: Tuple[Tuple[torch.Tensor, ...], ...],
        num_logprobs: int,
    ) -> Tuple[List[Dict[int, float]], int]:
        seq_logprobs = self._hidden_states_to_seq_logprobs(hidden_states)
        output_len = len(hidden_states)

        # convert to dict
        seq_logprobs_lst: List[Dict[int, float]] = []
        for tok_idx, tok_logprobs in enumerate(seq_logprobs):
            # drop prompt logprobs
            if tok_idx == 0:
                tok_logprobs = tok_logprobs[-1, :].reshape(1, -1)
            topk = tok_logprobs.topk(num_logprobs)

            tok_logprobs_dct = {}
            for token_id, logprob in zip(topk.indices[0], topk.values[0]):
                tok_logprobs_dct[token_id.item()] = logprob.item()

            seq_logprobs_lst.append(tok_logprobs_dct)

        return (
            seq_logprobs_lst,
            output_len,
        )

    def _hidden_states_to_seq_logprobs(
        self,
        hidden_states: Tuple[Tuple[torch.Tensor, ...], ...],
    ) -> List[torch.Tensor]:
        output_embeddings = self.model.get_output_embeddings()

        seq_logprobs: List[torch.Tensor] = []
        for _, hidden_state in enumerate(hidden_states):
            last_hidden_states = hidden_state[-1][0]
            logits = torch.matmul(
                last_hidden_states.to(output_embeddings.weight.device),
                output_embeddings.weight.t(),
            )
            if getattr(output_embeddings, "bias", None) is not None:
                logits += output_embeddings.bias.unsqueeze(0)
            logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            seq_logprobs.append(logprobs)

        return seq_logprobs
