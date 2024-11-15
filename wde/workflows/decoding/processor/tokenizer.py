from typing import Dict, List, Optional

from vllm.transformers_utils.detokenizer import (convert_prompt_ids_to_tokens,
                                                 detokenize_incrementally)
from vllm.transformers_utils.tokenizer import get_tokenizer

from wde.workflows.decoding import SamplingParams
from wde.workflows.decoding.backends.sequence import (Logprob, Sequence,
                                                      SequenceGroup)

INVALID_TOKEN_ID = -1


class Tokenizer(object):

    def __init__(self, tokenizer_name: str, **kwargs):
        self.tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, **kwargs)

    @classmethod
    def from_engine(cls, engine):
        init_kwargs = dict(
            tokenizer_name=engine.engine_config.model_config.tokenizer,
            tokenizer_mode=engine.engine_config.model_config.tokenizer_mode,
            trust_remote_code=engine.engine_config.model_config.
            trust_remote_code,
            revision=engine.engine_config.model_config.tokenizer_revision)

        return cls(**init_kwargs)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def decode_prompt_logprobs_inplace(self, seq_group: SequenceGroup,
                                       prompt_logprobs: List[Optional[Dict[
                                           int, Logprob]]],
                                       position_offset: int) -> None:
        """Decodes the logprobs for the prompt of a sequence group.

        Args:
            seq_group: The sequence group to decode.
            prompt_logprobs: The logprobs to decode.
            position_offset: Offset of the first index of the logprobs
                relative to the start of the sequence (for chunked prefill).

        Returns:
            The prompt logprobs with the decoded tokens.
        """
        prms = seq_group.sampling_params
        assert prms is not None

        # We can pick any sequence for the prompt.
        seq = seq_group.get_seqs()[0]
        # Only prompt, without the generated token.
        all_token_ids = seq.get_token_ids()
        prompt_token_ids = all_token_ids[:-1]
        tokenizer = self.tokenizer
        prefix_offset = 0
        read_offset = 0
        next_iter_prefix_offset = 0
        next_iter_read_offset = 0
        next_iter_tokens: List[str] = []
        prev_tokens = None

        for token_position_in_logprob, prompt_logprobs_for_token in enumerate(
                prompt_logprobs):

            # Absolute token position equals the index in the logprobs
            # list plus the offset of the entire logprobs list relative
            # to the start of the sequence.
            token_position = token_position_in_logprob + position_offset
            if not prompt_logprobs_for_token:
                continue
            for token_id, sample_logprob in prompt_logprobs_for_token.items():
                if (sample_logprob.decoded_token is None
                        and token_id != INVALID_TOKEN_ID):
                    prompt_token_ids_with_token = (
                        prompt_token_ids[:token_position] + [token_id])
                    (new_tokens, new_text, new_prefix_offset,
                     new_read_offset) = detokenize_incrementally(
                         tokenizer=tokenizer,
                         all_input_ids=prompt_token_ids_with_token,
                         prev_tokens=prev_tokens,
                         prefix_offset=prefix_offset,
                         read_offset=read_offset,
                         skip_special_tokens=prms.skip_special_tokens,
                         spaces_between_special_tokens=prms.
                         spaces_between_special_tokens,
                     )

                    sample_logprob.decoded_token = new_text

                    # Use the offsets & prev tokens corresponding to
                    # real tokens to ensure detokenization is consistent
                    # actual with prompt.
                    if token_id == all_token_ids[token_position]:
                        next_iter_prefix_offset = new_prefix_offset
                        next_iter_read_offset = new_read_offset
                        next_iter_tokens = new_tokens

            # Advance to the next token position.
            prefix_offset = next_iter_prefix_offset
            read_offset = next_iter_read_offset
            if prev_tokens is None:
                prev_tokens = next_iter_tokens
            else:
                prev_tokens.extend(next_iter_tokens)

    def decode_sequence_inplace(self, seq: Sequence,
                                prms: SamplingParams) -> int:
        """Decodes the new token for a sequence. In-place operation.

        Args:
            seq: The sequence to decode.
            prms: The sampling parameters used to generate the sequence.

        Returns:
            The number of characters added to the output text.
        """
        all_input_ids = seq.get_token_ids()
        token_id_generated_this_iteration = all_input_ids[-1]
        tokenizer = self.tokenizer

        # Convert prompt token IDs to tokens if necessary.
        # Do it here so that we don't have to repeat this
        # computation for each logprob.
        if seq.tokens is None:
            (seq.tokens, seq.prefix_offset,
             seq.read_offset) = convert_prompt_ids_to_tokens(
                 tokenizer=tokenizer,
                 prompt_ids=all_input_ids[:-1],
                 skip_special_tokens=prms.skip_special_tokens,
             )

        (new_tokens, new_decoded_token_text, prefix_offset,
         read_offset) = detokenize_incrementally(
             tokenizer=tokenizer,
             all_input_ids=all_input_ids,
             prev_tokens=seq.tokens,
             prefix_offset=seq.prefix_offset,
             read_offset=seq.read_offset,
             skip_special_tokens=prms.skip_special_tokens,
             spaces_between_special_tokens=prms.spaces_between_special_tokens,
         )

        # Decode logprobs
        logprobs = seq.output_logprobs[-1]
        if logprobs:
            previous_tokens = all_input_ids[:-1]
            for token_id, sample_logprob in logprobs.items():
                # If the token was generated this iteration,
                # use the provided text.
                if token_id == token_id_generated_this_iteration:
                    sample_logprob.decoded_token = new_decoded_token_text
                    continue

                if (sample_logprob.decoded_token is None
                        and token_id != INVALID_TOKEN_ID):
                    all_input_ids_with_logprob = previous_tokens + [token_id]
                    (_, new_text, _, _) = detokenize_incrementally(
                        tokenizer=tokenizer,
                        all_input_ids=all_input_ids_with_logprob,
                        prev_tokens=seq.tokens,
                        prefix_offset=seq.prefix_offset,
                        read_offset=seq.read_offset,
                        skip_special_tokens=prms.skip_special_tokens,
                        spaces_between_special_tokens=prms.
                        spaces_between_special_tokens,
                    )
                    sample_logprob.decoded_token = new_text

        seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_decoded_token_text

        return len(new_decoded_token_text)
