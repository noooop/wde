from vllm.transformers_utils.detokenizer import (convert_prompt_ids_to_tokens,
                                                 detokenize_incrementally)
from vllm.transformers_utils.tokenizer import get_tokenizer

from wde.workflows.decoding import SamplingParams
from wde.workflows.decoding.schema.request import DecodingSchedulableRequest

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

    def decode_inplace(self, request: DecodingSchedulableRequest,
                       prms: SamplingParams) -> int:
        """Decodes the new token for a sequence. In-place operation.

        Args:
            seq: The sequence to decode.
            prms: The sampling parameters used to generate the sequence.

        Returns:
            The number of characters added to the output text.
        """

        all_input_ids = request.get_token_ids()
        token_id_generated_this_iteration = all_input_ids[-1]
        tokenizer = self.tokenizer

        # Convert prompt token IDs to tokens if necessary.
        # Do it here so that we don't have to repeat this
        # computation for each logprob.
        if request.tokens is None:
            (request.tokens, request.prefix_offset,
             request.read_offset) = convert_prompt_ids_to_tokens(
                 tokenizer=tokenizer,
                 prompt_ids=all_input_ids[:-1],
                 skip_special_tokens=prms.skip_special_tokens,
             )

        (new_tokens, new_decoded_token_text, prefix_offset,
         read_offset) = detokenize_incrementally(
             tokenizer=tokenizer,
             all_input_ids=all_input_ids,
             prev_tokens=request.tokens,
             prefix_offset=request.prefix_offset,
             read_offset=request.read_offset,
             skip_special_tokens=prms.skip_special_tokens,
             spaces_between_special_tokens=prms.spaces_between_special_tokens,
         )

        # Decode logprobs
        logprobs = request.output_logprobs[-1]
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
                        prev_tokens=request.tokens,
                        prefix_offset=request.prefix_offset,
                        read_offset=request.read_offset,
                        skip_special_tokens=prms.skip_special_tokens,
                        spaces_between_special_tokens=prms.
                        spaces_between_special_tokens,
                    )
                    sample_logprob.decoded_token = new_text

        request.tokens.extend(new_tokens)
        request.prefix_offset = prefix_offset
        request.read_offset = read_offset
        request.output_text += new_decoded_token_text
        return len(new_decoded_token_text)
