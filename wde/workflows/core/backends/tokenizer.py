from vllm.transformers_utils.detokenizer import (convert_prompt_ids_to_tokens,
                                                 detokenize_incrementally)
from vllm.transformers_utils.tokenizer import \
    get_tokenizer as vllm_get_tokenizer

from wde.workflows.core.backends.models.transformers_utils.config import \
    maybe_model_redirect


def get_tokenizer(tokenizer_name, *args, **kwargs):
    tokenizer_name = maybe_model_redirect(tokenizer_name)
    return vllm_get_tokenizer(tokenizer_name, *args, **kwargs)


__all__ = [
    "get_tokenizer", "convert_prompt_ids_to_tokens", "detokenize_incrementally"
]
