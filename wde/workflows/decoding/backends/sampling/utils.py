import random
from typing import List


class TokenSampler:

    def __init__(self, tokenizer, trust_remote_code=False):
        if isinstance(tokenizer, str):
            from vllm.transformers_utils.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(tokenizer,
                                      trust_remote_code=trust_remote_code)

        vocab = tokenizer.get_vocab()
        vocab = {
            k: v
            for k, v in vocab.items() if k not in tokenizer.all_special_ids
        }
        vocab = list(vocab.values())

        self.vocab = vocab

    def random_sample(self, length: int) -> List[int]:
        return random.choices(self.vocab, k=length)
