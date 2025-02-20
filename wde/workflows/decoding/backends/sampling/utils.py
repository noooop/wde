import random
from typing import List, Union


class TokenSampler:

    def __init__(self, tokenizer, trust_remote_code=True):
        if isinstance(tokenizer, str):
            from vllm.transformers_utils.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(tokenizer,
                                      trust_remote_code=trust_remote_code)

        self.tokenizer = tokenizer

        vocab = tokenizer.get_vocab()
        vocab = {
            k: v
            for k, v in vocab.items() if k not in tokenizer.all_special_ids
        }
        vocab = list(vocab.values())

        self.vocab = vocab

    def random_sample(self,
                      length: int,
                      decode: bool = False) -> Union[List[int], str]:
        prompt_token_ids = random.choices(self.vocab, k=length)

        if not decode:
            return prompt_token_ids

        prompt = self.tokenizer.decode(prompt_token_ids)
        return prompt
