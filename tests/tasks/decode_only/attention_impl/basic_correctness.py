import itertools as it
import random

import pytest
import torch

from tests.tasks.utils import compare_embeddings


@pytest.fixture(scope="session")
def example_prompts():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 11
    random.shuffle(prompts)
    return prompts


MODELS = ["Qwen/Qwen2-0.5B-Instruct"]

AttentionImpls_fp32 = ["TORCH_SDPA", "XFORMERS", "TORCH_NAIVE"]
AttentionImpls_fp16 = [
    "FLASH_ATTN", "TORCH_SDPA", "XFORMERS", "FLASHINFER", "TORCH_NAIVE"
]
AttentionImpls_bf16 = [
    "FLASH_ATTN", "TORCH_SDPA", "XFORMERS", "FLASHINFER", "TORCH_NAIVE"
]

AttentionImpls = {
    "float": AttentionImpls_fp32,
    "half": AttentionImpls_fp16,
    "bfloat16": AttentionImpls_bf16,
}


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float", "half", "bfloat16"])
@pytest.mark.parametrize("max_num_requests", [2])
@pytest.mark.parametrize("scheduling", ["sync"])
@torch.inference_mode
def test_basic_correctness_fp16(
    wde_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_num_requests: int,
    scheduling: str,
) -> None:
    attention_impls = AttentionImpls[dtype]

    impl_outputs_list = []

    for attention_impl in attention_impls:
        with wde_runner(model,
                        dtype=dtype,
                        max_num_requests=max_num_requests,
                        scheduling=scheduling,
                        attention_impl=attention_impl,
                        output_last_hidden_states=True) as engine:
            outputs = engine.encode(example_prompts)
            impl_outputs_list.append((attention_impl, outputs))

    tolerance = 1e-2
    for a, b in it.combinations(impl_outputs_list, 2):
        similarities = compare_embeddings(a[1], b[1])
        all_similarities = torch.stack(similarities)

        assert torch.all(
            (all_similarities <= 1.0 + tolerance)
            & (all_similarities >= 1.0 - tolerance)
        ), f"{a[0]} vs {b[0]}, not all values are within {tolerance} of 1.0"
