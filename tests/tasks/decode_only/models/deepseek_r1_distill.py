# example_prompts from https://github.com/langgptai/awesome-deepseek-prompts

import time

import pytest

from tests.tasks.decode_only.util import (HfDecodingRunner, WDERunner,
                                          check_logprobs_close)
from wde.workflows.core.backends.tokenizer import get_tokenizer

MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
]

example_prompts = [
    "What is 1+1?", """
    下面这段的代码的效率很低，且没有处理边界情况。请先解释这段代码的问题与解决方法，然后进行优化：
---
def fib(n):
    if n <= 2:
        return n
    return fib(n-1) + fib(n-2)
---
    """, """
请帮我用 HTML 生成一个五子棋游戏，所有代码都保存在一个 HTML 中。
    """, """
    用户将提供给你一段新闻内容，请你分析新闻内容，并提取其中的关键信息，以 JSON 的形式输出，输出的 JSON 需遵守以下的格式：
{
  "entiry": <新闻实体>,
  "time": <新闻时间，格式为 YYYY-mm-dd HH:MM:SS，没有请填 null>,
  "summary": <新闻内容总结>
}
    """, """
    假设诸葛亮死后在地府遇到了刘备，请模拟两个人展开一段对话。
    """
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("scheduling", ["sync"])
def test_models(model: str, max_tokens: int, scheduling: str) -> None:

    NUM_LOG_PROBS = 4

    tokenizer = get_tokenizer(model, trust_remote_code=True)

    prompts = []
    for content in example_prompts:
        prompt = tokenizer.apply_chat_template([
            {
                "role": "user",
                "content": content
            },
            {
                "role": "assistant",
                "content": "<think>\n"
            },
        ],
                                               tokenize=False)

        prompts.append(prompt)

    with HfDecodingRunner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs(prompts, max_tokens,
                                                       NUM_LOG_PROBS)

    with WDERunner(model, scheduling=scheduling) as wde_model:
        outputs = wde_model.generate_greedy_logprobs(prompts, max_tokens,
                                                     NUM_LOG_PROBS)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=outputs,
        name_0="hf",
        name_1="wde",
    )

    time.sleep(1)
