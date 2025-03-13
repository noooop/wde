import random

import numpy as np
from transformers import AutoModelForSequenceClassification

from tests.tasks.utils import HfRerankerRunner


def hf_reranker_runner(model, dtype, example_prompts):
    with HfRerankerRunner(
            model, dtype=dtype,
            auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.compute_score(example_prompts)
    return hf_outputs


def get_reranker_example_prompts():
    pairs = [
        ["query", "passage"],
        ["what is panda?", "hi"],
        [
            "what is panda?",
            "The giant panda (Ailuropoda melanoleuca), "
            "sometimes called a panda bear or simply panda, "
            "is a bear species endemic to China.",
        ],
    ] * 11
    random.shuffle(pairs)
    return pairs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
