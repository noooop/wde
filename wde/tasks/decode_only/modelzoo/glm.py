# SPDX-License-Identifier: Apache-2.0
"""Inference-only HF format GLM-4 model compatible with THUDM weights."""

from .llama import LlamaForCausalLM


class GlmForCausalLM(LlamaForCausalLM):

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        # Hack Llama model to fit HF format GLM implementation
        # Attention difference between GLM and Llama:
        # 1. Half partial rotary_dim and no Neox style.
        # 2. There is no bias for o_proj in attention
        for layer in self.model.layers:
            layer.self_attn.rotary_emb.rotary_dim //= 2
            layer.self_attn.rotary_emb.is_neox_style = False
            layer.self_attn.o_proj.bias = None
            layer.self_attn.o_proj.skip_bias_add = True
