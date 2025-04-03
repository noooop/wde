from typing import Iterable, Tuple

import torch
from easydict import EasyDict as edict
from torch import nn
from vllm.config import CacheConfig, ModelConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from wde.workflows.core.backends.distributed import patch_parallel_state
from wde.workflows.core.backends.models.transformers_utils.config import (
    get_config, model_overwrite)

patch_parallel_state()

model = "deepseek-ai/DeepSeek-R1"

hf_config = get_config(model=model_overwrite(model), trust_remote_code=True)

GB = 1 << 30

config = edict()

config.vocab_size = 129280
config.hidden_size = 7168
config.intermediate_size = 18432
config.moe_intermediate_size = 2048
config.hidden_act = "silu"
config.dtype = torch.bfloat16
config.device = "cuda"

config.num_attention_heads = 128
config.qk_nope_head_dim = 128
config.qk_rope_head_dim = 64
config.block_size = 16

config.q_lora_rank = 1536
config.kv_lora_rank = 512
config.v_head_dim = 128
config.c_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim

config.rms_norm_eps = 1e-06

config.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
config.scaling = config.qk_head_dim**-0.5

config.weight_block_size = 128

config.n_blocks = 2048

config.rope_theta = 10000
config.rope_scaling = hf_config.rope_scaling
config.max_position_embeddings = 163840

config.n_routed_experts = 256

config.num_hidden_layers = 61
config.first_k_dense_replace = 3

torch.set_default_dtype(config.dtype)

quant_config = Fp8Config(
    is_checkpoint_fp8_serialized=True,
    activation_scheme='dynamic',
    weight_block_size=[config.weight_block_size, config.weight_block_size])

cache_config = CacheConfig(
    block_size=config.block_size,
    gpu_memory_utilization=0.9,
    swap_space=0,
    cache_dtype="auto",
)

model_config = ModelConfig(model=model_overwrite(model),
                           task="generate",
                           tokenizer=model,
                           tokenizer_mode="auto",
                           trust_remote_code=True,
                           dtype="auto",
                           seed=0)


def get_mlp_weights(layer_idx=0):
    gate_proj = torch.rand(config.intermediate_size,
                           config.hidden_size,
                           dtype=torch.float32,
                           device="cpu").to(torch.float8_e4m3fn)

    gate_proj_weight_scale_inv = torch.rand(
        config.intermediate_size // config.weight_block_size,
        config.hidden_size // config.weight_block_size,
        dtype=torch.float32,
        device="cpu")

    up_proj = torch.rand(config.intermediate_size,
                         config.hidden_size,
                         dtype=torch.float32,
                         device="cpu").to(torch.float8_e4m3fn)
    up_proj_weight_scale_inv = torch.rand(
        config.intermediate_size // config.weight_block_size,
        config.hidden_size // config.weight_block_size,
        dtype=torch.float32,
        device="cpu")

    down_proj_weight = torch.rand(config.hidden_size,
                                  config.intermediate_size,
                                  dtype=torch.float32,
                                  device="cpu").to(torch.float8_e4m3fn)

    down_proj_weight_scale_inv = torch.rand(
        config.hidden_size // config.weight_block_size,
        config.intermediate_size // config.weight_block_size,
        dtype=torch.float32,
        device="cpu")

    prefix = f"model.layers.{layer_idx}."

    weights = [
        (prefix + "mlp.down_proj.weight", down_proj_weight),
        (prefix + "mlp.down_proj.weight_scale_inv",
         down_proj_weight_scale_inv),
        (prefix + "mlp.gate_proj.weight", gate_proj),
        (prefix + "mlp.gate_proj.weight_scale_inv",
         gate_proj_weight_scale_inv),
        (prefix + "mlp.up_proj.weight", up_proj),
        (prefix + "mlp.up_proj.weight_scale_inv", up_proj_weight_scale_inv),
    ]

    return weights


def get_self_attn_weights(layer_idx=0):
    kv_a_proj_with_mqa = torch.rand(576,
                                    config.hidden_size,
                                    dtype=torch.float32,
                                    device="cpu").to(torch.float8_e4m3fn)

    kv_a_proj_with_mqa_scale_inv = torch.rand(5,
                                              56,
                                              dtype=torch.float32,
                                              device="cpu")

    kv_b_proj = torch.rand(32768, 512, dtype=torch.float32,
                           device="cpu").to(torch.float8_e4m3fn)
    kv_b_proj_scale_inv = torch.rand(256, 4, dtype=torch.float32, device="cpu")

    o_proj = torch.rand(config.hidden_size,
                        16384,
                        dtype=torch.float32,
                        device="cpu").to(torch.float8_e4m3fn)

    o_proj_scale_inv = torch.rand(56, 128, dtype=torch.float32, device="cpu")

    kv_a_layernorm = torch.rand(512, dtype=torch.bfloat16, device="cpu")

    q_a_layernorm = torch.rand(1536, dtype=torch.bfloat16, device="cpu")

    q_a_proj = torch.rand(1536,
                          config.hidden_size,
                          dtype=torch.float32,
                          device="cpu").to(torch.float8_e4m3fn)

    q_a_proj_scale_inv = torch.rand(12, 56, dtype=torch.float32, device="cpu")

    q_b_proj = torch.rand(24576, 1536, dtype=torch.float32,
                          device="cpu").to(torch.float8_e4m3fn)

    q_b_proj_scale_inv = torch.rand(192, 12, dtype=torch.float32, device="cpu")

    prefix = f"model.layers.{layer_idx}."

    weights = [
        (prefix + "self_attn.kv_a_layernorm.weight", kv_a_layernorm),
        (prefix + "self_attn.kv_a_proj_with_mqa.weight", kv_a_proj_with_mqa),
        (prefix + "self_attn.kv_a_proj_with_mqa.weight_scale_inv",
         kv_a_proj_with_mqa_scale_inv),
        (prefix + "self_attn.kv_b_proj.weight", kv_b_proj),
        (prefix + "self_attn.kv_b_proj.weight_scale_inv", kv_b_proj_scale_inv),
        (prefix + "self_attn.o_proj.weight", o_proj),
        (prefix + "self_attn.o_proj.weight_scale_inv", o_proj_scale_inv),
        (prefix + "self_attn.q_a_layernorm.weight", q_a_layernorm),
        (prefix + "self_attn.q_a_proj.weight", q_a_proj),
        (prefix + "self_attn.q_a_proj.weight_scale_inv", q_a_proj_scale_inv),
        (prefix + "self_attn.q_b_proj.weight", q_b_proj),
        (prefix + "self_attn.q_b_proj.weight_scale_inv", q_b_proj_scale_inv),
    ]
    return weights


def get_dense_layer_weights(layer_idx=0):
    prefix = f"model.layers.{layer_idx}."

    input_layernorm = torch.rand(config.hidden_size,
                                 dtype=torch.bfloat16,
                                 device="cpu")
    post_attention_layernorm = torch.rand(config.hidden_size,
                                          dtype=torch.bfloat16,
                                          device="cpu")

    layernorm_weights = [
        (prefix + "input_layernorm.weight", input_layernorm),
        (prefix + "post_attention_layernorm.weight", post_attention_layernorm),
    ]

    mlp_weights = get_mlp_weights(layer_idx)
    mha_weights = get_self_attn_weights(layer_idx)

    weights = mlp_weights + mha_weights + layernorm_weights

    return weights


def get_expert_weights(layer_idx=3, expert_idx=0):
    down_proj = torch.rand(config.hidden_size,
                           config.moe_intermediate_size,
                           dtype=torch.float32,
                           device="cpu").to(torch.float8_e4m3fn)

    down_proj_weight_scale_inv = torch.rand(
        config.hidden_size // config.weight_block_size,
        config.moe_intermediate_size // config.weight_block_size,
        dtype=torch.float32,
        device="cpu")

    gate_proj = torch.rand(config.moe_intermediate_size,
                           config.hidden_size,
                           dtype=torch.float32,
                           device="cpu").to(torch.float8_e4m3fn)
    gate_proj_weight_scale_inv = torch.rand(
        config.moe_intermediate_size // config.weight_block_size,
        config.hidden_size // config.weight_block_size,
        dtype=torch.float32,
        device="cpu")

    up_proj = torch.rand(config.moe_intermediate_size,
                         config.hidden_size,
                         dtype=torch.float32,
                         device="cpu").to(torch.float8_e4m3fn)

    up_proj_weight_scale_inv = torch.rand(
        config.moe_intermediate_size // config.weight_block_size,
        config.hidden_size // config.weight_block_size,
        dtype=torch.float32,
        device="cpu")

    prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."

    weights = [
        (prefix + "down_proj.weight", down_proj),
        (prefix + "down_proj.weight_scale_inv", down_proj_weight_scale_inv),
        (prefix + "gate_proj.weight", gate_proj),
        (prefix + "gate_proj.weight_scale_inv", gate_proj_weight_scale_inv),
        (prefix + "up_proj.weight", up_proj),
        (prefix + "up_proj.weight_scale_inv", up_proj_weight_scale_inv),
    ]

    return weights


def get_shared_experts_weights(layer_idx=3):
    down_proj = torch.rand(config.hidden_size,
                           config.moe_intermediate_size,
                           dtype=torch.float32,
                           device="cpu").to(torch.float8_e4m3fn)

    down_proj_weight_scale_inv = torch.rand(
        config.hidden_size // config.weight_block_size,
        config.moe_intermediate_size // config.weight_block_size,
        dtype=torch.float32,
        device="cpu")

    gate_proj = torch.rand(config.moe_intermediate_size,
                           config.hidden_size,
                           dtype=torch.float32,
                           device="cpu").to(torch.float8_e4m3fn)
    gate_proj_weight_scale_inv = torch.rand(
        config.moe_intermediate_size // config.weight_block_size,
        config.hidden_size // config.weight_block_size,
        dtype=torch.float32,
        device="cpu")

    up_proj = torch.rand(config.moe_intermediate_size,
                         config.hidden_size,
                         dtype=torch.float32,
                         device="cpu").to(torch.float8_e4m3fn)

    up_proj_weight_scale_inv = torch.rand(
        config.moe_intermediate_size // config.weight_block_size,
        config.hidden_size // config.weight_block_size,
        dtype=torch.float32,
        device="cpu")

    prefix = f"model.layers.{layer_idx}.mlp.shared_experts."

    weights = [
        (prefix + "down_proj.weight", down_proj),
        (prefix + "down_proj.weight_scale_inv", down_proj_weight_scale_inv),
        (prefix + "gate_proj.weight", gate_proj),
        (prefix + "gate_proj.weight_scale_inv", gate_proj_weight_scale_inv),
        (prefix + "up_proj.weight", up_proj),
        (prefix + "up_proj.weight_scale_inv", up_proj_weight_scale_inv),
    ]

    return weights


def get_moe_weights(layer_idx=3, wo_moe=False):
    gate = torch.rand(256, 7168, dtype=torch.bfloat16, device="cpu")

    e_score_correction_bias = torch.rand(256,
                                         dtype=torch.float32,
                                         device="cpu")

    prefix = f"model.layers.{layer_idx}.mlp."

    weights = [
        (prefix + "gate.weight", gate),
        (prefix + "gate.e_score_correction_bias", e_score_correction_bias),
    ]

    shared_experts_weights = get_shared_experts_weights(layer_idx)

    weights += shared_experts_weights

    if not wo_moe:
        for expert_idx in range(config.n_routed_experts):
            weights += get_expert_weights(layer_idx=layer_idx,
                                          expert_idx=expert_idx)

    return weights


def get_backbone_layer_weights(layer_idx=3, wo_moe=False):
    prefix = f"model.layers.{layer_idx}."

    input_layernorm = torch.rand(config.hidden_size,
                                 dtype=torch.bfloat16,
                                 device="cpu")
    post_attention_layernorm = torch.rand(config.hidden_size,
                                          dtype=torch.bfloat16,
                                          device="cpu")

    layernorm_weights = [
        (prefix + "input_layernorm.weight", input_layernorm),
        (prefix + "post_attention_layernorm.weight", post_attention_layernorm),
    ]

    mha_weights = get_self_attn_weights(layer_idx)

    weights = mha_weights + layernorm_weights

    if not wo_moe:
        mlp_weights = get_moe_weights(layer_idx, wo_moe)
        weights += mlp_weights

    return weights


def get_backbone_weights(wo_moe=False):
    norm = torch.rand(config.hidden_size, dtype=torch.bfloat16, device="cpu")

    weights = [('model.norm.weight', norm)]

    for i in range(config.num_hidden_layers):
        if i < config.first_k_dense_replace:
            weights += get_dense_layer_weights(i)
        else:
            weights += get_backbone_layer_weights(i, wo_moe=wo_moe)
    return weights


def load_weights(model: nn.Module, weights: Iterable[Tuple[str, torch.Tensor]],
                 prefix):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    params_dict = dict(model.named_parameters())

    params = set(name for name in params_dict.keys()
                 if "k_scale" not in name and "v_scale" not in name
                 and "q_scale" not in name and "q_scale")

    for name, loaded_weight in weights:
        try:

            name = name.replace(prefix, "")

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader

                weight_loader(param, loaded_weight, shard_id)
                params.discard(name)
                break

            else:
                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                params.remove(name)
        except Exception as e:
            print(name)
            print(param.shape, loaded_weight.shape)
            raise e

    assert len(params) == 0, str(params)


def load_fused_moe_weights(model: nn.Module,
                           weights: Iterable[Tuple[str,
                                                   torch.Tensor]], prefix):
    from vllm.model_executor.layers.fused_moe import FusedMoE

    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=hf_config.n_routed_experts)

    params_dict = dict(model.named_parameters())

    params = set(name for name in params_dict.keys()
                 if "k_scale" not in name and "v_scale" not in name
                 and "q_scale" not in name and "q_scale")

    for name, loaded_weight in weights:
        name = name.replace(prefix, "")

        for (param_name, weight_name, shard_id) in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue

            if (name.startswith("experts") and name not in params_dict):
                continue

            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)

            params.discard(name)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param,
                              loaded_weight,
                              name,
                              shard_id=shard_id,
                              expert_id=expert_id)
                params.discard(name)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                params.remove(name)

    assert len(params) == 0, str(params)


def offloading(model, device, non_blocking):
    device = torch.device(device)
    params_dict = dict(model.named_parameters())
    for name, param in params_dict.items():
        param.data = param.data.to(device, non_blocking=non_blocking)
