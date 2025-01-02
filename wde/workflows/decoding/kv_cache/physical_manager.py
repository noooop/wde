import gc
from typing import List, Tuple

import torch
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size,
                        is_pin_memory_available)

from wde.logger import init_logger
from wde.workflows.core.backends.utils import set_random_seed
from wde.workflows.core.schema.execute_io import ExecuteInput
from wde.workflows.decoding.backends.sampling.sampling_params import \
    SamplingParams
from wde.workflows.decoding.schema.request import DecodingSchedulableRequest

logger = init_logger(__name__)

_GB = float(2**30)


class PhysicalGPUKVCacheManager:

    def __init__(self, engine_config, worker, model_inputs_builder,
                 attn_backend):
        self.engine_config = engine_config
        self.worker = worker
        self.model_inputs_builder = model_inputs_builder
        self.init_gpu_memory = worker.init_gpu_memory
        self.attn_backend = attn_backend

        self.determine_num_available_blocks()
        self.initialize_cache()

        self.gpu_cache = None

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config, engine.executor.worker,
                   engine.model_inputs_builder, engine.attn_backend)

    def determine_num_available_blocks(self) -> None:
        num_gpu_blocks, num_cpu_blocks = (
            self._determine_num_available_blocks())

        if self.engine_config.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = self.engine_config.cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.engine_config.cache_config.num_gpu_blocks = num_gpu_blocks
        self.engine_config.cache_config.num_cpu_blocks = num_cpu_blocks

    def initialize_cache(self) -> None:
        num_gpu_blocks = self.engine_config.cache_config.num_gpu_blocks
        num_cpu_blocks = self.engine_config.cache_config.num_cpu_blocks

        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        raise_if_cache_size_invalid(
            num_gpu_blocks, self.engine_config.cache_config.block_size,
            self.engine_config.model_config.max_model_len)

        self.gpu_cache = self._allocate_kv_cache(
            num_gpu_blocks, self.engine_config.device_config.device_type)

        set_random_seed(self.engine_config.model_config.seed)
        self.model_inputs_builder.kv_caches = self.gpu_cache

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:

        model_config = self.engine_config.model_config
        cache_config = self.engine_config.cache_config

        block_size = cache_config.block_size

        head_size = model_config.get_head_size()
        num_attention_layers = model_config.get_num_attention_layers()
        num_kv_heads = model_config.get_num_kv_heads()

        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size)

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        logger.info(
            f"KV cache shape:{(num_attention_layers, num_blocks, block_size, num_kv_heads, head_size, dtype)}."
        )

        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(num_attention_layers):
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def _get_cache_block_size_bytes(self) -> int:
        model_config = self.engine_config.model_config
        cache_config = self.engine_config.cache_config

        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads()
        num_attention_layers = model_config.get_num_attention_layers()

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total

    @torch.inference_mode()
    def _determine_num_available_blocks(self) -> Tuple[int, int]:
        logger.info('init gpu free memory %.4f GB', self.init_gpu_memory / _GB)

        torch.cuda.empty_cache()

        free_gpu_memory = torch.cuda.mem_get_info()[0]

        model_memory_usage = self.init_gpu_memory - free_gpu_memory

        logger.info("Loading model weights took %.4f GB",
                    model_memory_usage / _GB)

        self._profile_run()

        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()

        peak_memory = self.init_gpu_memory - free_gpu_memory

        runtime_memory = peak_memory - model_memory_usage

        if self.engine_config.scheduler_config.scheduling in ["async"]:
            pass
            # peak_memory += runtime_memory
            # runtime_memory *= 2

        assert peak_memory > 0, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_gpu_memory}, current free memory"
            f" {free_gpu_memory}. This happens when the GPU memory was "
            "not properly cleaned up before initializing the wde instance.")

        logger.info(
            "After profile run: Peak Memory %.4f GB, "
            "of which Runtime Memory %.4f GB, "
            "%.4f GB leave for KV cache", peak_memory / _GB,
            runtime_memory / _GB,
            (total_gpu_memory *
             self.engine_config.cache_config.gpu_memory_utilization -
             peak_memory) / _GB)

        cache_block_size = self._get_cache_block_size_bytes()

        logger.info(f"Cache block size: {cache_block_size}")

        num_gpu_blocks = int(
            (total_gpu_memory *
             self.engine_config.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(
            self.engine_config.cache_config.swap_space_bytes //
            cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    @torch.inference_mode()
    def _profile_run(self) -> None:
        model_config = self.engine_config.model_config
        vocab_size = model_config.get_vocab_size()
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.engine_config.scheduler_config.max_num_batched_tokens
        max_num_requests = self.engine_config.scheduler_config.max_num_requests

        requests: List[DecodingSchedulableRequest] = []
        batch_size = 0
        for requese_id in range(max_num_requests):
            seq_len = (
                max_num_batched_tokens // max_num_requests +
                (requese_id < max_num_batched_tokens % max_num_requests))
            batch_size += seq_len

            request = DecodingSchedulableRequest(
                request_id=str(requese_id),
                arrival_time=0.,
                prompt_token_ids=[0] * seq_len,
                sampling_params=sampling_params,
                vblock=None,
                token_chunk_size=seq_len)
            requests.append(request)

        # Run the model with the dummy inputs.
        num_layers = model_config.get_num_layers()
        kv_caches = [None] * num_layers
        model_input = self.model_inputs_builder.prepare_model_input(requests)
        model_input.to("cuda")
        model_input.deferred_to("cuda")
        model_input.kv_caches = kv_caches

        self.worker.runner.execute_model(
            ExecuteInput(worker_input=None, model_input=model_input))
        torch.cuda.synchronize()
        return


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
