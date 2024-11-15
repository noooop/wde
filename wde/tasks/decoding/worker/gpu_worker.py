"""A GPU worker class."""
import gc
import os
from typing import List, Optional, Tuple

import torch
from vllm.platforms import current_platform

from wde.backends.utils import set_random_seed
from wde.logger import init_logger
from wde.tasks.core.config import DeviceConfig, LoadConfig
from wde.tasks.core.schema.execute_io import ExecuteInput, ExecuteOutput
from wde.tasks.decoding.backends.attention import DecodeOnlyAttentionBackend
from wde.tasks.decoding.backends.core.cache_engine import CacheEngine
from wde.tasks.decoding.config import (CacheConfig, DecodingEngineConfig,
                                       DecodingModelConfig,
                                       DecodingSchedulerConfig)
from wde.tasks.decoding.runner.model_runner import GPUModelRunner
from wde.tasks.decoding.schema.execute_io import (DecodingExecuteInput,
                                                  SamplerOutput)

logger = init_logger(__name__)

_GB = float(2**30)


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        engine_config: DecodingEngineConfig,
        attn_backend: DecodeOnlyAttentionBackend,
    ) -> None:
        self.model_config: DecodingModelConfig = engine_config.model_config
        self.scheduler_config: DecodingSchedulerConfig = engine_config.scheduler_config
        self.device_config: DeviceConfig = engine_config.device_config
        self.cache_config: CacheConfig = engine_config.cache_config
        self.load_config: LoadConfig = engine_config.load_config
        self.attn_backend = attn_backend

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.model_runner = GPUModelRunner(
            self.model_config,
            self.scheduler_config,
            self.device_config,
            self.cache_config,
            load_config=self.load_config,
            attn_backend=attn_backend,
            kv_cache_dtype=self.cache_config.cache_dtype,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: CacheEngine
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: Optional[List[torch.Tensor]] = None

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")

        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        logger.info('init gpu free memory %.4f GB', self.init_gpu_memory / _GB)

        torch.cuda.empty_cache()

        free_gpu_memory = torch.cuda.mem_get_info()[0]

        model_memory_usage = self.init_gpu_memory - free_gpu_memory

        logger.info("Loading model weights took %.4f GB",
                    model_memory_usage / _GB)

        self.model_runner.profile_run()

        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()

        peak_memory = self.init_gpu_memory - free_gpu_memory

        runtime_memory = peak_memory - model_memory_usage

        if self.scheduler_config.scheduling in ["async", "double_buffer"]:
            pass
            #peak_memory += runtime_memory
            #runtime_memory *= 2

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
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) / _GB)

        cache_block_size = self.get_cache_block_size_bytes()
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.device_config, self.attn_backend)
        self.gpu_cache = self.cache_engine.gpu_cache

    def _warm_up_model(self) -> None:
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @property
    def kv_cache(self) -> Optional[List[torch.Tensor]]:
        return self.gpu_cache

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config)

    @torch.inference_mode
    def __call__(
        self,
        execute_input: Optional[DecodingExecuteInput] = None
    ) -> Optional[SamplerOutput]:

        if execute_input.worker_input.num_requests == 0:
            return

        output = self.model_runner.execute_model(
            execute_input.model_input,
            self.kv_cache if self.kv_cache is not None else None)

        return output

    def non_blocking_h2d(self, execute_input: ExecuteInput):
        # worker_input
        worker_input = execute_input.worker_input
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine.swap_in(worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine.swap_out(worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine.copy(worker_input.blocks_to_copy)

        # model_input
        model_input = execute_input.model_input
        model_input.to("cuda", non_blocking=True)

    def non_blocking_d2h(self, execute_output: ExecuteOutput):
        execute_output.to("cpu", non_blocking=True)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = current_platform.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


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
