from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch
from vllm.utils import flatten_2d_lists, is_pin_memory_available

from wde.workflows.core.backends.attention import AttentionBackend
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.processor.model_input_builder import ModelInputBuilder
from wde.workflows.core.schema.execute_io import ExecuteInput
from wde.workflows.decoding.backends.sampling.sampling_metadata import \
    SamplingMetadata
from wde.workflows.decoding.schema.engine_io import (
    DecodingSchedulableRequest, DecodingSchedulerOutput)
from wde.workflows.decoding.schema.execute_io import DecodingModelInput

pin_memory = is_pin_memory_available()


class DecodingModelInputBuilder(ModelInputBuilder):

    def __init__(
        self,
        engine_config: EngineConfig,
        attn_backend: AttentionBackend,
    ):
        self.engine_config = engine_config
        self.device = engine_config.device_config.device
        self.vocab_size = self.engine_config.model_config.hf_config.vocab_size

        self.attn_backend = attn_backend
        self.attn_metadata_builder = attn_backend.make_metadata_builder(
            self.engine_config.cache_config.block_size)
        self.threads = ThreadPoolExecutor(
            max_workers=self.engine_config.scheduler_config.max_num_on_the_fly)

        self.kv_caches = None

    @classmethod
    def from_engine(cls, engine):
        return cls(engine_config=engine.engine_config,
                   attn_backend=engine.attn_backend)

    def _prepare_model_tensor_input(
            self, scheduled_requests: List[DecodingSchedulableRequest]):
        input_tokens_tensor = torch.tensor(flatten_2d_lists(
            [request.input_tokens for request in scheduled_requests]),
                                           dtype=torch.long,
                                           device="cpu",
                                           pin_memory=pin_memory)

        input_positions_tensor = torch.tensor(flatten_2d_lists(
            [request.input_positions for request in scheduled_requests]),
                                              dtype=torch.long,
                                              device="cpu",
                                              pin_memory=pin_memory)
        return input_tokens_tensor, input_positions_tensor

    def _prepare_intermediate_results(self,
                                      request: DecodingSchedulableRequest):

        token_len = request.get_len()
        token_chunk_size = request.token_chunk_size
        request.is_prefill_cached = request.is_prefill
        request.context_len = request.num_computed_tokens
        request.seq_len = min(token_len,
                              request.context_len + token_chunk_size)
        request.query_len = token_chunk_size

        if request.is_prefill_cached:
            request.input_tokens = request.get_token_ids(
            )[request.context_len:request.seq_len]
        else:
            request.input_tokens = [request.get_last_token_id()]

        request.input_positions = list(
            range(request.context_len, request.seq_len))
        if request.vblock is None:
            # profile_run
            request.physical_block_ids = []
        else:
            request.physical_block_ids = request.vblock.physical_block_ids
        request.do_sample = token_len == request.context_len + token_chunk_size

    @torch.inference_mode
    def prepare_model_input(
        self, scheduled_requests: List[DecodingSchedulableRequest]
    ) -> DecodingModelInput:
        for request in scheduled_requests:
            self._prepare_intermediate_results(request)

        input_tokens_tensor, input_positions_tensor = self._prepare_model_tensor_input(
            scheduled_requests)
        attn_metadata = self.attn_metadata_builder(scheduled_requests)

        sampling_metadata = self.threads.submit(SamplingMetadata.prepare,
                                                scheduled_requests,
                                                vocab_size=self.vocab_size,
                                                dtype=torch.float)

        return DecodingModelInput(input_tokens=input_tokens_tensor,
                                  input_positions=input_positions_tensor,
                                  attn_metadata=attn_metadata,
                                  sampling_metadata=sampling_metadata,
                                  kv_caches=self.kv_caches)

    def __call__(self,
                 scheduler_output: DecodingSchedulerOutput) -> ExecuteInput:

        scheduled_requests = scheduler_output.scheduled_requests

        if scheduled_requests is None:
            return ExecuteInput(worker_input=None, model_input=None)

        model_input = self.prepare_model_input(scheduled_requests)
        return ExecuteInput(worker_input=None, model_input=model_input)
