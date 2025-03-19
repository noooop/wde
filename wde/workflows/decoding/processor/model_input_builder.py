from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch
from vllm.utils import flatten_2d_lists, is_pin_memory_available

from wde.workflows.core.backends.attention import AttentionBackend
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.processor.model_input_builder import ModelInputBuilder
from wde.workflows.decoding.backends.sampling.sampling_metadata import \
    SamplingMetadata
from wde.workflows.decoding.schema.engine_io import (
    DecodingSchedulableRequest, DecodingSchedulerOutput)
from wde.workflows.decoding.schema.execute_io import (DecodingExecuteInput,
                                                      DecodingModelInput)

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
            [request.c_input_tokens for request in scheduled_requests]),
                                           dtype=torch.long,
                                           device="cpu",
                                           pin_memory=pin_memory)

        input_positions_tensor = torch.tensor(flatten_2d_lists(
            [request.c_input_positions for request in scheduled_requests]),
                                              dtype=torch.long,
                                              device="cpu",
                                              pin_memory=pin_memory)
        return input_tokens_tensor, input_positions_tensor

    def _prepare_intermediate_results(self,
                                      request: DecodingSchedulableRequest):

        token_len = request.get_token_len()
        token_chunk_size = request.token_chunk_size

        request.c_is_prefill = request.get_is_prefill()
        request.c_is_prompt = request.get_is_prompt()

        request.c_context_len = request.get_context_len()
        request.c_seq_len = request.get_seq_len()

        request.c_query_len = token_chunk_size

        if request.c_is_prefill:
            request.c_input_tokens = request.get_token_ids(
            )[request.c_context_len:request.c_seq_len]
        else:
            request.c_input_tokens = [request.get_last_token_id()]

        request.c_input_positions = list(
            range(request.c_context_len, request.c_seq_len))

        request.c_physical_block_ids = request.get_physical_block_ids()
        request.c_do_sample = token_len == request.c_context_len + token_chunk_size

        assert request.c_seq_len == request.c_context_len + request.c_query_len
        assert request.c_query_len == len(request.c_input_tokens) == len(
            request.c_input_positions)

    @torch.inference_mode
    def prepare_model_input(
        self, scheduled_requests: List[DecodingSchedulableRequest]
    ) -> DecodingModelInput:

        prefills = []
        decodes = []
        for request in scheduled_requests:
            self._prepare_intermediate_results(request)

            if request.c_is_prefill:
                prefills.append(request)
            else:
                decodes.append(request)

        # attn_metadata
        # prefills first
        scheduled_requests = prefills + decodes

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

    def __call__(
            self,
            scheduler_output: DecodingSchedulerOutput) -> DecodingExecuteInput:

        scheduled_requests = scheduler_output.scheduled_requests

        if scheduled_requests is None:
            return DecodingExecuteInput(worker_input=None, model_input=None)

        model_input = self.prepare_model_input(scheduled_requests)

        execute_input = DecodingExecuteInput(worker_input=None,
                                             model_input=model_input)

        execute_input.swap_out_task = getattr(scheduler_output,
                                              "swap_out_task", None)
        return execute_input
