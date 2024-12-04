from typing import Any, Dict

from wde.workflows.core.processor.input_processor import TextRequestProcessor
from wde.workflows.core.schema.engine_io import RequestMetrics, TextRequest
from wde.workflows.decoding.backends.sampling.detokenizer import Tokenizer
from wde.workflows.decoding.config import ModelConfig
from wde.workflows.decoding.schema.engine_io import DecodingSchedulableRequest


class DecodingModelRequestProcessor(TextRequestProcessor):

    def __init__(self,
                 model_config: ModelConfig,
                 tokenizer: Tokenizer,
                 record_metrics=False):
        super().__init__(tokenizer)
        self.record_metrics = record_metrics
        self.eos_token_id = tokenizer.eos_token_id

        self.generation_config_fields = self._load_generation_config_dict(
            model_config)

    def _load_generation_config_dict(
            self, model_config: ModelConfig) -> Dict[str, Any]:
        from wde.backends.models.transformers_utils.config import \
            try_get_generation_config
        config = try_get_generation_config(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )

        if config is None:
            return {}

        return config.to_diff_dict()

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config.model_config, engine.tokenizer,
                   engine.engine_config.sys_config.record_metrics)

    def __call__(self, request: TextRequest) -> DecodingSchedulableRequest:
        schedulable_request = TextRequestProcessor.__call__(self, request)

        sampling_params = request.params
        sampling_params = sampling_params.clone()
        sampling_params.update_from_generation_config(
            self.generation_config_fields, self.eos_token_id)

        if self.record_metrics:
            metrics = RequestMetrics(arrival_ts=request.arrival_time)
        else:
            metrics = None

        return DecodingSchedulableRequest(
            request_id=request.request_id,
            arrival_time=request.arrival_time,
            metrics=metrics,
            prompt=schedulable_request.inputs.prompt,
            prompt_token_ids=schedulable_request.inputs.prompt_token_ids,
            sampling_params=sampling_params,
        )
