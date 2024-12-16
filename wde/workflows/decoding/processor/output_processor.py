from typing import List

from wde.workflows.core.processor.output_processor import OutputProcessor
from wde.workflows.decoding.backends.sampling.sampler import (
    get_sample, get_sampler_output)
from wde.workflows.decoding.backends.sampling.stop_checker import StopChecker
from wde.workflows.decoding.schema.engine_io import (DecodingRequestOutput,
                                                     DecodingSchedulerOutput)
from wde.workflows.decoding.schema.execute_io import Sample, SamplerOutput
from wde.workflows.decoding.schema.request import DecodingSchedulableRequest


class DecodingModelOutputProcessor(OutputProcessor):

    def __init__(self, scheduler_config, scheduler, tokenizer):
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.stop_checker = StopChecker(scheduler_config.max_model_len,
                                        tokenizer)

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config.scheduler_config, engine.scheduler,
                   engine.tokenizer)

    def decode_and_stop_check(self, request: DecodingSchedulableRequest,
                              sample: Sample):
        sampling_params = request.sampling_params
        assert sampling_params.n == 1

        request.append_token_id(sample.output_token, sample.logprobs)

        if sampling_params.detokenize and self.tokenizer:
            new_char_count = self.tokenizer.decode_inplace(
                request, sampling_params)
        else:
            new_char_count = 0

        self.stop_checker.maybe_stop(request, new_char_count, sampling_params)

    def __call__(self, scheduler_output: DecodingSchedulerOutput,
                 execute_output: SamplerOutput) -> List[DecodingRequestOutput]:
        scheduled_requests = scheduler_output.scheduled_requests
        ignored_requests = scheduler_output.ignored_requests

        output_token_dict = get_sampler_output(execute_output)
        logprobs = execute_output.logprobs

        for request in scheduled_requests:
            request.update_num_computed_tokens()

            if not request.do_sample:
                continue

            output_token = output_token_dict[request.request_id]
            sample = get_sample(request, output_token, logprobs)

            self.decode_and_stop_check(request, sample)

        request_outputs = []
        for request in scheduled_requests:
            request_output = DecodingRequestOutput.from_request(request)
            request_outputs.append(request_output)
        for request in ignored_requests:
            request_output = DecodingRequestOutput.from_request(request)
            request_outputs.append(request_output)
        return request_outputs
