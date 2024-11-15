from typing import List

from wde.tasks.core.processor.output_processor import OutputProcessor
from wde.tasks.decoding.backends.sampler import (get_logprobs,
                                                 get_pythonized_sample_results)
from wde.tasks.decoding.scheduler import DecodingSchedulerOutput
from wde.tasks.decoding.schema.engine_io import DecodingRequestOutput
from wde.tasks.decoding.schema.execute_io import (
    CompletionSequenceGroupOutput, SamplerOutput, SequenceOutput)


class DecodingModelOutputProcessor(OutputProcessor):

    def __init__(self, scheduler_config, scheduler, tokenizer, seq_counter):
        from wde.tasks.decoding.backends.processor.single_step import \
            SingleStepOutputProcessor
        from wde.tasks.decoding.backends.processor.stop_checker import \
            StopChecker
        self.scheduler = scheduler

        self.output_processor = SingleStepOutputProcessor(
            tokenizer,
            seq_counter,
            stop_checker=StopChecker(scheduler_config.max_model_len,
                                     tokenizer),
            max_model_len=scheduler_config.max_model_len,
        )

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config.scheduler_config, engine.scheduler,
                   engine.tokenizer, engine.seq_counter)

    def get_sampler_output(self, execute_output: SamplerOutput):

        sampling_metadata = execute_output.sampling_metadata
        logprobs = execute_output.logprobs

        sample_results = get_pythonized_sample_results(execute_output)

        prompt_logprobs, sample_logprobs = get_logprobs(
            logprobs, sampling_metadata, sample_results)

        sampler_output: List[List[CompletionSequenceGroupOutput]] = []

        for (seq_group, sample_result, group_prompt_logprobs,
             group_sample_logprobs) in zip(sampling_metadata.seq_groups,
                                           sample_results, prompt_logprobs,
                                           sample_logprobs):
            seq_ids = seq_group.seq_ids
            next_token_ids, parent_ids = sample_result
            seq_outputs: List[SequenceOutput] = []
            for parent_id, next_token_id, logprobs in zip(
                    parent_ids, next_token_ids, group_sample_logprobs):
                seq_outputs.append(
                    SequenceOutput(seq_ids[parent_id], next_token_id,
                                   logprobs))
            sampler_output.append([
                CompletionSequenceGroupOutput(seq_outputs,
                                              group_prompt_logprobs)
            ])

        return sampler_output

    def __call__(self, scheduler_output: DecodingSchedulerOutput,
                 execute_output: SamplerOutput) -> List[DecodingRequestOutput]:
        scheduled_requests = scheduler_output.scheduled_requests
        ignored_requests = scheduler_output.ignored_requests
        seq_group_metadata_list = scheduler_output.seq_group_metadata_list

        sampler_output = self.get_sampler_output(execute_output)

        for request, outputs, seq_group_meta in zip(scheduled_requests,
                                                    sampler_output,
                                                    seq_group_metadata_list):
            seq_group = request.seq_group
            seq_group.update_num_computed_tokens(request.token_chunk_size)

            self.output_processor.process_prompt_logprob(seq_group, outputs)
            if seq_group_meta.do_sample:
                seq_need_fork, seq_need_free = self.output_processor.process_outputs(
                    seq_group, outputs)

                for parent, seq in seq_need_fork:
                    self.scheduler.fork_seq(parent, seq)
                for seq in seq_need_free:
                    self.scheduler.free_seq(seq)

        # Create the outputs.
        request_outputs = []
        for request in scheduled_requests:
            request_output = DecodingRequestOutput.from_seq_group(request)
            request_outputs.append(request_output)
        for request in ignored_requests:
            request_output = DecodingRequestOutput.from_seq_group(request)
            request_outputs.append(request_output)
        return request_outputs
