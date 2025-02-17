from typing import List, Optional, Sequence, Union, cast

from tqdm import tqdm
from vllm.utils import Counter

from wde.logger import init_logger
from wde.tasks.reranker.schema.engine_io import RerankerInputs
from wde.workflows.core.llm_engine import LLMEngine
from wde.workflows.core.schema.engine_io import (Params, PromptInputs,
                                                 RequestOutput)
from wde.workflows.decoding import SamplingParams
from wde.workflows.decoding.schema.engine_io import DecodingRequestOutput

logger = init_logger(__name__)


class LLM:

    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        seed: int = 0,
        **kwargs,
    ) -> None:
        engine_args = dict(
            model=model,
            dtype=dtype,
            seed=seed,
            **kwargs,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

    def generate(
        self,
        inputs: Union[Union[PromptInputs, Sequence[PromptInputs]],
                      Optional[Union[str, List[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:

        inputs = cast(Union[PromptInputs, Sequence[PromptInputs]], inputs)

        if sampling_params is None:
            sampling_params = SamplingParams()

        self._validate_and_add_requests(inputs=inputs, params=sampling_params)

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return outputs

    def encode(
        self,
        inputs: Union[Union[PromptInputs, Sequence[PromptInputs]]],
        pooling_params: Optional[Union[Params, Sequence[Params]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:

        if pooling_params is None:
            pooling_params = Params()

        self._validate_and_add_requests(
            inputs=inputs,
            params=pooling_params,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return outputs

    def compute_score(
        self,
        inputs: RerankerInputs,
        params: Optional[Union[Params, Sequence[Params]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        for i, request_inputs in enumerate(inputs):
            self._add_request(
                request_inputs,
                params[i] if isinstance(params, Sequence) else params)

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return outputs

    def _validate_and_add_requests(
        self,
        inputs: Union[Union[PromptInputs, Sequence[PromptInputs]]],
        params: Optional[Union[Params, Sequence[Params]]] = None,
    ) -> None:
        if isinstance(inputs, (str, dict)):
            inputs = [inputs]

        num_requests = len(inputs)

        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")

        # Add requests to the engine.
        for i, request_inputs in enumerate(inputs):
            self._add_request(
                request_inputs,
                params[i] if isinstance(params, Sequence) else params)

    def _add_request(
        self,
        inputs: Union[Union[PromptInputs, Sequence[PromptInputs]]],
        params: Params,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.engine.add_request(request_id, inputs, params)

    def _run_engine(self, *, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )
        # Run the engine.
        outputs: List[RequestOutput] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, DecodingRequestOutput):
                            # Calculate tokens only for RequestOutput
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids)
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            out_spd = (total_out_toks /
                                       pbar.format_dict["elapsed"])
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s")
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))
