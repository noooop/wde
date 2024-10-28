from wde.tasks.encode_only.workflow import EncodeOnlyWorkflow
from wde.tasks.reranker.engine.schema import PROTOCOL


class RerankerWorkflow(EncodeOnlyWorkflow):
    InputProcessor: str = ("wde.tasks.reranker.processor."
                           "input_processor:RerankerInputProcessor")
    RequestProcessor: str = ("wde.tasks.reranker.processor."
                             "input_processor:RerankerRequestProcessor")
    OutputProcessor: str = ("wde.tasks.reranker.processor."
                            "output_processor:RerankerOutputProcessor")
    protocol: str = PROTOCOL
