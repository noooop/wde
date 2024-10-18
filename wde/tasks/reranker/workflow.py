from wde.tasks.encode_only.workflow import EncodeOnlyWorkflow


class RerankerWorkflow(EncodeOnlyWorkflow):
    InputProcessor: str = ("wde.tasks.reranker.processor."
                           "input_processor:RerankerInputProcessor")
    RequestProcessor: str = ("wde.tasks.reranker.processor."
                             "input_processor:RerankerRequestProcessor")
    OutputProcessor: str = ("wde.tasks.reranker.processor."
                            "output_processor:RerankerOutputProcessor")
