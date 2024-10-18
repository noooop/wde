TASK = "retriever"
RETRIEVER_ENCODER_ONLY_PREFIX = f"wde.tasks.{TASK}.modelzoo"
RETRIEVER_ENCODER_ONLY_WORKFLOW = ("wde.tasks.retriever.workflow:"
                                   "RetrieverEncodeOnlyWorkflow")

# Architecture -> (module, workflow).
RETRIEVER_ENCODER_ONLY_MODELS = {
    "XLMRobertaModel": (RETRIEVER_ENCODER_ONLY_PREFIX + ".bge_m3:BGEM3Model",
                        RETRIEVER_ENCODER_ONLY_WORKFLOW),
    "BertModel":
    (RETRIEVER_ENCODER_ONLY_PREFIX + ".bert_retriever:BertRetriever",
     RETRIEVER_ENCODER_ONLY_WORKFLOW),
}

RETRIEVER_DECODER_ONLY_WORKFLOW = ("wde.tasks.retriever.workflow:"
                                   "RetrieverDecodeOnlyWorkflow")

# Architecture -> (module, workflow).
RETRIEVER_DECODER_ONLY_MODELS = {}

RETRIEVER_MODELS = {
    **RETRIEVER_ENCODER_ONLY_MODELS,
    **RETRIEVER_DECODER_ONLY_MODELS
}
