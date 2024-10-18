TASK = "reranker"
PREFIX = f"wde.tasks.{TASK}.modelzoo"
WORKFLOW = "wde.tasks.reranker.workflow:RerankerWorkflow"

# Architecture -> (module, workflow).
RERANKER_MODELS = {
    "XLMRobertaForSequenceClassification":
    (PREFIX + ".bge_reranker_v2_m3:BGERerankerV2M3", WORKFLOW),
}
