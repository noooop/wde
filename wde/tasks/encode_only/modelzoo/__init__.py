TASK = "encode_only"
PREFIX = f"wde.tasks.{TASK}.modelzoo"
WORKFLOW = "wde.tasks.encode_only.workflow:EncodeOnlyWorkflow"

# Architecture -> (module, workflow).
ENCODE_ONLY_MODELS = {
    "XLMRobertaForMaskedLM":
    (PREFIX + ".xlm_roberta:XLMRobertaForMaskedLM", WORKFLOW),
    "BertForMaskedLM": (PREFIX + ".bert:BertForMaskedLM", WORKFLOW),
}
