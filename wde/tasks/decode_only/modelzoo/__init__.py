TASK = "decode_only"
PREFIX = f"wde.tasks.{TASK}.modelzoo"
WORKFLOW = "wde.tasks.decode_only.workflow:DecodeOnlyWorkflow"

# Architecture -> (module, workflow).
DECODE_ONLY_MODELS = {
    "Qwen2ForCausalLM":
    (PREFIX + ".qwen2:Qwen2ForCausalLM",
     "wde.tasks.retriever.modelzoo.gte_qwen.workflow:Qwen2Workflow"),
}
