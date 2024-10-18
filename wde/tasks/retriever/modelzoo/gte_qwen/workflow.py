from typing import Dict

from wde.tasks.decode_only.workflow import DecodeOnlyWorkflow, Workflow


class Qwen2Workflow(Workflow):

    @classmethod
    def from_engine_args(cls, engine_args: Dict):
        # gte-Qwen2 and Qwen2 use the same architecture name，Qwen2ForCausalLM.
        # gte-Qwen2 family may have multiple different architectures.
        # gte-Qwen2-1.5B-instruct, does not use enable bidirectional.
        #     I'm not sure if this is a bug
        # gte-Qwen2-7B-instruct use enable bidirectional
        if "gte-Qwen2-1.5B-instruct" in engine_args["model"]:
            engine_args["output_last_hidden_states"] = True
        elif "gte-Qwen2-7B-instruct" in engine_args["model"]:
            engine_args["output_last_hidden_states"] = True
            engine_args["enable_bidirectional"] = True

        return DecodeOnlyWorkflow.from_engine_args(engine_args)
