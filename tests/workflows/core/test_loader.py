import pytest
from vllm.utils import DeviceMemoryProfiler

from wde.tasks.encode_only.arg_utils import EncodeOnlyEngineArgs as EngineArgs
from wde.workflows.core.backends.attention import AttentionBackend
from wde.workflows.core.backends.loader.loader import (get_model_loader,
                                                       initialize_model)
from wde.workflows.core.config import EngineConfig
from wde.workflows.prefill_only.backends.attention.selector import (
    AttentionImpls, AttentionType, AttnBackend, _Backend)

MODELS = ["google-bert/bert-base-uncased"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float", "half", "bfloat16"])
@pytest.mark.parametrize("attn_type", ["DECODER"])
def test_loader(model: str, dtype: str, attn_type: str):
    attention_impls = AttentionImpls[dtype]
    for attention_impl in attention_impls:
        selected_backend = _Backend.backend_name_to_enum(attention_impl)
        backend_cls = AttnBackend.get_backend_cls(selected_backend)
        attn_type_enum = AttentionType.attn_type_name_to_enum(attn_type)
        attn_backend = backend_cls(attn_type_enum)

        engine_args = EngineArgs(model=model)
        engine_config = engine_args.create_engine_config()

        model_memory_usage = load_model(engine_config=engine_config,
                                        attn_backend=attn_backend)

        assert model_memory_usage > 0


def load_model(engine_config: EngineConfig, attn_backend: AttentionBackend):

    with DeviceMemoryProfiler() as m:
        loader = get_model_loader(engine_config.load_config)
        model = initialize_model(engine_config=engine_config,
                                 attn_backend=attn_backend)

        loader.load_model(model,
                          model_config=engine_config.model_config,
                          device_config=engine_config.device_config)

    model_memory_usage = m.consumed_memory
    return model_memory_usage
