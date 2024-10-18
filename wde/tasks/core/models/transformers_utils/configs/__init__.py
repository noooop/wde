from wde.tasks.core.models.transformers_utils.configs.chatglm import \
    ChatGLMConfig
from wde.tasks.core.models.transformers_utils.configs.dbrx import DbrxConfig
# RWConfig is for the original tiiuae/falcon-40b(-instruct) and
# tiiuae/falcon-7b(-instruct) models. Newer Falcon models will use the
# `FalconConfig` class from the official HuggingFace transformers library.
from wde.tasks.core.models.transformers_utils.configs.falcon import RWConfig
from wde.tasks.core.models.transformers_utils.configs.internvl import \
    InternVLChatConfig
from wde.tasks.core.models.transformers_utils.configs.jais import JAISConfig
from wde.tasks.core.models.transformers_utils.configs.medusa import \
    MedusaConfig
from wde.tasks.core.models.transformers_utils.configs.mlp_speculator import \
    MLPSpeculatorConfig
from wde.tasks.core.models.transformers_utils.configs.mpt import MPTConfig
from wde.tasks.core.models.transformers_utils.configs.nemotron import \
    NemotronConfig

__all__ = [
    "ChatGLMConfig",
    "DbrxConfig",
    "MPTConfig",
    "RWConfig",
    "InternVLChatConfig",
    "JAISConfig",
    "MedusaConfig",
    "MLPSpeculatorConfig",
    "NemotronConfig",
]
