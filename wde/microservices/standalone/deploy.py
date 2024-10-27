import yaml
from easydict import EasyDict as edict

from wde import envs
from wde.logger import init_logger
from wde.microservices.framework.zero_manager.client import ZeroManagerClient

logger = init_logger(__name__)


class Deploy:
    INFERENCE_ENGINE_CLASS = "wde.engine.zero_engine:ZeroEngine"

    def __init__(self, config_path):
        with open(config_path, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.manager_client = None

    def root_manager_init(self):
        self.manager_client = ZeroManagerClient(envs.ROOT_MANAGER_NAME)

    def verify(self):
        for protocol in ["retriever", "reranker"]:
            if protocol not in self.config:
                continue

            for i, config in enumerate(self.config[protocol]["models"]):
                if isinstance(config, str):
                    self.config[protocol]["models"][i] = {"model": config}

        self.config = edict(self.config)

    def model_init(self):
        for protocol in ["retriever", "reranker"]:
            if protocol not in self.config:
                continue

            for config in self.config[protocol]["models"]:
                engine_args = config.get("engine_args", {})
                engine_args["model"] = config.model
                out = self.manager_client.start(
                    name=config.model,
                    engine_kwargs={
                        "server_class": self.INFERENCE_ENGINE_CLASS,
                        "engine_args": engine_args
                    })
                logger.info("%s : %s", config.model, out)

    def __call__(self):
        self.verify()
        self.root_manager_init()
        self.model_init()
