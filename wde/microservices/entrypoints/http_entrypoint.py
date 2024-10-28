from multiprocess import Process

from wde.logger import init_logger

logger = init_logger(__name__)


class HttpEntrypoint(Process):

    def __init__(self, name, engine_class, engine_kwargs=None, **kwargs):
        super().__init__()

        self.name = name
        self.engine_class = engine_class
        self.engine_kwargs = engine_kwargs or {}

    def init(self):
        pass

    def run(self):
        import uvicorn

        host = self.engine_kwargs.get("host", "0.0.0.0")
        port = self.engine_kwargs.get("port", 8000)

        self.engine_kwargs["host"] = host
        self.engine_kwargs["port"] = port

        logger.info(
            f"HttpEntrypoints {self.name} running! host: {host}, port: {port}")

        try:
            uvicorn.run(self.engine_class, **self.engine_kwargs)
        except (KeyboardInterrupt, EOFError):
            pass
        logger.info("HttpEntrypoints clean_up!")

    def wait(self):
        try:
            self.join()
        except (KeyboardInterrupt, EOFError):
            self.terminate()
        finally:
            self.terminate()
