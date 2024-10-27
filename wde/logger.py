import logging
from logging import Logger

from wde import envs

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


def _configure_root_logger() -> None:
    logging.basicConfig(
        level=envs.LOGGING_LEVEL,
        format=_FORMAT,
        datefmt=_DATE_FORMAT,
    )


def init_logger(name: str) -> Logger:
    return logging.getLogger(name)


_configure_root_logger()
