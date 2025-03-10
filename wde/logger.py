import logging
from functools import lru_cache
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
logger = init_logger(__name__)


@lru_cache
def print_info_once(msg: str) -> None:
    # Set the stacklevel to 2 to print the caller's line info
    logger.info(msg, stacklevel=2)


@lru_cache
def print_warning_once(msg: str) -> None:
    # Set the stacklevel to 2 to print the caller's line info
    logger.warning(msg, stacklevel=2)
