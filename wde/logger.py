import logging
from logging import Logger

import wde.envs as envs

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "wde": {
            "class": "wde.logging.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "wde": {
            "class": "logging.StreamHandler",
            "formatter": "wde",
            "level": envs.LOGGING_LEVEL,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "wde": {
            "handlers": ["wde"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "version": 1,
}


def init_logger(name: str) -> Logger:
    return logging.getLogger(name)