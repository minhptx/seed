import logging

from pytorch_lightning.utilities import rank_zero_only
from loguru import logger
import sys

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    logger.add(sys.stdout, format="[{time}] [<level>{level}</level>] {message}")

    return logger