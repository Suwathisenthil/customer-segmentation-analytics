"""
logger.py - Centralised logging configuration.
All modules import get_logger() instead of using print().
"""
import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str, log_file: str = "outputs/pipeline.log",
               level: str = "INFO") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:          # prevent duplicate handlers on re-import
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file (5 MB × 3 backups)
    fh = RotatingFileHandler(log_file, maxBytes=5_242_880, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
