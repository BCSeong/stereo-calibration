from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_LOGGER_NAME = 'calib_v3'
_logger: Optional[logging.Logger] = None


def init_logger(log_path: Path, console_level: int = logging.INFO, file_level: int = logging.DEBUG) -> logging.Logger:
    global _logger
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to allow re-init
    while logger.handlers:
        logger.handlers.pop()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # File handler (DEBUG+)
    fh = logging.FileHandler(str(log_path), mode='w', encoding='utf-8')
    fh.setLevel(file_level)
    ffmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    fh.setFormatter(ffmt)
    logger.addHandler(fh)

    # Console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    cfmt = logging.Formatter('%(message)s')
    ch.setFormatter(cfmt)
    logger.addHandler(ch)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    # Fallback basic logger if not initialized
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
    _logger = logger
    return logger


