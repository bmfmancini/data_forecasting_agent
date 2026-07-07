"""Logging configuration for the Data Forecaster backend.

Configures the root logger with a rotating file handler and a console
handler.  Call :func:`get_logger` to obtain a named logger that inherits
this configuration.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR: str = "logs"
LOG_FILE: str = os.path.join(LOG_DIR, "app.log")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT: int = 5


def configure_logging() -> None:
    """Configure the root logger with file and console handlers.

    Creates the log directory if it does not exist, sets the root logger
    level to ``INFO``, and attaches a :class:`RotatingFileHandler` and a
    :class:`logging.StreamHandler` with the project format.  Safe to call
    multiple times — existing handlers are not duplicated.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.handlers:
        return

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging is configured.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance with project handlers attached.
    """
    configure_logging()
    return logging.getLogger(name)
