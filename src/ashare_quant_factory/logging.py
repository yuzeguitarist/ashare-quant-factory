from __future__ import annotations

import logging
import os
from typing import Optional

from rich.logging import RichHandler


def setup_logging(level: Optional[str] = None) -> None:
    """Configure structured console logging via Rich.

    Environment override:
      - AQF_LOG_LEVEL: DEBUG/INFO/WARNING/ERROR
    """
    env_level = os.getenv("AQF_LOG_LEVEL")
    log_level = (level or env_level or "INFO").upper()

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )

    # Reduce noise from dependencies
    logging.getLogger("apscheduler").setLevel(max(logging.INFO, logging.getLogger().level))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
