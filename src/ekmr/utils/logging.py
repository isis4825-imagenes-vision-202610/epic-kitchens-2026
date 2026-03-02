"""Structured logging utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def setup_logging(log_dir: str | Path | None = None, rank: int = 0) -> None:
    """Configure structured logging with loguru.

    Args:
        log_dir: Optional directory for log files.
        rank: Process rank for distributed training.
    """
    from loguru import logger

    logger.remove()
    if rank == 0:
        logger.add(
            sys.stderr,
            level="INFO",
            format="{time:HH:mm:ss} | {level:<7} | {message}",
        )
        if log_dir is not None:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            logger.add(
                str(log_path / "train.log"),
                level="DEBUG",
                format="{time} | {level} | {message}",
                rotation="100 MB",
            )


def log_metrics(metrics: dict[str, Any], step: int, prefix: str = "") -> None:
    """Log metrics to console.

    Args:
        metrics: Dictionary of metric names to values.
        step: Current step or epoch number.
        prefix: Optional prefix for metric names.
    """
    from loguru import logger

    parts = [
        f"{prefix}/{k}: {v:.4f}" if prefix else f"{k}: {v:.4f}"
        for k, v in metrics.items()
    ]
    logger.info(f"Step {step} | {' | '.join(parts)}")
