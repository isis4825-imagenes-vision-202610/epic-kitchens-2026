"""Checkpoint save/load/resume utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside a model checkpoint."""

    epoch: int
    global_step: int
    avg_map: float
    avg_ndcg: float
    config: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: CheckpointMetadata,
    path: str | Path,
) -> Path:
    """Save model checkpoint with optimizer state and metadata.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        metadata: Checkpoint metadata.
        path: File path to save the checkpoint.

    Returns:
        The path where the checkpoint was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": asdict(metadata),
    }
    torch.save(state, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> CheckpointMetadata:
    """Load a checkpoint and restore model/optimizer state.

    Args:
        path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to restore state.
        map_location: Device mapping for loading.

    Returns:
        The checkpoint metadata.
    """
    checkpoint: dict[str, Any] = torch.load(
        path, map_location=map_location, weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    meta: dict[str, Any] = checkpoint.get("metadata", {})
    return CheckpointMetadata(
        epoch=meta.get("epoch", 0),
        global_step=meta.get("global_step", 0),
        avg_map=meta.get("avg_map", 0.0),
        avg_ndcg=meta.get("avg_ndcg", 0.0),
        config=meta.get("config", {}),
    )


def get_best_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the checkpoint with the highest avg_map in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files.

    Returns:
        Path to the best checkpoint, or None if no checkpoints found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    best_path: Path | None = None
    best_map = -1.0

    for ckpt_path in checkpoint_dir.glob("*.pt"):
        try:
            checkpoint: dict[str, Any] = torch.load(
                ckpt_path, map_location="cpu", weights_only=False,
            )
            meta: dict[str, Any] = checkpoint.get("metadata", {})
            avg_map: float = meta.get("avg_map", 0.0)
            if avg_map > best_map:
                best_map = avg_map
                best_path = ckpt_path
        except Exception:  # noqa: BLE001
            continue

    return best_path
