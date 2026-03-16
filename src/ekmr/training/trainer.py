"""Training loop for video-text retrieval models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from ekmr.utils.checkpoints import CheckpointMetadata, save_checkpoint
from ekmr.utils.distributed import is_main_process
from ekmr.utils.logging import log_metrics


class Trainer:
    """Training manager for retrieval models.

    Handles the training loop with AMP, gradient clipping, logging,
    and checkpoint management.

    Args:
        model: The retrieval model to train.
        optimizer: Optimizer for model parameters.
        scheduler: Learning rate scheduler.
        loss_fn: Loss function.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        config: Training configuration dictionary.
        output_dir: Directory for saving outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        loss_fn: nn.Module,
        train_loader: DataLoader[dict[str, Any]],
        val_loader: DataLoader[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
        output_dir: str | Path = "experiments/default",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amp_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._amp_device = amp_device
        self.scaler = torch.amp.GradScaler(amp_device, enabled=self.config.get("amp", True) and torch.cuda.is_available())
        self.grad_clip = self.config.get("grad_clip", 1.0)
        self.global_step = 0
        # max_steps: if set to a positive integer, training stops after that many
        # batches per epoch (useful for fast smoke tests).
        max_steps_cfg = self.config.get("training", {}).get("max_steps", None)
        self.max_steps: int | None = int(max_steps_cfg) if max_steps_cfg is not None else None

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary with training metrics for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(self.train_loader):
            if self.max_steps is not None and step >= self.max_steps:
                break
            step_metrics = self._train_step(batch)
            total_loss += step_metrics["loss"]
            num_batches += 1
            self.global_step += 1

            if is_main_process() and self.global_step % 50 == 0:
                log_metrics(step_metrics, self.global_step, prefix="train")

        avg_loss = total_loss / max(num_batches, 1)
        if self.scheduler is not None:
            self.scheduler.step()

        return {"epoch": float(epoch), "avg_loss": avg_loss}

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Batch dictionary from the data loader.

        Returns:
            Dictionary with step metrics.
        """
        start_time = time.time()
        self.optimizer.zero_grad()

        video = batch["video"].to(self.device, non_blocking=True)
        text = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.get("text", {}).items()}

        relevancy = batch.get("relevancy")
        if relevancy is not None:
            relevancy = relevancy.to(self.device, non_blocking=True)

        with torch.amp.autocast(self._amp_device, enabled=self.config.get("amp", True) and torch.cuda.is_available()):
            video_embeds, text_embeds = self.model(video, text)
            if relevancy is not None:
                loss = self.loss_fn(video_embeds, text_embeds, relevancy)
            else:
                b = video_embeds.size(0)
                identity_rel = torch.eye(b, device=self.device)
                loss = self.loss_fn(video_embeds, text_embeds, identity_rel)

        self.scaler.scale(loss).backward()

        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
        else:
            grad_norm = torch.tensor(0.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        step_time = time.time() - start_time
        return {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "lr": self.optimizer.param_groups[0]["lr"],
            "step_time_s": step_time,
        }

    def save(self, epoch: int, avg_map: float = 0.0, avg_ndcg: float = 0.0) -> Path:
        """Save a checkpoint.

        Args:
            epoch: Current epoch number.
            avg_map: Average mAP metric.
            avg_ndcg: Average nDCG metric.

        Returns:
            Path to the saved checkpoint.
        """
        metadata = CheckpointMetadata(
            epoch=epoch,
            global_step=self.global_step,
            avg_map=avg_map,
            avg_ndcg=avg_ndcg,
            config=self.config,
        )
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        return save_checkpoint(self.model, self.optimizer, metadata, ckpt_path)
