"""Main training script for EPIC-KITCHENS-100 retrieval models.

Usage:
    # Single GPU
    python scripts/train.py model=cr_clip experiment=baseline_crclip

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 scripts/train.py model=cr_clip

    # Hyperparameter sweep
    python scripts/train.py --multirun optimizer.lr=1e-5,1.8e-5,3e-5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ekmr.datasets.ek100 import EK100Dataset
from ekmr.datasets.transforms import VideoTransform
from ekmr.losses.sms_loss import SMSLoss
from ekmr.models.cr_clip import CRClip
from ekmr.training.evaluator import Evaluator
from ekmr.training.trainer import Trainer
from ekmr.utils.distributed import cleanup_distributed, is_main_process, setup_distributed
from ekmr.utils.logging import setup_logging
from ekmr.utils.seed import seed_everything


def build_model(cfg: DictConfig) -> torch.nn.Module:
    """Build the retrieval model from config.

    Args:
        cfg: Model configuration.

    Returns:
        Initialized model.
    """
    model_cfg = cfg.get("model", {})
    target = str(model_cfg.get("_target_", "ekmr.models.cr_clip.CRClip"))

    if "cr_clip" in target:
        return CRClip(
            embed_dim=model_cfg.get("embed_dim", 768),
            refinement_dim=model_cfg.get("refinement_dim", 512),
            num_heads=model_cfg.get("num_heads", 8),
            num_frames=model_cfg.get("num_frames", 16),
            dropout=model_cfg.get("dropout", 0.1),
        )
    from ekmr.models.sms_loss_model import SMSLossModel

    return SMSLossModel(
        embed_dim=model_cfg.get("embed_dim", 768),
        num_frames=model_cfg.get("num_frames", 16),
    )


def build_loss(cfg: DictConfig) -> torch.nn.Module:
    """Build the loss function from config.

    Args:
        cfg: Loss configuration.

    Returns:
        Initialized loss function.
    """
    loss_cfg = cfg.get("loss", {})
    target = str(loss_cfg.get("_target_", "ekmr.losses.sms_loss.SMSLoss"))

    if "mil_nce" in target:
        from ekmr.losses.mil_nce import MILNCELoss

        return MILNCELoss(temperature=loss_cfg.get("temperature", 0.07))
    return SMSLoss(tau=loss_cfg.get("tau", 0.05))


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Hydra configuration.
    """
    seed_everything(cfg.get("seed", 42))

    if cfg.get("training", {}).get("ddp", False):
        setup_distributed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.get("output_dir", "experiments/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        setup_logging(output_dir)
        OmegaConf.save(cfg, str(output_dir / "config.yaml"))

    # Build components
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("optimizer", {}).get("lr", 1.8e-5)),
        weight_decay=float(cfg.get("optimizer", {}).get("weight_decay", 0.01)),
    )

    training_cfg = cfg.get("training", {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_cfg.get("epochs", 10)),
    )

    # Build datasets
    dataset_cfg = cfg.get("dataset", {})
    train_transform = VideoTransform(is_training=True)
    train_dataset = EK100Dataset(
        root=str(dataset_cfg.get("root", "data/raw/EK100")),
        split="train",
        num_frames=int(dataset_cfg.get("num_frames", 16)),
        transform=train_transform,
    )
    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_dataset,
        batch_size=int(dataset_cfg.get("batch_size", 40)),
        shuffle=True,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=bool(dataset_cfg.get("pin_memory", True)),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        output_dir=output_dir,
    )

    # Training loop
    epochs = int(training_cfg.get("epochs", 10))
    results: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        epoch_metrics = trainer.train_epoch(epoch)
        trainer.save(epoch)
        results.append({"epoch": epoch, **epoch_metrics})

        if is_main_process():
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

    cleanup_distributed()


if __name__ == "__main__":
    main()
