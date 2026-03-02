"""Hyperparameter sweep launcher using Hydra multirun.

Usage:
    python scripts/sweep.py --multirun optimizer.lr=1e-5,1.8e-5,3e-5
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Launch training sweep.

    Args:
        cfg: Hydra configuration with sweep parameters.
    """
    from scripts.train import build_loss, build_model

    print(f"Running sweep with config:")
    print(f"  Model: {cfg.get('model', {}).get('_target_', 'unknown')}")
    print(f"  LR: {cfg.get('optimizer', {}).get('lr', 'unknown')}")
    print(f"  Loss tau: {cfg.get('loss', {}).get('tau', 'unknown')}")


if __name__ == "__main__":
    main()
