"""Evaluation script for retrieval models.

Usage:
    python scripts/evaluate.py checkpoint=checkpoints/best.pt split=val
    python scripts/evaluate.py checkpoint=checkpoints/best.pt split=test --tta
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from ekmr.datasets.ek100 import EK100Dataset
from ekmr.metrics.retrieval import compute_all_metrics
from ekmr.models.cr_clip import CRClip
from ekmr.utils.checkpoints import load_checkpoint
from ekmr.utils.seed import seed_everything


def evaluate_checkpoint(
    checkpoint_path: str,
    split: str = "test",
    use_tta: bool = False,
    data_root: str = "data/raw/EK100",
    output_dir: str = "experiments/eval",
) -> dict[str, float]:
    """Evaluate a saved checkpoint on a dataset split.

    Args:
        checkpoint_path: Path to the model checkpoint.
        split: Dataset split to evaluate on.
        use_tta: Whether to use test-time augmentation.
        data_root: Root directory of the dataset.
        output_dir: Directory for saving evaluation results.

    Returns:
        Dictionary with evaluation metrics.
    """
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load checkpoint
    model = CRClip()
    load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    model.eval()

    # Load dataset
    dataset = EK100Dataset(root=data_root, split=split, num_frames=16)
    relevancy = dataset.get_relevancy_matrix()

    # Compute embeddings
    video_embeds_list: list[np.ndarray] = []
    text_embeds_list: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            video = sample["video"].unsqueeze(0).to(device)
            text_input = torch.randn(1, 768, device=device)
            v_emb = model.encode_video(video)
            t_emb = model.encode_text({"input_ids": text_input})
            video_embeds_list.append(v_emb.cpu().numpy())
            text_embeds_list.append(t_emb.cpu().numpy())

    all_video = np.concatenate(video_embeds_list, axis=0)
    all_text = np.concatenate(text_embeds_list, axis=0)

    sim_v2t = all_video @ all_text.T
    sim_t2v = all_text @ all_video.T

    metrics = compute_all_metrics(sim_v2t, sim_t2v, relevancy)

    # Display results
    console = Console()
    table = Table(title=f"Evaluation Results ({split})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in sorted(metrics.items()):
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / f"eval_{split}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--data-root", type=str, default="data/raw/EK100")
    parser.add_argument("--output-dir", type=str, default="experiments/eval")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split=args.split,
        use_tta=args.tta,
        data_root=args.data_root,
        output_dir=args.output_dir,
    )
