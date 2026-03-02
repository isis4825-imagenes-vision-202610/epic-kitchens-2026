"""Visualization of metrics over training epochs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def plot_metrics_over_epochs(
    epochs: list[int],
    metrics: dict[str, list[float]],
    output_path: str | Path | None = None,
) -> None:
    """Plot metrics (mAP, nDCG) over training epochs.

    Args:
        epochs: List of epoch numbers.
        metrics: Dictionary mapping metric names to lists of values.
        output_path: Optional path to save the plot.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    map_metrics = {k: v for k, v in metrics.items() if "mAP" in k}
    ndcg_metrics = {k: v for k, v in metrics.items() if "nDCG" in k}

    ax_map = axes[0]
    for name, values in map_metrics.items():
        ax_map.plot(epochs[: len(values)], values, marker="o", label=name)
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("mAP")
    ax_map.set_title("Mean Average Precision")
    ax_map.legend()
    ax_map.grid(True, alpha=0.3)

    ax_ndcg = axes[1]
    for name, values in ndcg_metrics.items():
        ax_ndcg.plot(epochs[: len(values)], values, marker="s", label=name)
    ax_ndcg.set_xlabel("Epoch")
    ax_ndcg.set_ylabel("nDCG")
    ax_ndcg.set_title("Normalized DCG")
    ax_ndcg.legend()
    ax_ndcg.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
