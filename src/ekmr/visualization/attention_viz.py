"""Cross-modal attention weight visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    video_labels: list[str] | None = None,
    text_labels: list[str] | None = None,
    output_path: str | Path | None = None,
) -> None:
    """Plot cross-modal attention weights as a heatmap.

    Args:
        attention_weights: Attention matrix of shape (N_video, N_text).
        video_labels: Optional labels for video frames.
        text_labels: Optional labels for text tokens.
        output_path: Optional path to save the plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")

    if video_labels is not None:
        ax.set_yticks(range(len(video_labels)))
        ax.set_yticklabels(video_labels, fontsize=8)

    if text_labels is not None:
        ax.set_xticks(range(len(text_labels)))
        ax.set_xticklabels(text_labels, fontsize=8, rotation=45, ha="right")

    ax.set_ylabel("Video Frames")
    ax.set_xlabel("Text Tokens")
    ax.set_title("Cross-Modal Attention Weights")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
