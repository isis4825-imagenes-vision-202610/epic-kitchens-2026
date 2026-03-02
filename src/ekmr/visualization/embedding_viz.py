"""Embedding space visualization with dimensionality reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def plot_embeddings(
    video_embeds: np.ndarray,
    text_embeds: np.ndarray,
    labels: list[str] | None = None,
    method: str = "tsne",
    output_path: str | Path | None = None,
) -> None:
    """Plot 2D projection of video and text embeddings.

    Args:
        video_embeds: Video embeddings of shape (N, D).
        text_embeds: Text embeddings of shape (M, D).
        labels: Optional labels for coloring points.
        method: Dimensionality reduction method ('tsne' or 'pca').
        output_path: Optional path to save the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    all_embeds = np.concatenate([video_embeds, text_embeds], axis=0)
    n_video = len(video_embeds)

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeds) - 1))
    else:
        reducer = PCA(n_components=2, random_state=42)

    coords = reducer.fit_transform(all_embeds)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        coords[:n_video, 0],
        coords[:n_video, 1],
        c="blue",
        alpha=0.5,
        label="Video",
        s=20,
    )
    ax.scatter(
        coords[n_video:, 0],
        coords[n_video:, 1],
        c="red",
        alpha=0.5,
        label="Text",
        s=20,
    )
    ax.legend()
    ax.set_title(f"Embedding Space ({method.upper()})")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
