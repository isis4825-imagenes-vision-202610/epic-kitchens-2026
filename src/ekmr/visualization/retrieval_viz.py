"""Visualization of retrieval results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def plot_top_k_retrievals(
    query_text: str,
    sim_scores: np.ndarray,
    captions: list[str],
    relevancy_scores: np.ndarray | None = None,
    k: int = 5,
    output_path: str | Path | None = None,
) -> None:
    """Plot top-K retrieved results for a text query.

    Args:
        query_text: The text query.
        sim_scores: Similarity scores array of shape (N,).
        captions: List of candidate captions.
        relevancy_scores: Optional ground-truth relevancy scores.
        k: Number of top results to show.
        output_path: Optional path to save the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    top_indices = np.argsort(-sim_scores)[:k]
    top_scores = sim_scores[top_indices]
    top_captions = [captions[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(k)
    bars = ax.barh(y_pos, top_scores, align="center")

    if relevancy_scores is not None:
        top_rel = relevancy_scores[top_indices]
        for bar, rel in zip(bars, top_rel, strict=True):
            bar.set_color("green" if rel > 0.5 else "orange" if rel > 0 else "red")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c[:60] + "..." if len(c) > 60 else c for c in top_captions])
    ax.invert_yaxis()
    ax.set_xlabel("Similarity Score")
    ax.set_title(f"Query: {query_text[:80]}")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
