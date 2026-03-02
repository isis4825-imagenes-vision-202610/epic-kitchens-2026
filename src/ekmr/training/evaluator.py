"""Evaluation utilities for retrieval models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    import numpy as np
    from torch.utils.data import DataLoader

from ekmr.metrics.retrieval import compute_all_metrics


class Evaluator:
    """Evaluator for video-text retrieval models.

    Runs inference on a dataset and computes retrieval metrics.

    Args:
        model: The retrieval model to evaluate.
        device: Device to run inference on.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader[dict[str, Any]],
        relevancy_matrix: np.ndarray,
    ) -> dict[str, float]:
        """Run evaluation and compute metrics.

        Args:
            data_loader: Data loader for the evaluation split.
            relevancy_matrix: Ground truth relevancy matrix.

        Returns:
            Dictionary with all retrieval metrics.
        """
        self.model.eval()
        video_embeds_list: list[torch.Tensor] = []
        text_embeds_list: list[torch.Tensor] = []

        for batch in data_loader:
            video = batch["video"].to(self.device, non_blocking=True)
            text = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.get("text", {}).items()
            }
            v_emb, t_emb = self.model(video, text)
            video_embeds_list.append(v_emb.cpu())
            text_embeds_list.append(t_emb.cpu())

        all_video = torch.cat(video_embeds_list, dim=0)
        all_text = torch.cat(text_embeds_list, dim=0)

        sim_v2t = (all_video @ all_text.t()).numpy()
        sim_t2v = (all_text @ all_video.t()).numpy()

        return compute_all_metrics(sim_v2t, sim_t2v, relevancy_matrix)
