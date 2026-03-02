"""Adaptive Multi-Instance Max-Margin loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f  # noqa: N812


class AdaptiveMIMMoss(nn.Module):
    """Adaptive Multi-Instance Max-Margin loss.

    Adaptive margin loss that weights pairs by their relevancy scores.

    Args:
        margin: Base margin for negative pairs.
    """

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        video_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        relevancy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Adaptive MI-MM loss.

        Args:
            video_embeds: Normalized video embeddings (B, D).
            text_embeds: Normalized text embeddings (B, D).
            relevancy: Relevancy matrix (B, B) with values in [0, 1].

        Returns:
            Scalar loss tensor.
        """
        sim = video_embeds @ text_embeds.t()
        pos_mask = relevancy > 0
        neg_mask = ~pos_mask

        # For each query, get max positive sim and hardest negative
        pos_sim = sim.clone()
        pos_sim[~pos_mask] = float("-inf")
        max_pos = pos_sim.max(dim=1).values

        neg_sim = sim.clone()
        neg_sim[~neg_mask] = float("-inf")
        max_neg = neg_sim.max(dim=1).values

        # Triplet loss with margin
        loss_v2t = f.relu(self.margin + max_neg - max_pos).mean()

        # Text-to-video direction
        pos_sim_t = sim.t().clone()
        pos_sim_t[~pos_mask.t()] = float("-inf")
        max_pos_t = pos_sim_t.max(dim=1).values

        neg_sim_t = sim.t().clone()
        neg_sim_t[~neg_mask.t()] = float("-inf")
        max_neg_t = neg_sim_t.max(dim=1).values

        loss_t2v = f.relu(self.margin + max_neg_t - max_pos_t).mean()

        return (loss_v2t + loss_t2v) / 2.0
