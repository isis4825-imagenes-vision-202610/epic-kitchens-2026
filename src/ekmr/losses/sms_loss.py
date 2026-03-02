"""Symmetric Multi-Similarity Loss for multi-instance retrieval (CVPR 2024)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SMSLoss(nn.Module):
    """Symmetric Multi-Similarity Loss.

    Computes pairwise losses using a correlation/relevancy matrix to define
    positive and negative pairs with graded relevance.

    Args:
        tau: Relaxation factor to prevent loss dominance.
        alpha: Scaling factor for exponential terms.
        pos_threshold: Threshold above which a pair is considered positive.
    """

    def __init__(
        self,
        tau: float = 0.05,
        alpha: float = 2.0,
        pos_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.pos_threshold = pos_threshold

    def forward(
        self,
        video_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        relevancy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SMS loss.

        Args:
            video_embeds: Normalized video embeddings of shape (B, D).
            text_embeds: Normalized text embeddings of shape (B, D).
            relevancy: Correlation matrix of shape (B, B) with values in [0, 1].

        Returns:
            Scalar loss tensor.
        """
        sim = video_embeds @ text_embeds.t()
        pos_mask = relevancy > self.pos_threshold
        neg_mask = relevancy == 0.0

        loss_v2t = self._direction_loss(sim, pos_mask, neg_mask)
        loss_t2v = self._direction_loss(sim.t(), pos_mask.t(), neg_mask.t())
        return (loss_v2t + loss_t2v) / 2.0

    def _direction_loss(
        self,
        sim: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one-directional SMS loss.

        Args:
            sim: Similarity matrix of shape (B, B).
            pos_mask: Boolean mask for positive pairs.
            neg_mask: Boolean mask for negative pairs.

        Returns:
            Scalar loss for one direction.
        """
        batch_size = sim.size(0)
        loss = torch.tensor(0.0, device=sim.device, dtype=sim.dtype)
        count = 0

        for i in range(batch_size):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_idx = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            pos_sim = sim[i, pos_idx]
            neg_sim = sim[i, neg_idx]

            # For each positive, compute log-sum-exp over negatives
            pos_term = torch.logsumexp(
                self.alpha * (neg_sim.unsqueeze(0) - pos_sim.unsqueeze(1) + self.tau),
                dim=1,
            )
            pos_loss = torch.log(1.0 + pos_term.exp()).mean()

            # For each negative, compute log-sum-exp over positives
            neg_term = torch.logsumexp(
                self.alpha * (pos_sim.unsqueeze(0) - neg_sim.unsqueeze(1) - self.tau),
                dim=1,
            )
            neg_loss = torch.log(1.0 + neg_term.exp()).mean()

            loss = loss + pos_loss + neg_loss
            count += 1

        if count > 0:
            loss = loss / count
        return loss
