"""Multiple Instance Learning NCE loss for video-text retrieval."""

from __future__ import annotations

import torch
import torch.nn as nn


class MILNCELoss(nn.Module):
    """Multiple Instance Learning Noise Contrastive Estimation loss.

    Used as a baseline loss for video-text retrieval where multiple
    captions may be relevant for a single video.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        """Initialize MIL-NCE loss.

        Args:
            temperature: Temperature scaling factor for similarity scores.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        video_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        relevancy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MIL-NCE loss.

        Args:
            video_embeds: Normalized video embeddings (B, D).
            text_embeds: Normalized text embeddings (B, D).
            relevancy: Optional relevancy matrix (B, B). If None, uses identity.

        Returns:
            Scalar loss tensor.
        """
        sim = video_embeds @ text_embeds.t() / self.temperature

        if relevancy is None:
            relevancy = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)

        # Video-to-text direction
        pos_mask = relevancy > 0
        v2t_log_sum = torch.logsumexp(sim, dim=1)
        v2t_pos = (sim * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp(min=1)
        loss_v2t = (v2t_log_sum - v2t_pos).mean()

        # Text-to-video direction
        t2v_log_sum = torch.logsumexp(sim.t(), dim=1)
        pos_mask_t = pos_mask.t()
        pos_t_float = pos_mask_t.float()
        t2v_pos = (sim.t() * pos_t_float).sum(dim=1) / pos_t_float.sum(dim=1).clamp(min=1)
        loss_t2v = (t2v_log_sum - t2v_pos).mean()

        return (loss_v2t + loss_t2v) / 2.0
