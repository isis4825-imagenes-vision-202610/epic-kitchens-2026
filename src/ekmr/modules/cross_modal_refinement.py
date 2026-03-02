"""Cross-Modal Contextual Refinement module for CR-CLIP."""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalContextRefinement(nn.Module):
    """Bidirectional cross-modal attention refinement.

    Implements the cross-attention mechanism from CR-CLIP (CVPR 2025)
    that refines video and text embeddings through bidirectional attention.

    Args:
        embed_dim: Input embedding dimension.
        refinement_dim: Internal refinement dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        refinement_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.refinement_dim = refinement_dim

        # Video-to-text cross attention
        self.v2t_attention = nn.MultiheadAttention(
            embed_dim=refinement_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Text-to-video cross attention
        self.t2v_attention = nn.MultiheadAttention(
            embed_dim=refinement_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Projections
        self.video_proj = nn.Linear(embed_dim, refinement_dim)
        self.text_proj = nn.Linear(embed_dim, refinement_dim)

        # Output projections back to embed_dim
        self.video_out = nn.Linear(refinement_dim + embed_dim, embed_dim)
        self.text_out = nn.Linear(refinement_dim + embed_dim, embed_dim)

        self.layer_norm_v = nn.LayerNorm(embed_dim)
        self.layer_norm_t = nn.LayerNorm(embed_dim)

    def forward(
        self,
        video_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine video and text embeddings through cross-modal attention.

        Args:
            video_embeds: Video embeddings of shape (B, D).
            text_embeds: Text embeddings of shape (B, D).

        Returns:
            Tuple of refined (video_embeddings, text_embeddings).
        """
        # Add sequence dim for attention: (B, D) -> (B, 1, D)
        v = video_embeds.unsqueeze(1)
        t = text_embeds.unsqueeze(1)

        # Project to refinement dim
        v_proj = self.video_proj(v)  # (B, 1, refinement_dim)
        t_proj = self.text_proj(t)  # (B, 1, refinement_dim)

        # Cross attention: video attends to text
        v_refined, _ = self.v2t_attention(
            query=v_proj, key=t_proj, value=t_proj
        )

        # Cross attention: text attends to video
        t_refined, _ = self.t2v_attention(
            query=t_proj, key=v_proj, value=v_proj
        )

        # Squeeze back: (B, 1, refinement_dim) -> (B, refinement_dim)
        v_refined = v_refined.squeeze(1)
        t_refined = t_refined.squeeze(1)

        # Concatenate original + refined, project back
        v_out = self.video_out(torch.cat([video_embeds, v_refined], dim=-1))
        t_out = self.text_out(torch.cat([text_embeds, t_refined], dim=-1))

        # Layer norm + residual
        v_out = self.layer_norm_v(v_out + video_embeds)
        t_out = self.layer_norm_t(t_out + text_embeds)

        return v_out, t_out
