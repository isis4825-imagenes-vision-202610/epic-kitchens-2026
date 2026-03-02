"""CR-CLIP: Cross-Modal Contextual Refinement CLIP (CVPR 2025 winner)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as func

from ekmr.models.backbones.avion import AVIONEncoder
from ekmr.models.base import BaseRetriever
from ekmr.modules.cross_modal_refinement import CrossModalContextRefinement


class CRClip(BaseRetriever):
    """CR-CLIP model for video-text multi-instance retrieval.

    Combines AVION video encoder with a text encoder and
    cross-modal contextual refinement for improved retrieval.

    Args:
        embed_dim: Embedding dimension for both modalities.
        refinement_dim: Internal dimension for cross-modal refinement.
        num_heads: Number of attention heads in refinement module.
        num_frames: Number of input video frames.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        refinement_dim: int = 512,
        num_heads: int = 8,
        num_frames: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Video encoder (AVION ViT-L/14)
        self.video_encoder = AVIONEncoder(embed_dim=embed_dim, num_frames=num_frames)

        # Text encoder (simple projection, would be CLIP text transformer in production)
        self.text_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Cross-modal refinement
        self.refinement = CrossModalContextRefinement(
            embed_dim=embed_dim,
            refinement_dim=refinement_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)  # ln(100)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video frames into normalized embeddings.

        Args:
            video: Video tensor of shape (B, T, C, H, W).

        Returns:
            Normalized video embeddings of shape (B, D).
        """
        embeds = self.video_encoder(video)
        return func.normalize(embeds, dim=-1)

    def encode_text(self, text: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode tokenized text into normalized embeddings.

        Args:
            text: Dictionary with 'input_ids' tensor of shape (B, D).

        Returns:
            Normalized text embeddings of shape (B, D).
        """
        features = text["input_ids"].float()
        embeds = self.text_projection(features)
        return func.normalize(embeds, dim=-1)

    def forward(
        self,
        video: torch.Tensor,
        text: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with cross-modal refinement.

        Args:
            video: Video tensor of shape (B, T, C, H, W).
            text: Dictionary with tokenized text tensors.

        Returns:
            Tuple of refined (video_embeddings, text_embeddings).
        """
        video_embeds = self.encode_video(video)
        text_embeds = self.encode_text(text)
        refined_v, refined_t = self.refinement(video_embeds, text_embeds)
        return func.normalize(refined_v, dim=-1), func.normalize(refined_t, dim=-1)
