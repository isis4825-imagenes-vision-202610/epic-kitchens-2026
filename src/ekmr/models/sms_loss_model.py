"""SMS-Loss model: AVION + Symmetric Multi-Similarity Loss (CVPR 2024)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as func

from ekmr.models.backbones.avion import AVIONEncoder
from ekmr.models.base import BaseRetriever


class SMSLossModel(BaseRetriever):
    """AVION backbone with SMS-Loss training objective.

    The 2024 CVPR challenge winner using AVION backbone trained with
    Symmetric Multi-Similarity Loss.

    Args:
        embed_dim: Embedding dimension.
        num_frames: Number of input video frames.
    """

    def __init__(self, embed_dim: int = 768, num_frames: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.video_encoder = AVIONEncoder(embed_dim=embed_dim, num_frames=num_frames)
        self.text_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video frames into normalized embeddings.

        Args:
            video: Video tensor of shape (B, T, C, H, W).

        Returns:
            Normalized video embeddings of shape (B, D).
        """
        return func.normalize(self.video_encoder(video), dim=-1)

    def encode_text(self, text: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode tokenized text into normalized embeddings.

        Args:
            text: Dictionary with 'input_ids' tensor of shape (B, D).

        Returns:
            Normalized text embeddings of shape (B, D).
        """
        features = text["input_ids"].float()
        return func.normalize(self.text_projection(features), dim=-1)
