"""Custom model skeleton for novel retrieval approaches."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as func

from ekmr.models.base import BaseRetriever


class CustomRetriever(BaseRetriever):
    """Skeleton for a custom video-text retrieval model.

    Implement your novel architecture here. Must provide
    encode_video and encode_text methods.

    Args:
        embed_dim: Embedding dimension.
        num_frames: Number of input video frames.
    """

    def __init__(self, embed_dim: int = 768, num_frames: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.video_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video frames into normalized embeddings.

        Args:
            video: Video tensor of shape (B, T, C, H, W).

        Returns:
            Normalized video embeddings of shape (B, D).
        """
        b = video.size(0)
        features = video.reshape(b, -1).mean(dim=-1, keepdim=True)
        features = features.expand(-1, self.embed_dim)
        return func.normalize(self.video_proj(features), dim=-1)

    def encode_text(self, text: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode tokenized text into normalized embeddings.

        Args:
            text: Dictionary with 'input_ids' tensor of shape (B, D).

        Returns:
            Normalized text embeddings of shape (B, D).
        """
        features = text["input_ids"].float()
        return func.normalize(self.text_proj(features), dim=-1)
