"""AVION video encoder backbone (stub implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AVIONEncoder(nn.Module):
    """AVION Video Encoder (ViT-L/14 based).

    Stub implementation of the AVION video encoder. In production,
    this would load pretrained weights from the AVION repository.

    Args:
        embed_dim: Output embedding dimension.
        num_frames: Number of input frames.
    """

    def __init__(self, embed_dim: int = 768, num_frames: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.temporal_embed = nn.Parameter(torch.randn(1, num_frames, embed_dim) * 0.02)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_post = nn.LayerNorm(embed_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video frames into embeddings.

        Args:
            video: Video tensor of shape (B, T, C, H, W).

        Returns:
            Video embeddings of shape (B, embed_dim).
        """
        b, t, c, h, w = video.shape
        # Stub: spatial average pooling as placeholder for ViT
        frame_features = video.reshape(b, t, c, -1).mean(dim=-1)  # (B, T, C)

        # Project to embed_dim if needed
        if c != self.embed_dim:
            frame_features = nn.functional.adaptive_avg_pool1d(
                frame_features.transpose(1, 2), t
            ).transpose(1, 2)
            frame_features = frame_features[..., : self.embed_dim]
            if frame_features.size(-1) < self.embed_dim:
                pad = torch.zeros(
                    b, t, self.embed_dim - frame_features.size(-1),
                    device=video.device, dtype=video.dtype,
                )
                frame_features = torch.cat([frame_features, pad], dim=-1)

        # Add temporal embeddings
        temporal_embed = self.temporal_embed[:, :t, :]
        frame_features = frame_features + temporal_embed

        # Temporal pooling
        pooled = frame_features.mean(dim=1)  # (B, embed_dim)
        result: torch.Tensor = self.ln_post(self.projection(pooled))
        return result
