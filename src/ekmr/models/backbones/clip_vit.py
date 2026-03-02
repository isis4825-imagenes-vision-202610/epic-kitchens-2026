"""CLIP ViT backbone wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn


class CLIPViT(nn.Module):
    """CLIP ViT-L/14 wrapper for video encoding.

    Wraps a CLIP Vision Transformer to process video frames by
    treating each frame as an independent image and averaging.

    Args:
        embed_dim: Output embedding dimension.
        pretrained: Whether to load pretrained weights.
    """

    def __init__(self, embed_dim: int = 768, pretrained: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.projection = nn.Linear(embed_dim, embed_dim)
        self._pretrained = pretrained

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames.

        Args:
            frames: Tensor of shape (B, T, C, H, W) or (B, C, H, W).

        Returns:
            Embeddings of shape (B, embed_dim).
        """
        if frames.dim() == 5:
            b, t, c, h, w = frames.shape
            frames_flat = frames.reshape(b * t, c, h, w)
        else:
            b = frames.size(0)
            t = 1
            frames_flat = frames

        # Placeholder: in production, this would go through a CLIP ViT
        feat = frames_flat.mean(dim=(-2, -1))  # (B*T, C)
        if feat.size(-1) != self.embed_dim:
            feat = nn.functional.adaptive_avg_pool1d(
                feat.unsqueeze(1), self.embed_dim
            ).squeeze(1)

        feat = feat.reshape(b, t, -1).mean(dim=1)  # average over time
        result: torch.Tensor = self.projection(feat)
        return result
