"""Test-time augmentation for retrieval inference."""

from __future__ import annotations

import torch
import torch.nn as nn


def horizontal_flip_tta(
    model: nn.Module,
    video_frames: torch.Tensor,
) -> torch.Tensor:
    """Apply horizontal flip TTA to video embeddings.

    Extracts features on both original and horizontally-flipped frames,
    then averages the resulting embeddings.

    Args:
        model: Model with an encode_video method.
        video_frames: Video frames tensor of shape (B, T, C, H, W).

    Returns:
        Averaged video embeddings of shape (B, D).
    """
    with torch.no_grad():
        emb_orig = model.encode_video(video_frames)  # type: ignore[operator]
        flipped = torch.flip(video_frames, dims=[-1])
        emb_flip = model.encode_video(flipped)  # type: ignore[operator]
    avg_emb = (emb_orig + emb_flip) / 2.0
    return torch.nn.functional.normalize(avg_emb, dim=-1)


def multi_scale_tta(
    model: nn.Module,
    video_frames: torch.Tensor,
    scales: list[int] | None = None,
) -> torch.Tensor:
    """Apply multi-scale TTA to video embeddings.

    Resizes frames to multiple scales, extracts features, and averages.

    Args:
        model: Model with an encode_video method.
        video_frames: Video frames tensor of shape (B, T, C, H, W).
        scales: List of spatial sizes to resize to. Defaults to [224, 256, 288].

    Returns:
        Averaged video embeddings of shape (B, D).
    """
    if scales is None:
        scales = [224, 256, 288]

    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for size in scales:
            b, t, c, h, w = video_frames.shape
            frames_flat = video_frames.reshape(b * t, c, h, w)
            resized = torch.nn.functional.interpolate(
                frames_flat,
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )
            resized = resized.reshape(b, t, c, size, size)
            emb = model.encode_video(resized)  # type: ignore[operator]
            embeddings.append(emb)

    avg_emb = torch.stack(embeddings).mean(dim=0)
    return torch.nn.functional.normalize(avg_emb, dim=-1)
