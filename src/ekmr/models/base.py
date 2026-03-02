"""Abstract base class for retrieval models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseRetriever(nn.Module, ABC):
    """Abstract base class for video-text retrieval models.

    All retrieval models must implement encode_video and encode_text methods
    that produce normalized embeddings suitable for similarity computation.
    """

    @abstractmethod
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video frames into embedding space.

        Args:
            video: Video tensor of shape (B, T, C, H, W).

        Returns:
            Normalized video embeddings of shape (B, D).
        """
        ...

    @abstractmethod
    def encode_text(self, text: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode tokenized text into embedding space.

        Args:
            text: Dictionary with tokenized text tensors.

        Returns:
            Normalized text embeddings of shape (B, D).
        """
        ...

    def forward(
        self,
        video: torch.Tensor,
        text: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both modalities.

        Args:
            video: Video tensor of shape (B, T, C, H, W).
            text: Dictionary with tokenized text tensors.

        Returns:
            Tuple of (video_embeddings, text_embeddings), each shape (B, D).
        """
        return self.encode_video(video), self.encode_text(text)
