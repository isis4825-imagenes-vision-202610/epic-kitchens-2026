"""Video augmentation transforms for training and evaluation."""

from __future__ import annotations

import torch


class VideoTransform:
    """Composable video transform pipeline.

    Args:
        is_training: Whether to apply training augmentations.
        crop_size: Spatial crop size.
    """

    def __init__(self, is_training: bool = True, crop_size: int = 224) -> None:
        self.is_training = is_training
        self.crop_size = crop_size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Apply transforms to video frames.

        Args:
            video: Video tensor of shape (T, C, H, W).

        Returns:
            Transformed video tensor.
        """
        if self.is_training:
            video = self._random_crop(video)
            video = self._random_horizontal_flip(video)
            video = self._color_jitter(video)
        else:
            video = self._center_crop(video)

        video = self._normalize(video)
        return video

    def _random_crop(self, video: torch.Tensor) -> torch.Tensor:
        """Apply random spatial crop.

        Args:
            video: Video tensor of shape (T, C, H, W).

        Returns:
            Cropped video tensor.
        """
        _, _, h, w = video.shape
        if h <= self.crop_size or w <= self.crop_size:
            return torch.nn.functional.interpolate(
                video, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False
            )
        top = torch.randint(0, h - self.crop_size, (1,)).item()
        left = torch.randint(0, w - self.crop_size, (1,)).item()
        assert isinstance(top, int)
        assert isinstance(left, int)
        return video[:, :, top : top + self.crop_size, left : left + self.crop_size]

    def _center_crop(self, video: torch.Tensor) -> torch.Tensor:
        """Apply center spatial crop.

        Args:
            video: Video tensor of shape (T, C, H, W).

        Returns:
            Center-cropped video tensor.
        """
        _, _, h, w = video.shape
        if h <= self.crop_size or w <= self.crop_size:
            return torch.nn.functional.interpolate(
                video, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False
            )
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2
        return video[:, :, top : top + self.crop_size, left : left + self.crop_size]

    def _random_horizontal_flip(self, video: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """Randomly flip video horizontally.

        Args:
            video: Video tensor of shape (T, C, H, W).
            p: Probability of flipping.

        Returns:
            Possibly flipped video tensor.
        """
        if torch.rand(1).item() < p:
            return torch.flip(video, dims=[-1])
        return video

    def _color_jitter(self, video: torch.Tensor) -> torch.Tensor:
        """Apply simple color jitter to video frames.

        Args:
            video: Video tensor of shape (T, C, H, W).

        Returns:
            Color-jittered video tensor.
        """
        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
        return (video * brightness).clamp(0.0, 1.0)

    def _normalize(self, video: torch.Tensor) -> torch.Tensor:
        """Normalize video with ImageNet statistics.

        Args:
            video: Video tensor of shape (T, C, H, W).

        Returns:
            Normalized video tensor.
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        return (video - mean.to(video.device)) / std.to(video.device)
