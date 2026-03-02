"""Unit tests for dataset loading."""

from __future__ import annotations

import torch

from ekmr.datasets.ek100 import EK100Dataset
from ekmr.datasets.transforms import VideoTransform


class TestEK100Dataset:
    """Tests for EK100 Dataset."""

    def test_placeholder_length(self):
        """Placeholder dataset should have 100 samples."""
        dataset = EK100Dataset(root="/nonexistent", split="train")
        assert len(dataset) == 100

    def test_getitem_keys(self):
        """Each sample should have expected keys."""
        dataset = EK100Dataset(root="/nonexistent", split="train")
        sample = dataset[0]
        assert "video" in sample
        assert "caption" in sample
        assert "narration_id" in sample

    def test_video_shape(self):
        """Video tensor should have correct shape."""
        dataset = EK100Dataset(root="/nonexistent", split="train", num_frames=8)
        sample = dataset[0]
        assert sample["video"].shape[0] == 8
        assert sample["video"].shape[1] == 3

    def test_relevancy_matrix(self):
        """Relevancy matrix should be square identity for placeholders."""
        dataset = EK100Dataset(root="/nonexistent", split="train")
        rel = dataset.get_relevancy_matrix()
        assert rel.shape[0] == len(dataset)
        assert rel.shape[1] == len(dataset)


class TestVideoTransform:
    """Tests for video transforms."""

    def test_training_transform(self):
        """Training transform should output correct spatial size."""
        transform = VideoTransform(is_training=True, crop_size=32)
        video = torch.randn(8, 3, 64, 64)
        out = transform(video)
        assert out.shape[-2:] == (32, 32)

    def test_eval_transform(self):
        """Eval transform should output correct spatial size."""
        transform = VideoTransform(is_training=False, crop_size=32)
        video = torch.randn(8, 3, 64, 64)
        out = transform(video)
        assert out.shape[-2:] == (32, 32)
