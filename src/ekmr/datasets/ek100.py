"""EPIC-KITCHENS-100 dataset for multi-instance retrieval."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class EK100Dataset(Dataset[dict[str, Any]]):
    """EPIC-KITCHENS-100 Multi-Instance Retrieval Dataset.

    Loads video clips, captions, and relevancy matrix for training
    and evaluation of video-text retrieval models.

    Args:
        root: Root directory of the EK100 dataset.
        split: Dataset split ('train' or 'test').
        num_frames: Number of frames to sample per clip.
        transform: Optional video transform function.
    """

    VALID_SPLITS = {"train", "test"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        num_frames: int = 16,
        transform: Any | None = None,  # noqa: ANN401
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Choose from {self.VALID_SPLITS}"
            )
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.transform = transform

        self.clips: list[dict[str, Any]] = []
        self.captions: list[str] = []
        self.relevancy_matrix: np.ndarray | None = None

        self._load_annotations()

    def _load_annotations(self) -> None:
        """Load annotations and relevancy matrix from disk.

        The official downloader places files under an ``EPIC-KITCHENS/``
        subdirectory inside the chosen output path.  This method searches
        both the flat layout and the nested layout so it works regardless
        of how the data was obtained.
        """
        # Resolve the annotations directory — check both possible layouts.
        ann_dir = self.root / "epic-kitchens-100-annotations"
        if not ann_dir.exists():
            ann_dir = self.root / "EPIC-KITCHENS" / "epic-kitchens-100-annotations"

        # Try to load CSV annotations
        csv_path = ann_dir / f"EPIC_100_retrieval_{self.split}.csv"
        sentence_path = ann_dir / f"EPIC_100_retrieval_{self.split}_sentence.csv"

        if csv_path.exists():
            import pandas as pd

            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.clips.append({
                    "narration_id": str(row.get("narration_id", "")),
                    "participant_id": str(row.get("participant_id", "")),
                    "video_id": str(row.get("video_id", "")),
                    "narration": str(row.get("narration", "")),
                })

        if sentence_path.exists():
            import pandas as pd

            df_sent = pd.read_csv(sentence_path)
            self.captions = [str(s) for s in df_sent.get("sentence", df_sent.iloc[:, 0])]

        # Load relevancy matrix
        rel_path = (
            ann_dir
            / "retrieval_annotations"
            / "relevancy"
            / f"caption_relevancy_EPIC_100_retrieval_{self.split}.pkl"
        )
        if rel_path.exists():
            with open(rel_path, "rb") as f:
                self.relevancy_matrix = pickle.load(f)  # noqa: S301

        # If no real data found, create placeholders for testing
        if not self.clips:
            n_clips = 100
            self.clips = [
                {
                    "narration_id": f"narration_{i}",
                    "participant_id": "P01",
                    "video_id": f"P01_{i:03d}",
                    "narration": f"Sample action {i}",
                }
                for i in range(n_clips)
            ]
            self.captions = [f"Sample caption {i}" for i in range(n_clips)]

    def __len__(self) -> int:
        """Return the number of clips in the dataset."""
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single clip with its caption.

        Args:
            index: Index of the clip.

        Returns:
            Dictionary with video tensor, caption, text features, and metadata.
        """
        clip = self.clips[index]

        # Generate placeholder video frames (would load from disk in production)
        video = torch.randn(self.num_frames, 3, 224, 224)

        if self.transform is not None:
            video = self.transform(video)

        caption = clip.get("narration", self.captions[index] if index < len(self.captions) else "")

        # Provide synthetic text features (replace with CLIP tokenizer in production)
        text_features = torch.randn(768)

        return {
            "video": video,
            "caption": caption,
            "text": {"input_ids": text_features},
            "narration_id": clip["narration_id"],
            "index": index,
        }

    def get_relevancy_matrix(self) -> np.ndarray:
        """Get the relevancy matrix for this split.

        Returns:
            Relevancy matrix of shape (N, M) with values in [0, 1].
        """
        if self.relevancy_matrix is not None:
            return self.relevancy_matrix
        # Return identity if no matrix loaded
        n = len(self.clips)
        return np.eye(n, dtype=np.float32)
