"""Build and validate relevancy matrices.

Usage:
    python data/preprocess/build_relevancy.py \
        --annotations data/raw/EK100/epic-kitchens-100-annotations
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def build_relevancy(annotations_dir: str) -> None:
    """Build and validate relevancy matrices from annotations.

    Args:
        annotations_dir: Path to the annotations directory.
    """
    ann_path = Path(annotations_dir)
    rel_dir = ann_path / "retrieval_annotations" / "relevancy"

    for split in ["train", "test"]:
        pkl_path = rel_dir / f"caption_relevancy_EPIC_100_retrieval_{split}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                matrix = pickle.load(f)  # noqa: S301
            print(f"{split}: shape={getattr(matrix, 'shape', 'unknown')}, "
                  f"dtype={getattr(matrix, 'dtype', 'unknown')}")
            if hasattr(matrix, "shape"):
                print(f"  Range: [{matrix.min():.4f}, {matrix.max():.4f}]")
                print(f"  Non-zero: {(matrix > 0).sum()}")
        else:
            print(f"{split}: {pkl_path} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build relevancy matrices")
    parser.add_argument("--annotations", type=str, required=True)
    args = parser.parse_args()
    build_relevancy(args.annotations)
