"""Pre-extract ViT features for fast training.

Usage:
    python data/preprocess/extract_features.py --input data/processed/frames --output data/features
"""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_features(
    input_dir: str,
    output_dir: str,
    model_name: str = "avion",
    batch_size: int = 32,
) -> None:
    """Pre-extract visual features from frames.

    Args:
        input_dir: Directory containing extracted frames.
        output_dir: Directory to save feature files.
        model_name: Name of the feature extraction model.
        batch_size: Batch size for feature extraction.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Feature extraction from: {in_path}")
    print(f"Model: {model_name}, Batch size: {batch_size}")
    print(f"Output: {out_path}")
    print("Note: This requires the actual model weights to be loaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="avion")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    extract_features(args.input, args.output, args.model, args.batch_size)
