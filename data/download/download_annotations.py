"""Download EPIC-KITCHENS-100 annotations.

Clones the official annotation repository into the dataset directory.
Safe to re-run: pulls updates if the repository already exists.

Usage:
    python data/download/download_annotations.py
    python data/download/download_annotations.py --output-dir /path/to/data
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ANNOTATIONS_REPO = "https://github.com/epic-kitchens/epic-kitchens-100-annotations.git"
DEFAULT_OUTPUT_DIR = "data/raw/EK100"


def download_annotations(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    """Download annotation files from the official repository.

    Clones (or updates) the epic-kitchens-100-annotations repository into
    ``output_dir/epic-kitchens-100-annotations``.

    Args:
        output_dir: Root dataset directory (will be created if needed).
    """
    ann_dir = Path(output_dir) / "epic-kitchens-100-annotations"
    ann_dir.parent.mkdir(parents=True, exist_ok=True)

    if ann_dir.exists():
        print(f"Annotations already cloned at {ann_dir}. Pulling latest changes ...")
        try:
            subprocess.run(
                ["git", "-C", str(ann_dir), "pull", "--ff-only"],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"  Warning: git pull failed ({exc}). Using existing copy.")
    else:
        print(f"Cloning annotations repository to {ann_dir} ...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    ANNOTATIONS_REPO,
                    str(ann_dir),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Failed to clone annotations: {exc}")
            print("Please clone manually:")
            print(f"  git clone {ANNOTATIONS_REPO} {ann_dir}")
            sys.exit(exc.returncode)

    # Sanity-check a few key files
    required = [
        "EPIC_100_retrieval_train.csv",
        "EPIC_100_retrieval_test.csv",
        "EPIC_100_retrieval_train_sentence.csv",
        "EPIC_100_retrieval_test_sentence.csv",
    ]
    missing = [f for f in required if not (ann_dir / f).exists()]
    if missing:
        print(f"Warning: the following expected files are missing: {missing}")
    else:
        print("Annotations downloaded and verified successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EK100 annotations")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Root dataset directory (default: data/raw/EK100)",
    )
    args = parser.parse_args()
    download_annotations(args.output_dir)
