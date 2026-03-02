"""Download EPIC-KITCHENS-100 dataset.

This script provides instructions and utilities for downloading the
EPIC-KITCHENS-100 dataset for the Multi-Instance Retrieval challenge.

Usage:
    python data/download/download_ek100.py --participants P01 P02

Prerequisites:
    Clone https://github.com/epic-kitchens/epic-kitchens-100-annotations
    Then run:
        python epic_downloader.py --videos --rgb-frames --action-retrieval
"""

from __future__ import annotations

import argparse
from pathlib import Path


def download_ek100(
    output_dir: str = "data/raw/EK100",
    participants: list[str] | None = None,
) -> None:
    """Download EPIC-KITCHENS-100 data.

    Args:
        output_dir: Target directory for downloads.
        participants: Optional list of participant IDs to download.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("EPIC-KITCHENS-100 Download Instructions:")
    print("=" * 50)
    print(f"1. Target directory: {out_path.resolve()}")
    print("2. Clone annotations repo:")
    print("   git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations")
    print("3. Run the official downloader:")
    cmd = "   python epic_downloader.py --videos --rgb-frames --action-retrieval"
    if participants:
        cmd += f" --participants {' '.join(participants)}"
    print(cmd)
    print()
    print("Estimated storage: ~740GB for full dataset")
    print("Use --participants flag to download a subset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EK100 dataset")
    parser.add_argument("--output-dir", type=str, default="data/raw/EK100")
    parser.add_argument("--participants", nargs="+", type=str, default=None)
    args = parser.parse_args()
    download_ek100(args.output_dir, args.participants)
