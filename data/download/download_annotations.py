"""Download EPIC-KITCHENS-100 annotations.

Usage:
    python data/download/download_annotations.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def download_annotations(output_dir: str = "data/raw/EK100") -> None:
    """Download annotation files from the official repository.

    Args:
        output_dir: Target directory for annotations.
    """
    ann_dir = Path(output_dir) / "epic-kitchens-100-annotations"
    if ann_dir.exists():
        print(f"Annotations directory already exists: {ann_dir}")
        return

    ann_dir.parent.mkdir(parents=True, exist_ok=True)
    print("Cloning EPIC-KITCHENS-100 annotations repository...")
    print(f"Target: {ann_dir}")

    try:
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/epic-kitchens/epic-kitchens-100-annotations",
                str(ann_dir),
            ],
            check=True,
        )
        print("Annotations downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone annotations: {e}")
        print("Please clone manually:")
        print(
            f"  git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations {ann_dir}"
        )


if __name__ == "__main__":
    download_annotations()
