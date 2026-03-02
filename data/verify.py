"""Verify data integrity of the EK100 dataset.

Usage:
    python data/verify.py --root data/raw/EK100
"""

from __future__ import annotations

import argparse
from pathlib import Path


def verify_dataset(root: str = "data/raw/EK100") -> bool:
    """Verify the integrity of the downloaded dataset.

    Args:
        root: Root directory of the EK100 dataset.

    Returns:
        True if all expected files are present.
    """
    root_path = Path(root)
    expected_files = [
        "epic-kitchens-100-annotations/EPIC_100_retrieval_train.csv",
        "epic-kitchens-100-annotations/EPIC_100_retrieval_test.csv",
        "epic-kitchens-100-annotations/EPIC_100_retrieval_train_sentence.csv",
        "epic-kitchens-100-annotations/EPIC_100_retrieval_test_sentence.csv",
    ]

    all_ok = True
    for f in expected_files:
        path = root_path / f
        if path.exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            all_ok = False

    if all_ok:
        print("\nAll expected files are present.")
    else:
        print("\nSome files are missing. Run download scripts first.")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify EK100 dataset")
    parser.add_argument("--root", type=str, default="data/raw/EK100")
    args = parser.parse_args()
    verify_dataset(args.root)
