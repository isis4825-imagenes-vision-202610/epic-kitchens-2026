"""Download pretrained model weights.

Downloads checkpoints from public GitHub/HuggingFace releases into
``checkpoints/pretrained/``.  Re-running is safe: existing files are skipped.

Usage:
    python data/download/download_pretrained.py
    python data/download/download_pretrained.py --output-dir checkpoints/pretrained
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------------------
# Registry of available pretrained weights
# ---------------------------------------------------------------------------
# Each entry: (filename, url, description)
# URLs point to public release assets.  Update if upstream links change.
PRETRAINED_WEIGHTS: list[tuple[str, str, str]] = [
    # AVION ViT-L/14 (Ego4D + LaViLa pretraining)
    # Placeholder URL — replace with the actual asset URL from:
    # https://github.com/zhaoyue-zephyrus/AVION/releases
    (
        "avion_vitl14.pt",
        "https://github.com/zhaoyue-zephyrus/AVION/releases/download/v0.1/avion_vitl14_ego4d_lavila.pt",
        "AVION ViT-L/14 (Ego4D+LaViLa pretrained)",
    ),
]

DEFAULT_OUTPUT_DIR = "checkpoints/pretrained"


def _download_file(url: str, dest: Path) -> None:
    """Download a single file with a simple progress indicator.

    Args:
        url: Source URL.
        dest: Destination file path.
    """
    print(f"  Downloading {dest.name} ...")
    try:
        def _progress(block: int, block_size: int, total: int) -> None:
            if total > 0:
                pct = min(100, block * block_size * 100 // total)
                print(f"\r    {pct:3d}%", end="", flush=True)

        urlretrieve(url, str(dest), reporthook=_progress)  # noqa: S310
        print(f"\r    Done → {dest}")
    except Exception as exc:  # noqa: BLE001
        print(f"\r    Failed: {exc}")
        if dest.exists():
            dest.unlink()  # remove partial file
        raise


def download_pretrained(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    """Download all available pretrained weights.

    Files that already exist (non-empty) are skipped.

    Args:
        output_dir: Directory to save the downloaded weights.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    any_failed = False
    for filename, url, description in PRETRAINED_WEIGHTS:
        dest = out_path / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {filename} already exists.")
            continue

        print(f"\n{description}")
        print(f"  Source: {url}")
        try:
            _download_file(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  Could not download {filename}: {exc}")
            print("  Download manually and place at:", dest)
            any_failed = True

    if any_failed:
        print(
            "\nSome weights could not be downloaded automatically."
            "\nSee messages above for manual download instructions."
        )
        sys.exit(1)
    else:
        print(f"\nAll pretrained weights are ready in {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretrained model weights")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save pretrained weights (default: checkpoints/pretrained)",
    )
    args = parser.parse_args()
    download_pretrained(args.output_dir)
