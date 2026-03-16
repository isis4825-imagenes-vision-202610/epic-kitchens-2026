"""Download EPIC-KITCHENS-100 dataset using the official downloader.

This script automates the download of the EPIC-KITCHENS-100 dataset,
including videos, RGB frames, and action-retrieval metadata.

Usage:
    # Full dataset (all participants)
    python data/download/download_ek100.py

    # Subset (specific participants)
    python data/download/download_ek100.py --participants P01 P02 P03

    # Only RGB frames (skip original videos)
    python data/download/download_ek100.py --rgb-frames-only

References:
    https://github.com/epic-kitchens/epic-kitchens-download-scripts
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


DOWNLOADER_REPO = "https://github.com/epic-kitchens/epic-kitchens-download-scripts.git"
DEFAULT_OUTPUT_DIR = "data/raw/EK100"


def _ensure_downloader(work_dir: Path) -> Path:
    """Clone or update the official download-scripts repo.

    Args:
        work_dir: Directory to clone the downloader into.

    Returns:
        Path to the cloned repository.
    """
    repo_dir = work_dir / "epic-kitchens-download-scripts"
    if repo_dir.exists():
        print(f"Updating existing downloader repo at {repo_dir} ...")
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
    else:
        print(f"Cloning official downloader to {repo_dir} ...")
        subprocess.run(
            ["git", "clone", "--depth=1", DOWNLOADER_REPO, str(repo_dir)],
            check=True,
        )
    return repo_dir


def _run_downloader(
    repo_dir: Path,
    output_dir: Path,
    participants: list[str] | None,
    videos: bool,
    rgb_frames: bool,
    flow_frames: bool,
) -> None:
    """Run the official epic_downloader.py script.

    Args:
        repo_dir: Path to the cloned downloader repository.
        output_dir: Target directory for downloaded files.
        participants: Optional list of participant IDs to restrict download.
        videos: Whether to download original videos.
        rgb_frames: Whether to download RGB frames.
        flow_frames: Whether to download optical flow frames.
    """
    downloader = repo_dir / "epic_downloader.py"
    if not downloader.exists():
        raise FileNotFoundError(f"Downloader script not found at {downloader}")

    cmd: list[str] = [sys.executable, str(downloader)]

    # Output directory — pass absolute path so the downloader works regardless of CWD
    cmd += ["--output-path", str(output_dir.resolve())]

    # Content flags
    if videos:
        cmd.append("--videos")
    if rgb_frames:
        cmd.append("--rgb-frames")
    if flow_frames:
        cmd.append("--flow-frames")

    # Always include action-retrieval annotations subset
    cmd.append("--action-retrieval")

    # Participant filter — the official downloader expects a single comma-separated string
    if participants:
        cmd += ["--participants", ",".join(participants)]

    print("Running official downloader:")
    print("  " + " ".join(cmd))
    print()

    try:
        # Run from the downloader repo directory so relative paths inside the
        # script (e.g. data/epic_55_splits.csv) resolve correctly.
        subprocess.run(cmd, check=True, cwd=str(repo_dir))
    except subprocess.CalledProcessError as exc:
        print(f"\nDownload failed (exit code {exc.returncode}).")
        print("Partial downloads may be resumed by re-running this script.")
        sys.exit(exc.returncode)


def download_ek100(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    participants: list[str] | None = None,
    videos: bool = True,
    rgb_frames: bool = True,
    flow_frames: bool = False,
    downloader_cache: str | None = None,
) -> None:
    """Download the EPIC-KITCHENS-100 dataset.

    Downloads the requested media using the official epic-kitchens downloader
    script.  Handles partial downloads automatically—simply re-run to resume.

    Args:
        output_dir: Target root directory for all downloaded files.
        participants: Participant IDs to download (e.g. ['P01', 'P02']).
                      If None, downloads all participants (~740 GB total).
        videos: Download original .MP4 video files.
        rgb_frames: Download pre-extracted RGB frame images.
        flow_frames: Download pre-extracted optical flow frames.
        downloader_cache: Directory to cache the downloader repo.
                          Defaults to a temporary directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if downloader_cache:
        cache_path = Path(downloader_cache)
        cache_path.mkdir(parents=True, exist_ok=True)
        repo_dir = _ensure_downloader(cache_path)
        _run_downloader(repo_dir, out_path, participants, videos, rgb_frames, flow_frames)
    else:
        with tempfile.TemporaryDirectory(prefix="ek100_downloader_") as tmp:
            repo_dir = _ensure_downloader(Path(tmp))
            _run_downloader(repo_dir, out_path, participants, videos, rgb_frames, flow_frames)

    print(f"\nDownload complete.  Files saved to: {out_path.resolve()}")
    print("Run  python data/verify.py --root", output_dir, " to validate the dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download EK100 dataset using the official downloader"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for downloaded files (default: data/raw/EK100)",
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        type=str,
        default=None,
        help="Participant IDs to download (e.g. P01 P02). Omit for all participants.",
    )
    parser.add_argument(
        "--no-videos",
        dest="videos",
        action="store_false",
        default=True,
        help="Skip original video download",
    )
    parser.add_argument(
        "--rgb-frames-only",
        action="store_true",
        default=False,
        help="Download only RGB frames (skips videos and flow)",
    )
    parser.add_argument(
        "--flow-frames",
        action="store_true",
        default=False,
        help="Also download optical flow frames",
    )
    parser.add_argument(
        "--downloader-cache",
        type=str,
        default=None,
        help="Directory to persist the downloader repo between runs",
    )
    args = parser.parse_args()

    download_ek100(
        output_dir=args.output_dir,
        participants=args.participants,
        videos=not args.rgb_frames_only and args.videos,
        rgb_frames=True,  # always download RGB frames (required for training)
        flow_frames=args.flow_frames,
        downloader_cache=args.downloader_cache,
    )
