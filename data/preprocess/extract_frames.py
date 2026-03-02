"""Extract and resize video frames for training.

Processes videos into 15-second chunks at 288px shorter side.

Usage:
    python data/preprocess/extract_frames.py \
        --input data/raw/EK100/videos --output data/processed/frames
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def extract_frames(
    input_dir: str,
    output_dir: str,
    shorter_side: int = 288,
    chunk_seconds: int = 15,
) -> None:
    """Extract and resize frames from video files.

    Args:
        input_dir: Directory containing input video files.
        output_dir: Directory to save extracted frames.
        shorter_side: Target size for the shorter side of frames.
        chunk_seconds: Duration of each video chunk in seconds.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    video_files = list(in_path.glob("**/*.MP4")) + list(in_path.glob("**/*.mp4"))
    print(f"Found {len(video_files)} video files")

    scale_filter = f"scale='if(gt(iw,ih),{shorter_side},-2)':'if(gt(iw,ih),-2,{shorter_side})'"

    for video_path in video_files:
        relative = video_path.relative_to(in_path)
        out_video_dir = out_path / relative.stem
        out_video_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", scale_filter,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            str(out_video_dir / f"{relative.stem}_%03d.jpg"),
        ]
        print(f"Processing: {video_path.name}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"  Failed: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--shorter-side", type=int, default=288)
    args = parser.parse_args()
    extract_frames(args.input, args.output, args.shorter_side)
