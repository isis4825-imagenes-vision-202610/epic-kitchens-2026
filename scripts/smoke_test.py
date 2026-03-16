"""Smoke test — end-to-end pipeline validator.

Runs the full pipeline (annotations download → training → evaluation →
export → demo) using synthetic/minimal data so the entire flow can be
verified in a few minutes without downloading the full dataset.

Usage:
    # Full smoke test (downloads annotations if not present)
    python scripts/smoke_test.py

    # Skip the annotation download step (data already present)
    python scripts/smoke_test.py --skip-download
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SMOKE_OUTPUT = "experiments/smoke_test"

STEPS_ALWAYS = [
    (
        "Dataset verify (synthetic OK)",
        [sys.executable, "data/verify.py", "--root", "data/raw/EK100"],
    ),
    (
        "Train 3 steps (smoke)",
        [
            sys.executable,
            "scripts/train.py",
            "model=cr_clip",
            "training.epochs=1",
            "training.max_steps=3",
            "training.ddp=false",
            "training.amp=false",
            "dataset.batch_size=2",
            "dataset.num_workers=0",
            "logging.wandb=false",
            f"output_dir={SMOKE_OUTPUT}",
        ],
    ),
    (
        "Evaluate on test split",
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            f"{SMOKE_OUTPUT}/checkpoints/epoch_001.pt",
            "--split",
            "test",
            "--output-dir",
            SMOKE_OUTPUT,
        ],
    ),
    (
        "Export submission",
        [
            sys.executable,
            "scripts/export_submission.py",
            "--checkpoint",
            f"{SMOKE_OUTPUT}/checkpoints/epoch_001.pt",
            "--output-dir",
            f"{SMOKE_OUTPUT}/submissions",
        ],
    ),
    (
        "Demo retrieval",
        [
            sys.executable,
            "scripts/demo.py",
            "--output-dir",
            f"{SMOKE_OUTPUT}/demo",
        ],
    ),
]

DOWNLOAD_STEP = (
    "Annotations download",
    [sys.executable, "data/download/download_annotations.py"],
)


def run_step(desc: str, cmd: list[str]) -> None:
    """Run a single pipeline step, exiting on failure.

    Args:
        desc: Human-readable description of the step.
        cmd: Command to execute.
    """
    print(f"\n{'=' * 60}")
    print(f"STEP: {desc}")
    print(f"CMD : {' '.join(cmd)}")
    result = subprocess.run(cmd)  # noqa: S603
    if result.returncode != 0:
        print(f"\n✗ FAILED: {desc}")
        sys.exit(result.returncode)
    print(f"✓ PASSED: {desc}")


def main() -> None:
    """Entry point for the smoke test runner."""
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        default=False,
        help="Skip the annotation download step (use when data is already present)",
    )
    args = parser.parse_args()

    steps = []
    if not args.skip_download:
        steps.append(DOWNLOAD_STEP)
    steps.extend(STEPS_ALWAYS)

    # Ensure the smoke output directory is clean so checkpoint paths are predictable
    import shutil

    smoke_dir = Path(SMOKE_OUTPUT)
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)

    for desc, cmd in steps:
        run_step(desc, cmd)

    print("\n" + "=" * 60)
    print("✓ ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
