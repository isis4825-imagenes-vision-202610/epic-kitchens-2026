"""Build Codabench-compatible submission zip files."""

from __future__ import annotations

import pickle
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def build_submission(
    sim_v2t: np.ndarray,
    sim_t2v: np.ndarray,
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """Build a Codabench submission zip from similarity matrices.

    Args:
        sim_v2t: Video-to-text similarity matrix of shape (N, M).
        sim_t2v: Text-to-video similarity matrix of shape (M, N).
        output_dir: Directory to save the submission zip.
        filename: Optional custom filename for the zip.

    Returns:
        Path to the created submission zip file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"submission_{timestamp}.zip"

    results = {
        "v2t": sim_v2t.astype(np.float32),
        "t2v": sim_t2v.astype(np.float32),
    }

    pkl_path = output_dir / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    zip_path = output_dir / filename
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pkl_path, "results.pkl")

    pkl_path.unlink()
    return zip_path
