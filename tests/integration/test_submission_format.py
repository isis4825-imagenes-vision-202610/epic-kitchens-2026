"""Integration test for submission export format."""

from __future__ import annotations

import pickle
import tempfile
import zipfile

import numpy as np

from ekmr.utils.export import build_submission


class TestSubmissionFormat:
    """Tests for Codabench submission format."""

    def test_zip_contains_results(self):
        """Submission zip should contain results.pkl."""
        sim_v2t = np.random.default_rng(42).random((100, 100)).astype(np.float32)
        sim_t2v = np.random.default_rng(42).random((100, 100)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = build_submission(sim_v2t, sim_t2v, tmpdir, "test_sub.zip")
            assert zip_path.exists()

            with zipfile.ZipFile(zip_path, "r") as zf:
                assert "results.pkl" in zf.namelist()

    def test_results_pkl_keys(self):
        """results.pkl should contain v2t and t2v keys."""
        sim_v2t = np.random.default_rng(42).random((50, 50)).astype(np.float32)
        sim_t2v = np.random.default_rng(42).random((50, 50)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = build_submission(sim_v2t, sim_t2v, tmpdir, "test_sub.zip")

            with zipfile.ZipFile(zip_path, "r") as zf, zf.open("results.pkl") as f:
                results = pickle.load(f)  # noqa: S301

            assert "v2t" in results
            assert "t2v" in results
            assert results["v2t"].shape == (50, 50)
            assert results["t2v"].shape == (50, 50)
            assert results["v2t"].dtype == np.float32

    def test_results_shapes_asymmetric(self):
        """Should handle asymmetric similarity matrices."""
        sim_v2t = np.random.default_rng(42).random((30, 50)).astype(np.float32)
        sim_t2v = np.random.default_rng(42).random((50, 30)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = build_submission(sim_v2t, sim_t2v, tmpdir)
            assert zip_path.exists()

            with zipfile.ZipFile(zip_path, "r") as zf, zf.open("results.pkl") as f:
                results = pickle.load(f)  # noqa: S301

            assert results["v2t"].shape == (30, 50)
            assert results["t2v"].shape == (50, 30)
