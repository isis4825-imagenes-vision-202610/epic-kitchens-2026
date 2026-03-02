"""Unit tests for retrieval metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ekmr.metrics.retrieval import compute_all_metrics, compute_map, compute_ndcg


class TestComputeMap:
    """Tests for Mean Average Precision computation."""

    def test_perfect_retrieval(self):
        """Perfect ranking should yield mAP close to 1.0."""
        n = 10
        sim = np.eye(n, dtype=np.float32)
        rel = np.eye(n, dtype=np.float32)
        result = compute_map(sim, rel)
        assert result["mAP"] == pytest.approx(1.0, abs=1e-6)

    def test_zero_relevancy(self):
        """All-zero relevancy should yield mAP of 0."""
        n = 5
        sim = np.random.default_rng(42).random((n, n)).astype(np.float32)
        rel = np.zeros((n, n), dtype=np.float32)
        result = compute_map(sim, rel)
        assert result["mAP"] == pytest.approx(0.0, abs=1e-6)

    def test_output_keys(self):
        """Output should contain expected key names."""
        n = 5
        sim = np.random.default_rng(42).random((n, n)).astype(np.float32)
        rel = np.eye(n, dtype=np.float32)
        result = compute_map(sim, rel, topk=(1, 5))
        assert "mAP@1" in result
        assert "mAP@5" in result
        assert "mAP" in result

    def test_values_in_range(self):
        """All mAP values should be between 0 and 1."""
        n = 10
        rng = np.random.default_rng(42)
        sim = rng.random((n, n)).astype(np.float32)
        rel = (rng.random((n, n)) > 0.7).astype(np.float32)
        result = compute_map(sim, rel)
        for v in result.values():
            assert 0.0 <= v <= 1.0


class TestComputeNdcg:
    """Tests for nDCG computation."""

    def test_perfect_ranking(self):
        """Perfect ranking should yield nDCG of 1.0."""
        n = 10
        rel = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            rel[i, i] = 1.0
        sim = rel.copy()
        result = compute_ndcg(sim, rel)
        assert result["nDCG"] == pytest.approx(1.0, abs=1e-6)

    def test_output_keys(self):
        """Output should contain expected key names."""
        n = 5
        sim = np.random.default_rng(42).random((n, n)).astype(np.float32)
        rel = np.eye(n, dtype=np.float32)
        result = compute_ndcg(sim, rel, topk=(1, 5))
        assert "nDCG@1" in result
        assert "nDCG@5" in result
        assert "nDCG" in result

    def test_graded_relevance(self):
        """Graded relevance values should be handled correctly."""
        n = 5
        rng = np.random.default_rng(42)
        sim = rng.random((n, n)).astype(np.float32)
        rel = rng.random((n, n)).astype(np.float32)  # Graded: [0, 1]
        result = compute_ndcg(sim, rel)
        for v in result.values():
            assert 0.0 <= v <= 1.0


class TestComputeAllMetrics:
    """Tests for combined metrics computation."""

    def test_output_completeness(self):
        """Should return both v2t and t2v metrics plus averages."""
        n = 8
        rng = np.random.default_rng(42)
        sim_v2t = rng.random((n, n)).astype(np.float32)
        sim_t2v = rng.random((n, n)).astype(np.float32)
        rel = np.eye(n, dtype=np.float32)
        result = compute_all_metrics(sim_v2t, sim_t2v, rel)
        assert "avg_mAP" in result
        assert "avg_nDCG" in result
        assert "v2t_mAP" in result
        assert "t2v_mAP" in result

    def test_symmetric_identity(self):
        """With identity relevancy and same sims, v2t and t2v should match."""
        n = 10
        sim = np.eye(n, dtype=np.float32)
        rel = np.eye(n, dtype=np.float32)
        result = compute_all_metrics(sim, sim, rel)
        assert result["v2t_mAP"] == pytest.approx(result["t2v_mAP"], abs=1e-6)
