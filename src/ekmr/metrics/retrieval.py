"""Retrieval metrics: mAP and nDCG for multi-instance retrieval."""

from __future__ import annotations

import numpy as np


def compute_map(
    sim_matrix: np.ndarray,
    relevancy_matrix: np.ndarray,
    topk: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Mean Average Precision at various K values.

    Args:
        sim_matrix: Similarity matrix of shape (N_queries, N_gallery).
        relevancy_matrix: Relevancy matrix of shape (N_queries, N_gallery).
        topk: Tuple of K values for mAP@K.

    Returns:
        Dictionary with mAP values at each K and overall mAP.
    """
    n_queries = sim_matrix.shape[0]
    sorted_indices = np.argsort(-sim_matrix, axis=1)
    results: dict[str, float] = {}

    for k in topk:
        ap_sum = 0.0
        for i in range(n_queries):
            top_indices = sorted_indices[i, :k]
            rel = relevancy_matrix[i, top_indices]
            if rel.sum() == 0:
                continue
            cumsum = np.cumsum(rel > 0)
            precision_at_rank = cumsum / np.arange(1, k + 1)
            ap_sum += (precision_at_rank * (rel > 0)).sum() / min(
                k, (relevancy_matrix[i] > 0).sum()
            )
        results[f"mAP@{k}"] = float(ap_sum / n_queries)

    # Full mAP (over all gallery items)
    n_gallery = sim_matrix.shape[1]
    ap_sum_full = 0.0
    for i in range(n_queries):
        rel = relevancy_matrix[i, sorted_indices[i]]
        if rel.sum() == 0:
            continue
        cumsum = np.cumsum(rel > 0)
        precision_at_rank = cumsum / np.arange(1, n_gallery + 1)
        ap_sum_full += (precision_at_rank * (rel > 0)).sum() / (relevancy_matrix[i] > 0).sum()
    results["mAP"] = float(ap_sum_full / n_queries)

    return results


def compute_ndcg(
    sim_matrix: np.ndarray,
    relevancy_matrix: np.ndarray,
    topk: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Normalized Discounted Cumulative Gain at various K values.

    Args:
        sim_matrix: Similarity matrix of shape (N_queries, N_gallery).
        relevancy_matrix: Relevancy matrix of shape (N_queries, N_gallery).
        topk: Tuple of K values for nDCG@K.

    Returns:
        Dictionary with nDCG values at each K and overall nDCG.
    """
    n_queries = sim_matrix.shape[0]
    sorted_indices = np.argsort(-sim_matrix, axis=1)
    results: dict[str, float] = {}

    for k in topk:
        ndcg_sum = 0.0
        for i in range(n_queries):
            top_indices = sorted_indices[i, :k]
            rel = relevancy_matrix[i, top_indices]
            dcg = _dcg(rel)

            ideal_rel = np.sort(relevancy_matrix[i])[::-1][:k]
            idcg = _dcg(ideal_rel)

            if idcg > 0:
                ndcg_sum += dcg / idcg
        results[f"nDCG@{k}"] = float(ndcg_sum / n_queries)

    # Full nDCG
    ndcg_sum_full = 0.0
    for i in range(n_queries):
        rel = relevancy_matrix[i, sorted_indices[i]]
        dcg = _dcg(rel)
        ideal_rel = np.sort(relevancy_matrix[i])[::-1]
        idcg = _dcg(ideal_rel)
        if idcg > 0:
            ndcg_sum_full += dcg / idcg
    results["nDCG"] = float(ndcg_sum_full / n_queries)

    return results


def _dcg(relevances: np.ndarray) -> float:
    """Compute Discounted Cumulative Gain.

    Args:
        relevances: Array of relevance scores.

    Returns:
        DCG value.
    """
    positions = np.arange(1, len(relevances) + 1)
    discounts = np.log2(positions + 1)
    return float(np.sum(relevances / discounts))


def compute_all_metrics(
    sim_v2t: np.ndarray,
    sim_t2v: np.ndarray,
    relevancy_matrix: np.ndarray,
    topk: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute all retrieval metrics for both directions.

    Args:
        sim_v2t: Video-to-text similarity matrix (N, M).
        sim_t2v: Text-to-video similarity matrix (M, N).
        relevancy_matrix: Relevancy matrix (N, M).
        topk: Tuple of K values.

    Returns:
        Dictionary with all metric values including averages.
    """
    v2t_map = compute_map(sim_v2t, relevancy_matrix, topk)
    v2t_ndcg = compute_ndcg(sim_v2t, relevancy_matrix, topk)

    t2v_map = compute_map(sim_t2v, relevancy_matrix.T, topk)
    t2v_ndcg = compute_ndcg(sim_t2v, relevancy_matrix.T, topk)

    results: dict[str, float] = {}
    for k, v in v2t_map.items():
        results[f"v2t_{k}"] = v
    for k, v in v2t_ndcg.items():
        results[f"v2t_{k}"] = v
    for k, v in t2v_map.items():
        results[f"t2v_{k}"] = v
    for k, v in t2v_ndcg.items():
        results[f"t2v_{k}"] = v

    results["avg_mAP"] = (results.get("v2t_mAP", 0.0) + results.get("t2v_mAP", 0.0)) / 2.0
    results["avg_nDCG"] = (results.get("v2t_nDCG", 0.0) + results.get("t2v_nDCG", 0.0)) / 2.0

    return results
