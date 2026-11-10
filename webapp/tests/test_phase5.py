"""Phase-5 pieces: pure/offline logic of the bandit, ANN benchmark, scaling and the reranker.

DB-backed (ann_search) and Gemini-backed (rerank call) paths are integration-only and skipped
here; these guard the deterministic logic that does not need a database or API key.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_thompson_bandit_learns_a_good_arm():
    """A cuisine that always rewards should dominate the posterior after a few updates."""
    from recsys.eval.bandit import ThompsonCuisineBandit
    b = ThompsonCuisineBandit()
    seg = (0, 2)
    rng = np.random.default_rng(0)
    avail = np.arange(5)
    good = 3
    # feed: good arm rewards 0.9, others 0.0
    for _ in range(200):
        arm = b.choose(seg, avail, rng)
        b.update(seg, arm, 0.9 if arm == good else 0.0)
    a, beta = b._post(seg)
    posterior_mean = a / (a + beta)
    assert np.argmax(posterior_mean) == good
    assert posterior_mean[good] > 0.6


def test_bandit_choose_respects_availability():
    from recsys.eval.bandit import ThompsonCuisineBandit
    b = ThompsonCuisineBandit()
    rng = np.random.default_rng(1)
    avail = np.array([1, 4])
    for _ in range(50):
        assert b.choose((9, 9), avail, rng) in (1, 4)


def test_ann_recall_and_exact_topk():
    from recsys.eval.ann_bench import _exact_topk, _recall
    vecs = np.eye(10, dtype=np.float32)          # orthonormal: nearest to row i is itself
    ids = np.arange(10)
    q = vecs[3]
    truth = _exact_topk(vecs, ids, q, 1)
    assert truth == {3}
    # recall: perfect vs half-missing
    assert _recall([{1, 2, 3, 4}], [{1, 2, 3, 4}], 4) == 1.0
    assert _recall([{1, 2}], [{1, 2, 3, 4}], 4) == 0.5


def test_scaling_projection_shapes_and_monotonic():
    from recsys.eval.scaling import _project
    base = {"n_items": 1000, "n_users": 1000, "n_orders": 10000, "dim": 768,
            "serve_ms": 10.0, "train_seconds": 30.0, "train_rows": 100000,
            "vector_mb_per_space": 3.0}
    rows = _project(base)
    assert len(rows) == 4
    # training time grows with users; serving latency grows far slower (sublinear in items)
    serve = [r["serve_ms_per_request"] for r in rows]
    train = [r["train_seconds"] for r in rows]
    assert train == sorted(train)                       # non-decreasing
    assert serve == sorted(serve)
    assert serve[-1] < serve[0] * 1.5                   # latency barely moves at 60x catalogue
    assert train[-1] > train[0] * 10                    # training scales with users


def test_reranker_order_validation():
    from recsys.eval.gemini_rerank import _rerank_order
    assert _rerank_order({"order": [2, 0, 1]}, 3) == [2, 0, 1]
    # missing an index -> completed into a full permutation
    assert set(_rerank_order({"order": [2, 0]}, 3)) == {0, 1, 2}
    # duplicates dropped, then completed
    assert set(_rerank_order({"order": [1, 1, 0]}, 3)) == {0, 1, 2}
    # out-of-range ignored
    assert set(_rerank_order({"order": [9, 0, 1, 2]}, 3)) == {0, 1, 2}
    # malformed
    assert _rerank_order({}, 3) is None
    assert _rerank_order({"order": "nope"}, 3) is None


def test_pipeline_rank_norm():
    from recsys.pipeline_serve import PipelineService
    rn = PipelineService._rank_norm
    out = rn(np.array([0.0, 0.0, 0.0]))
    assert np.all(out == 0.0)                            # all-equal -> all zero (no signal)
    out2 = rn(np.array([5.0, 1.0, 3.0]))
    assert out2.argmax() == 0 and out2.argmin() == 1     # highest value -> highest percentile
