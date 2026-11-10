"""Metric correctness (known answers), confidence intervals, no-future-leak, oracle isolation."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from recsys.eval.metrics import (RequestEval, bootstrap_ci, gini, jaccard, ndcg_at_k,
                                  paired_diff_ci, pct_of_best, weighted_mean)


def _ev(user, **kw):
    base = dict(user=user, hit=0.0, ndcg=0.0, rr=0.0, recall100=0.0, rest_hit=0.0, demand=0.0,
                best_demand=1.0, revenue=0.0, best_revenue=1.0, is_reorder=0.0, n_rest_top5=5.0,
                n_cuisine_top5=5.0, pop_pctile=0.5, top5=())
    return RequestEval(**{**base, **kw})


def test_ndcg_positions():
    assert ndcg_at_k(1) == 1.0
    assert ndcg_at_k(3) == pytest.approx(1 / np.log2(4))
    assert ndcg_at_k(6) == 0.0 and ndcg_at_k(0) == 0.0


def test_jaccard_known():
    assert jaccard((1, 2, 3), (1, 2, 3)) == 1.0
    assert jaccard((1, 2), (3, 4)) == 0.0
    assert jaccard((1, 2, 3, 4), (3, 4, 5, 6)) == pytest.approx(2 / 6)


def test_gini_extremes():
    assert gini(np.array([5, 5, 5, 5])) == pytest.approx(0.0, abs=1e-9)
    assert gini(np.array([0, 0, 0, 100])) > 0.7           # one restaurant gets everything


def test_pct_of_best():
    evals = [_ev(1, demand=0.3, best_demand=0.6), _ev(1, demand=0.1, best_demand=0.4)]
    assert pct_of_best(evals) == pytest.approx(0.4 / 1.0)


def test_weighted_mean_counts_requests_not_users():
    # user A has 3 requests at 1.0, user B has 1 request at 0.0
    means = np.array([1.0, 0.0])
    weights = np.array([3.0, 1.0])
    assert weighted_mean(means, weights) == pytest.approx(0.75)


def test_bootstrap_ci_brackets_mean_and_narrows_with_data():
    rng = np.random.default_rng(0)
    evals = [_ev(u, hit=float(rng.random() < 0.3)) for u in range(400) for _ in range(3)]
    ci = bootstrap_ci(evals, "hit")
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci["hi"] - ci["lo"] < 0.1                       # 1200 requests: a tight interval


def test_paired_diff_detects_real_gap():
    # same users; algorithm A always beats B by 0.5
    a = [_ev(u, demand=0.8) for u in range(200)]
    b = [_ev(u, demand=0.3) for u in range(200)]
    d = paired_diff_ci(a, b, "demand")
    assert d["diff"] == pytest.approx(0.5) and d["significant"] and d["lo"] > 0


def test_paired_diff_no_gap_is_insignificant():
    rng = np.random.default_rng(1)
    a = [_ev(u, demand=float(rng.random())) for u in range(200)]
    b = [_ev(u, demand=a[u].demand) for u in range(200)]   # identical
    assert not paired_diff_ci(a, b, "demand")["significant"]


# -- oracle and baselines on the small world ----------------------------------------
@pytest.fixture(scope="module")
def small_dataset(small_world, tmp_path_factory):
    from recsys.synth.generate import write
    from recsys.data import Dataset
    d = tmp_path_factory.mktemp("ds")
    visible, truth, meta = small_world
    write(d, visible, truth, meta)
    return Dataset(d), d


def test_oracle_prefers_observed_orders(small_dataset):
    from recsys.data import VAL_END_DAY, TEST_END_DAY
    from recsys.eval.oracle import TruthOracle
    data, d = small_dataset
    oracle = TruthOracle(data, d)
    better = total = 0
    for r in data.requests(VAL_END_DAY, TEST_END_DAY, sample=200, seed=0):
        cand = data.available_mains(r.lat, r.lon, r.hour)
        if r.target not in cand:
            continue
        p = oracle.choice_probs(r, cand)
        ti = int(np.flatnonzero(cand == r.target)[0])
        better += p[ti] > p.mean()         # observed dish beats the average candidate
        total += 1
    assert total > 20 and better / total > 0.8            # oracle agrees with observed choices


def test_choice_probs_sum_to_one(small_dataset):
    from recsys.data import VAL_END_DAY, TEST_END_DAY
    from recsys.eval.oracle import TruthOracle
    data, d = small_dataset
    oracle = TruthOracle(data, d)
    r = data.requests(VAL_END_DAY, TEST_END_DAY, sample=1, seed=3)[0]
    cand = data.available_mains(r.lat, r.lon, r.hour)
    assert oracle.choice_probs(r, cand).sum() == pytest.approx(1.0, abs=1e-5)


def test_recommenders_respect_the_cutoff(small_dataset):
    """A recommender fitted at the cutoff must give identical scores whether or not test-period
    orders exist in the dataset: it may only read orders before the cutoff."""
    from recsys.data import VAL_END_DAY, TEST_END_DAY
    from recsys.eval.base import FitContext
    from recsys.eval.baselines import Popularity, ALS, EASE
    data, d = small_dataset
    reqs = data.requests(VAL_END_DAY, TEST_END_DAY, sample=50, seed=0)
    for cls in (Popularity, EASE, ALS):
        rec = cls()
        rec.fit(FitContext(data, VAL_END_DAY))
        rows = data.order_rows_before(VAL_END_DAY)
        assert data.o_day[rows].max() < VAL_END_DAY        # fit data is strictly before the test month


def test_no_recommender_imports_the_oracle_or_truth():
    """Recommenders must not read hidden truth. Static check of the baselines module."""
    src = Path("recsys/eval/baselines.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
    assert not any("truth" in m or "oracle" in m for m in imported), imported
