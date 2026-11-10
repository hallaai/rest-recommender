"""Per-request metric values and user-level bootstrap confidence intervals.

The evaluator produces, for each request, a small record of raw numbers (hit@5, ndcg@5,
reciprocal rank, demand captured, best-possible demand, revenue, and behaviour counts). This
module aggregates those across requests and attaches 95% confidence intervals by resampling
users (a user and all their requests move together, because one user's requests are not
independent). Differences between two algorithms use the same user resamples (paired).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

K = 5
CAND_K = 100       # candidate-stage cutoff for recall@100


@dataclass(frozen=True)
class RequestEval:
    """Raw numbers for one request under one algorithm (all cheap to average)."""
    user: int
    hit: float               # ordered dish in top-5
    ndcg: float              # 1/log2(rank+1) if ordered dish in top-5 else 0
    rr: float                # reciprocal rank of the ordered dish over the whole list
    recall100: float         # ordered dish in top-100
    rest_hit: float          # a dish from the ordered restaurant in top-5
    demand: float            # true demand captured by top-5 (0..1)
    best_demand: float       # demand captured by the best-possible top-5
    revenue: float           # expected revenue (EUR) from top-5 under true demand
    best_revenue: float
    is_reorder: float        # top-1 dish is one the user ordered before
    n_rest_top5: float       # distinct restaurants in top-5
    n_cuisine_top5: float    # distinct cuisines in top-5
    pop_pctile: float        # mean popularity percentile of top-5 (0..1)
    top5: tuple[int, ...] = field(default=())    # the recommended item ids (for overlap metrics)


def ndcg_at_k(rank: int) -> float:
    """rank is 1-based position of the ordered dish, or 0 if not in top-K."""
    return 1.0 / np.log2(rank + 1) if 1 <= rank <= K else 0.0


# -- aggregation --------------------------------------------------------------------
POINT_METRICS = ("hit", "ndcg", "rr", "recall100", "rest_hit", "demand", "revenue",
                 "is_reorder", "n_rest_top5", "n_cuisine_top5", "pop_pctile")


def _by_user(evals: list[RequestEval], field_name: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for e in evals:
        out.setdefault(e.user, []).append(getattr(e, field_name))
    return out


def _user_means(evals: list[RequestEval], field_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-user mean of a field, and the per-user request count (weights)."""
    by = _by_user(evals, field_name)
    users = sorted(by)
    means = np.array([np.mean(by[u]) for u in users])
    weights = np.array([len(by[u]) for u in users], dtype=float)
    return means, weights


def weighted_mean(means: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(means, weights=weights)) if len(means) else 0.0


def bootstrap_ci(evals: list[RequestEval], field_name: str, n_boot: int = 1000,
                 seed: int = 0) -> dict[str, float]:
    """Mean of a per-request field with a 95% user-level bootstrap interval.

    Resamples users with replacement; each draw averages that user's requests, then takes the
    request-weighted mean. Reported mean is the request-level mean (not resampled)."""
    means, weights = _user_means(evals, field_name)
    point = weighted_mean(means, weights)
    if len(means) < 2:
        return {"mean": point, "lo": point, "hi": point}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(means), size=(n_boot, len(means)))
    boot = np.array([weighted_mean(means[i], weights[i]) for i in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": point, "lo": float(lo), "hi": float(hi)}


def paired_diff_ci(a: list[RequestEval], b: list[RequestEval], field_name: str,
                   n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """95% interval for (a - b) on a field, resampling the same users for both.

    a and b must be aligned request-for-request (same requests, two algorithms)."""
    assert len(a) == len(b)
    da = _by_user(a, field_name)
    db = _by_user(b, field_name)
    users = sorted(da)
    am = np.array([np.mean(da[u]) for u in users])
    bm = np.array([np.mean(db[u]) for u in users])
    w = np.array([len(da[u]) for u in users], dtype=float)
    point = weighted_mean(am, w) - weighted_mean(bm, w)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(users), size=(n_boot, len(users)))
    boot = np.array([weighted_mean(am[i], w[i]) - weighted_mean(bm[i], w[i]) for i in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": point, "lo": float(lo), "hi": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def pct_of_best(evals: list[RequestEval]) -> float:
    """Captured demand as a share of the best-possible demand (request-weighted)."""
    cap = np.array([e.demand for e in evals])
    best = np.array([e.best_demand for e in evals])
    return float(cap.sum() / best.sum()) if best.sum() > 0 else 0.0


def revenue_per_1000(evals: list[RequestEval]) -> float:
    """Expected revenue in EUR per 1000 sessions (mean top-5 revenue x 1000)."""
    return float(np.mean([e.revenue for e in evals]) * 1000) if evals else 0.0


def catalog_coverage(evals: list[RequestEval], n_mains: int) -> float:
    shown = {i for e in evals for i in e.top5}
    return len(shown) / n_mains if n_mains else 0.0


def gini(exposure: np.ndarray) -> float:
    """Gini of how often each restaurant appears in recommendations (0 = equal, 1 = concentrated)."""
    x = np.sort(exposure[exposure >= 0].astype(float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def mean_overlap(a: list[RequestEval], b: list[RequestEval]) -> float:
    """Average Jaccard of two algorithms' top-5 lists over the same requests."""
    return float(np.mean([jaccard(x.top5, y.top5) for x, y in zip(a, b)])) if a else 0.0
