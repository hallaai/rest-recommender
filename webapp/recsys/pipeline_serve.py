"""Live two-stage pipeline service: the real production path, served per request.

This is the fused pipeline the Compare page evaluates (artifacts/ranking.json), made live.
It fits the six retrievers on May-Aug, trains the listwise LightGBM ranker on September (so
there is no leak), then answers a request exactly as ``rank_runner.eval_pipeline`` does — minus
the oracle, which only exists for scoring. The output matches the baselines' shape so the
Recommend page can show it as one more algorithm row ("Fused pipeline").

Fitting is heavier than a single baseline (retrievers + a trained ranker), so the API warms it
in a background thread and shows the fast baselines meanwhile.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .data import Dataset, Request, TRAIN_END_DAY, VAL_END_DAY
from .eval.base import FitContext
from .eval.features import FeatureBuilder
from .eval.metrics import CAND_K, K
from .eval.rank_runner import apply_rules, make_retrievers, pool_candidates
from .eval.ranker import LGBMRanker, build_training_data
from .serve import RecommendService

log = logging.getLogger(__name__)

# The live ranker. Listwise is the best-value loss on the leaderboard (tied quality, half the
# train time of pairwise), so it is what we serve.
LIVE_LOSS = "listwise"
# Far fewer training requests than the offline run (4000): the live row is illustrative, and
# feature-building over thousands of requests is the slow part of the fit. 1200 warms in ~2 min
# and still trains a serviceable ranker; the rigorous numbers live on the Compare page.
TRAIN_SAMPLE = 1200


class PipelineService:
    """Fits the full two-stage pipeline once, then scores requests. Read-only after fit()."""

    def __init__(self, data: Dataset, base: RecommendService | None = None) -> None:
        self.data = data
        # Reuse the baseline service for request-building, business rules and describe/tooltip.
        self.base = base or RecommendService(data)
        self.fb = FeatureBuilder(data)
        self.retrievers: dict = {}
        self.ranker: LGBMRanker | None = None
        self._fit_lock = threading.Lock()
        self._fitted = False
        self.fit_seconds = 0.0
        self.train_rows = 0

    @property
    def name(self) -> str:
        return "pipeline"

    def fit(self) -> None:
        with self._fit_lock:
            if self._fitted:
                return
            t0 = time.perf_counter()
            # Retrievers on train only (May-Aug): September stays clean for the ranker.
            self.retrievers = make_retrievers()
            for rec in self.retrievers.values():
                rec.fit(FitContext(data=self.data, cutoff_day=TRAIN_END_DAY))
            self.fb.set_popularity(TRAIN_END_DAY)

            reqs = self.data.requests(TRAIN_END_DAY, VAL_END_DAY, sample=TRAIN_SAMPLE)
            cands = [self.data.available_mains(r.lat, r.lon, r.hour) for r in reqs]
            keep = [i for i, c in enumerate(cands)
                    if len(c) >= K and reqs[i].target in set(c.tolist())]
            reqs = [reqs[i] for i in keep]
            cands = [cands[i] for i in keep]
            train = build_training_data(self.data, self.retrievers, self.fb, reqs, cands)
            self.train_rows = len(train.y)

            rk = LGBMRanker(loss=LIVE_LOSS)
            rk.fit(train)
            self.ranker = rk
            self._fitted = True
            self.fit_seconds = time.perf_counter() - t0
            log.info("Fitted live pipeline (%s) in %.1fs over %d training rows",
                     LIVE_LOSS, self.fit_seconds, self.train_rows)

    def recommend(self, req: Request, k: int = K) -> dict:
        """Retrieve -> pool -> feature -> rank -> business rules. Returns the baselines' shape."""
        if self.ranker is None:
            raise RuntimeError("Pipeline not fitted")
        cand = self.data.available_mains(req.lat, req.lon, req.hour)
        if len(cand) == 0:
            return {"algorithm": self.name, "items": [], "latency_ms": 0.0, "n_candidates": 0,
                    "no_signal": True}
        t0 = time.perf_counter()
        retr = {name: rec.recommend(req, cand) for name, rec in self.retrievers.items()}
        pooled = pool_candidates(retr, cand, CAND_K)
        pc = cand[pooled]
        feat_in = {name: retr[name][pooled] for name in retr}
        feat_in["content"] = feat_in.pop("content_gemini", np.zeros(len(pc)))
        X = self.fb.row(req, pc, feat_in)
        scores = self.ranker.score(X)
        order = np.argsort(-scores, kind="stable")
        picks = apply_rules(self.data, pc, order, k)
        latency = (time.perf_counter() - t0) * 1000

        known = 0 <= req.user < self.data.n_users
        hist = set(self.data.history(req.user, req.day).items.tolist()) if known else set()
        # Rank-normalise each source across the pool so their contributions are comparable (raw
        # scores have wildly different scales). A dish's "source" is the one where it ranks highest.
        norm = {name: self._rank_norm(feat_in[name]) for name in feat_in}
        # Normalise ranker scores to 0..1 for display (ranker scores are unbounded logits).
        picked_scores = scores[picks]
        lo, hi = float(picked_scores.min()), float(picked_scores.max())
        span = hi - lo or 1.0
        items = []
        for rank, i in enumerate(picks):
            item = int(pc[i])
            disp = 1.0 - rank / max(len(picks), 1) if hi == lo else (float(scores[i]) - lo) / span
            d = self.base._describe(item, disp, "pipeline", req, hist)
            d["reason"] = self._reason(norm, int(i), item in hist)
            items.append(d)
        return {"algorithm": self.name, "items": items, "latency_ms": round(latency, 2),
                "n_candidates": int(len(cand)), "no_signal": False}

    @staticmethod
    def _rank_norm(x: np.ndarray) -> np.ndarray:
        """Percentile rank of each value within the pool (0..1); all-equal -> all zeros."""
        x = np.asarray(x, float)
        if not np.any(x) or x.max() == x.min():
            return np.zeros_like(x)
        return x.argsort().argsort() / max(len(x) - 1, 1)

    _LABEL = {
        "buy_again": "you ordered this before",
        "als": "similar users order this",
        "item_item": "pairs with your past orders",
        "content": "matches your taste profile",
        "popularity": "popular right now",
        "geo_time_pop": "popular nearby at this hour",
    }

    def _reason(self, norm: dict, idx: int, is_reorder: bool) -> str:
        """Name the source where this candidate ranks highest — the fusion, in words."""
        contributions = {name: float(norm[name][idx]) for name in norm}
        top = max(contributions, key=contributions.get) if contributions else None
        base = self._LABEL.get(top, "ranked by the fused model") if top and contributions[top] > 0 \
            else "ranked by the fused model"
        if is_reorder and top != "buy_again":
            base += " · reorder"
        return base
