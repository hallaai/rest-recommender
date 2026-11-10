"""Stage-2 ranking with LightGBM, comparing three learning-to-rank losses.

The ranker scores the pooled candidates from the retrieval sources using the ~40 features in
features.py. It is trained on the validation month (September): for each request there, the
retrieval models (fitted on May-Aug) propose candidates, features are built, and the ordered
dish is the positive. Three losses are compared:

  pointwise   binary logloss per (request, dish): predict P(order), then sort. Ignores ranking.
  pairwise    LambdaRank: push the ordered dish above each non-ordered one in the same request.
  listwise    rank_xendcg (cross-entropy over the list): optimises NDCG of the whole list.

rank_xendcg is the one tuned for "get the top of the list right", which is what a 5-slot screen
needs. DCNv2 (dcn.py) is the neural challenger on the same features.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from ..data import Dataset, Request
from .features import FEATURE_NAMES, FeatureBuilder

log = logging.getLogger(__name__)
LOSSES = {"pointwise": "binary", "pairwise": "lambdarank", "listwise": "rank_xendcg"}


@dataclass
class TrainSet:
    X: np.ndarray
    y: np.ndarray
    groups: list[int]        # rows per request


def build_training_data(data: Dataset, retrievers: dict, fb: FeatureBuilder, requests: list[Request],
                        cands: list[np.ndarray]) -> TrainSet:
    """Features + labels for a list of requests (candidates already available-filtered)."""
    X_parts, y_parts, groups = [], [], []
    for req, cand in zip(requests, cands):
        retr = {name: rec.recommend(req, cand) for name, rec in retrievers.items()}
        retr["content"] = retr.pop("content_gemini", np.zeros(len(cand)))
        X_parts.append(fb.row(req, cand, retr))
        y_parts.append((cand == req.target).astype(np.float32))
        groups.append(len(cand))
    return TrainSet(np.vstack(X_parts), np.concatenate(y_parts), groups)


class LGBMRanker:
    """LightGBM ranker with a selectable loss. The pointwise variant uses the classifier."""

    def __init__(self, loss: str = "listwise", num_leaves: int = 63, n_estimators: int = 300,
                 lr: float = 0.05, seed: int = 0) -> None:
        assert loss in LOSSES
        self.loss = loss
        self.name = f"ranker_{loss}"
        self.params = dict(num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=lr,
                           random_state=seed, n_jobs=-1, verbose=-1)

    def fit(self, train: TrainSet) -> None:
        import lightgbm as lgb
        t0 = time.perf_counter()
        if self.loss == "pointwise":
            self.model = lgb.LGBMClassifier(objective="binary", **self.params)
            self.model.fit(train.X, train.y)
        else:
            self.model = lgb.LGBMRanker(objective=LOSSES[self.loss],
                                        label_gain=[0, 1], **self.params)  # binary relevance
            self.model.fit(train.X, train.y.astype(int), group=train.groups)
        self.train_seconds = time.perf_counter() - t0
        self.feature_importance = dict(zip(FEATURE_NAMES, self.model.feature_importances_.tolist()))

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.loss == "pointwise":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)
