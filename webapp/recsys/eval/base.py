"""The recommender interface every algorithm implements, and the request/result records.

A recommender has two phases:
  fit(ctx)                 learn from orders before the test period (training cost is measured here)
  recommend(req, cand)     score the candidate main dishes for one request (serving cost is measured here)

`recommend` returns a score per candidate (higher = better); the harness sorts and truncates.
Candidates are the main dishes available at the request (open restaurants that deliver there),
so every recommender ranks the same set and only availability-respecting lists are produced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..data import Dataset, Request


@dataclass(frozen=True)
class FitContext:
    """What a recommender may learn from: the dataset and the cutoff day (exclusive).

    Only orders with day < cutoff_day may be used, so nothing from the test period leaks in.
    """
    data: Dataset
    cutoff_day: int


class Recommender(Protocol):
    name: str

    def fit(self, ctx: FitContext) -> None: ...

    def recommend(self, req: Request, cand: np.ndarray) -> np.ndarray:
        """Score each candidate item id in `cand`; return a float array of the same length."""
        ...


class BaseRecommender:
    """Convenience base: holds the dataset, no-op fit, requires recommend()."""
    name: str = "base"

    def fit(self, ctx: FitContext) -> None:
        self.data = ctx.data
        self.cutoff_day = ctx.cutoff_day

    def recommend(self, req: Request, cand: np.ndarray) -> np.ndarray:
        raise NotImplementedError
