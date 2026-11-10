"""The set of retrieval models evaluated on the leaderboard.

Baselines plus the phase-3 retrievers: the content (text-embedding ANN) retriever in three
backends, and the two-tower network. Each entry is a zero-argument factory so the runner can
fit a fresh instance. Ranking models (LightGBM, DCNv2) run on top of pooled candidates and are
evaluated separately in the ranking stage, not here.
"""
from __future__ import annotations

from typing import Callable

from .base import Recommender
from .baselines import ALL_BASELINES
from .content import Content
from .two_tower import TwoTower


def retriever_factories() -> dict[str, Callable[[], Recommender]]:
    out: dict[str, Callable[[], Recommender]] = {cls.name: cls for cls in ALL_BASELINES}
    for backend in ("tfidf", "bge", "gemini"):
        out[f"content_{backend}"] = (lambda b=backend: Content(b))
    out["two_tower"] = TwoTower
    return out


# Order shown on the leaderboard.
RETRIEVER_ORDER = [cls.name for cls in ALL_BASELINES] + ["content_tfidf", "content_bge", "content_gemini", "two_tower"]
