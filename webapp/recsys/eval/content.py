"""Content retriever: recommend dishes whose text embedding is close to the user's taste.

The user's taste vector is the time-weighted average of the embeddings of the dishes they
ordered (recent orders weigh more, 30-day half-life), L2-normalised. A candidate's score is
the cosine similarity between the user vector and the dish vector.

This is the "ANN over text embeddings" source from the design. It is the one retriever that
works on dish *meaning*, so it can surface dishes the user never ordered (and carry taste to
the other city). It has no signal for a user with no history (cold start) — the two-tower
model adds profile features to cover that.

Scoring is done in numpy here (fast for evaluation over fixed candidate sets). The same
vectors live in Postgres (vec_<backend>) for the pgvector serving path and the ANN benchmark.
"""
from __future__ import annotations

import numpy as np

from ..data import Request
from ..embed import load
from .base import BaseRecommender, FitContext


class Content(BaseRecommender):
    half_life_days = 30.0

    def __init__(self, backend: str = "gemini") -> None:
        self.backend = backend
        self.name = f"content_{backend}"

    def fit(self, ctx: FitContext) -> None:
        super().fit(ctx)
        vecs = load(self.backend, ctx.data.dir).astype(np.float32)
        # normalise so dot product is cosine similarity
        self.item_vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)

    def user_vector(self, req: Request) -> np.ndarray | None:
        """Time-weighted average of the user's ordered-dish vectors, or None if no history."""
        hist = self.data.history(req.user, req.day)
        if not hist.n:
            return None
        w = 0.5 ** ((req.day - hist.days) / self.half_life_days)
        v = (self.item_vecs[hist.items] * w[:, None]).sum(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None

    def recommend(self, req: Request, cand: np.ndarray) -> np.ndarray:
        uv = self.user_vector(req)
        if uv is None:
            return np.zeros(len(cand), np.float32)
        return (self.item_vecs[cand] @ uv).astype(np.float32)
