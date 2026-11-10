"""Baseline recommenders (retrieval stage). Each scores candidate main dishes for a request.

All fit only on orders before the cutoff day, so nothing from the test period leaks in.
    Random            floor
    Popularity        global order counts
    GeoTimePopularity popularity in the request's city and daypart (the cold-start answer)
    BuyAgain          the user's own past dishes, recency- and frequency-weighted
    ItemItem          item-to-item co-occurrence within a user's baskets
    EASE              closed-form item-item collaborative filtering (shrunk)
    ALS               implicit matrix factorisation (numpy, no external library)
"""
from __future__ import annotations

import numpy as np

from ..data import Dataset, Request
from .base import BaseRecommender, FitContext


class Random(BaseRecommender):
    name = "random"

    def fit(self, ctx: FitContext) -> None:
        super().fit(ctx)
        self._rng = np.random.default_rng(0)

    def recommend(self, req: Request, cand: np.ndarray) -> np.ndarray:
        return self._rng.random(len(cand))


class Popularity(BaseRecommender):
    name = "popularity"

    def fit(self, ctx: FitContext) -> None:
        super().fit(ctx)
        rows = ctx.data.order_rows_before(ctx.cutoff_day)
        self.score = np.log1p(np.bincount(ctx.data.o_main[rows], minlength=ctx.data.n_items)).astype(np.float32)

    def recommend(self, req: Request, cand: np.ndarray) -> np.ndarray:
        return self.score[cand]


class GeoTimePopularity(BaseRecommender):
    """Popularity within the request's city and daypart: the recommended cold-start answer.

    Falls back to city-only, then global, when a (city, daypart) cell is thin.
    """
    name = "geo_time_pop"

    def fit(self, ctx: FitContext) -> None:
        super().fit(ctx)
        d = ctx.data
        rows = d.order_rows_before(ctx.cutoff_day)
        daypart = _daypart(d.o_hour[rows])
        city = d.o_city[rows]
        items = d.o_main[rows]
        self.cell = {}
        for c in (0, 1):
            for dp in range(5):
                m = (city == c) & (daypart == dp)
                self.cell[(c, dp)] = np.log1p(np.bincount(items[m], minlength=d.n_items)).astype(np.float32)
        self.city_pop = {c: np.log1p(np.bincount(items[city == c], minlength=d.n_items)).astype(np.float32)
                         for c in (0, 1)}
        self.glob = np.log1p(np.bincount(items, minlength=d.n_items)).astype(np.float32)

    def recommend(self, req: Request, cand: np.ndarray) -> np.ndarray:
        s = self.cell[(req.city, req.daypart)][cand]
        # blend in coarser levels so an empty cell still ranks sensibly
        return s + 0.3 * self.city_pop[req.city][cand] + 0.1 * self.glob[cand]


class BuyAgain(BaseRecommender):
    """The user's own dishes, weighted by frequency and recency (exponential decay)."""
    name = "buy_again"
    half_life_days = 30.0

    def recommend(self, req: Request, cand: np.ndarray) -> np.ndarray:
        hist = self.data.history(req.user, req.day)
        score = np.zeros(len(cand), np.float32)
        if hist.n:
            decay = 0.5 ** ((req.day - hist.days) / self.half_life_days)
            w = np.bincount(hist.items, weights=decay, minlength=self.data.n_items)
            score = w[cand].astype(np.float32)
        return score


class ItemItem(BaseRecommender):
    """Item-to-item cosine similarity over who ordered what.

    S = D^-1/2 (M^T M) D^-1/2 where M is the binary user x main-item matrix and D holds item
    frequencies. A user's score for a candidate is the summed similarity to the mains they
    have ordered: S[past].sum() restricted to the candidate columns."""
    name = "item_item"

    def fit(self, ctx: FitContext) -> None:
        super().fit(ctx)
        from scipy.sparse import csr_matrix
        d = ctx.data
        rows = d.order_rows_before(ctx.cutoff_day)
        col = d.main_index[d.o_main[rows]]
        u = d.o_user[rows]
        m = csr_matrix((np.ones(len(u), np.float32), (u, col)), shape=(d.n_users, len(d.mains)))
        m.data[:] = 1.0
        m.sum_duplicates()
        m.data[:] = 1.0                                  # binary "user ordered this main"
        co = (m.T @ m).tocsr().astype(np.float32)        # main x main co-occurrence
        freq = np.asarray(m.sum(axis=0)).ravel()
        inv = 1.0 / np.sqrt(freq + 1e-6)
        co = co.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()   # cosine normalise
        co.setdiag(0.0)
        co.eliminate_zeros()
        self.sim = co
        self.col_of = d.main_index

    def recommend(self, req: Request, cand: np.ndarray) -> np.ndarray:
        hist = self.data.history(req.user, req.day)
        if not hist.n:
            return np.zeros(len(cand), np.float32)
        past = np.unique(self.col_of[hist.items])
        agg = np.asarray(self.sim[past].sum(axis=0)).ravel()           # summed similarity per main
        return agg[self.col_of[cand]].astype(np.float32)


class _MatrixCF(BaseRecommender):
    """Shared setup for CF models over the user x main-item order matrix (before cutoff)."""

    def _matrix(self, ctx: FitContext):
        d = ctx.data
        rows = d.order_rows_before(ctx.cutoff_day)
        self._main_ids = d.mains
        col_of = d.main_index                       # global item id -> column (mains only)
        u = d.o_user[rows]
        c = col_of[d.o_main[rows]]
        data = np.ones(len(u), np.float32)
        from scipy.sparse import csr_matrix
        R = csr_matrix((data, (u, c)), shape=(d.n_users, len(d.mains)))
        R.data[:] = 1.0                              # implicit: ordered or not
        return R


class EASE(_MatrixCF):
    """EASE: closed-form item-item CF. B = -(G+λI)^-1 off-diagonal, with G = R^T R."""
    name = "ease"
    reg = 250.0

    def fit(self, ctx: FitContext) -> None:
        super().fit(ctx)
        R = self._matrix(ctx)
        G = (R.T @ R).toarray().astype(np.float64)
        diag = np.diag_indices(G.shape[0])
        G[diag] += self.reg
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[diag] = 0.0
        self.B = B.astype(np.float32)               # (mains, mains)
        self.col_of = ctx.data.main_index

    def recommend(self, req: Request, cand: np.ndarray) -> np.ndarray:
        hist = self.data.history(req.user, req.day)
        cols = self.col_of[cand]
        if not hist.n:
            return np.zeros(len(cand), np.float32)
        hist_cols = self.col_of[hist.items]
        profile = np.bincount(hist_cols, minlength=self.B.shape[0]).astype(np.float32)
        return (self.B[:, cols].T @ profile).astype(np.float32)


class ALS(_MatrixCF):
    """Implicit ALS matrix factorisation in numpy (no external library)."""
    name = "als"
    factors = 64
    reg = 10.0
    iterations = 12
    alpha = 20.0

    def fit(self, ctx: FitContext) -> None:
        super().fit(ctx)
        R = self._matrix(ctx)
        C = (R * self.alpha).astype(np.float32)      # confidence = alpha * count
        n_u, n_i = R.shape
        rng = np.random.default_rng(0)
        X = rng.normal(0, 0.01, (n_u, self.factors)).astype(np.float32)
        Y = rng.normal(0, 0.01, (n_i, self.factors)).astype(np.float32)
        Ccsr, Ccsc = C.tocsr(), C.tocsc()
        for _ in range(self.iterations):
            X = _als_step(Ccsr, Y, self.reg, self.factors)
            Y = _als_step(Ccsc.T.tocsr(), X, self.reg, self.factors)
        self.X, self.Y = X, Y
        self.col_of = ctx.data.main_index

    def recommend(self, req: Request, cand: np.ndarray) -> np.ndarray:
        return (self.Y[self.col_of[cand]] @ self.X[req.user]).astype(np.float32)


def _als_step(C, F, reg, k):
    """One ALS half-step: solve for each row's latent vector given the other factor F."""
    FtF = F.T @ F + reg * np.eye(k, dtype=np.float32)
    out = np.zeros((C.shape[0], k), np.float32)
    for r in range(C.shape[0]):
        s, e = C.indptr[r], C.indptr[r + 1]
        if s == e:
            continue
        idx = C.indices[s:e]
        conf = C.data[s:e]
        Fi = F[idx]
        A = FtF + (Fi.T * conf) @ Fi
        b = (Fi.T * (conf + 1.0)) @ np.ones(len(idx), np.float32)   # preference = 1
        out[r] = np.linalg.solve(A, b)
    return out


def _daypart(hour: np.ndarray) -> np.ndarray:
    dp = np.full(len(hour), 4, np.int64)
    dp[(hour >= 7) & (hour < 12)] = 0
    dp[(hour >= 12) & (hour < 17)] = 1
    dp[(hour >= 17) & (hour < 20)] = 2
    dp[(hour >= 20) & (hour < 23)] = 3
    return dp


ALL_BASELINES = [Random, Popularity, GeoTimePopularity, BuyAgain, ItemItem, EASE, ALS]
