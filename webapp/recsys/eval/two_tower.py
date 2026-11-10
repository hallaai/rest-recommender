"""Two-tower retrieval model (PyTorch, CPU).

A user tower and an item tower map users and dishes into one vector space; their dot product
scores a (user, dish) pair. Trained on (user, ordered main dish) pairs with an in-batch
sampled-softmax loss: for each positive pair, the other items in the batch are negatives.

Why it matters for cold start: the user tower reads profile features (home district, age band,
platform, channel, home city), not just the user id. A user with no orders still gets a
meaningful vector from those features — the gap the collaborative baselines and the content
retriever cannot fill. The item tower reads the dish's text embedding too, so new dishes are
placed sensibly.

CPU-friendly: small embedding sizes, a few epochs, deterministic seed. The learned item vectors
are written to vec_two_tower for the pgvector serving path.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ..data import AGE_BANDS, CHANNELS, PLATFORMS, Request
from ..embed import load
from ..geo import DISTRICTS
from ..synth.vocab import CUISINES
from .base import BaseRecommender, FitContext

log = logging.getLogger(__name__)


class TwoTower(BaseRecommender):
    name = "two_tower"

    def __init__(self, dim: int = 64, epochs: int = 12, batch: int = 2048, lr: float = 1e-2,
                 temperature: float = 0.07, id_dropout: float = 0.5, text_backend: str = "gemini",
                 seed: int = 0) -> None:
        self.dim, self.epochs, self.batch, self.lr = dim, epochs, batch, lr
        self.temperature, self.id_dropout = temperature, id_dropout
        self.text_backend, self.seed = text_backend, seed

    def fit(self, ctx: FitContext) -> None:
        super().fit(ctx)
        import torch
        torch.manual_seed(self.seed)
        d = ctx.data
        rows = d.order_rows_before(ctx.cutoff_day)
        # positives: (user, main item)
        self._pos_u = torch.as_tensor(d.o_user[rows], dtype=torch.long)
        self._pos_i = torch.as_tensor(d.main_index[d.o_main[rows]], dtype=torch.long)
        self._build_features(d)
        self.model = _make_net(torch, d.n_users, len(AGE_BANDS), len(DISTRICTS), len(PLATFORMS),
                               len(CHANNELS), len(d.mains), len(CUISINES), d.n_rest, self._text.shape[1], self.dim)
        self._train(torch)
        self.model.eval()
        with torch.no_grad():
            self._item_vecs = self.model.item_tower(self._item_feats(torch)).cpu().numpy()
        self.col_of = d.main_index

    def _build_features(self, d) -> None:
        import torch
        tl = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long)
        self._u_district = tl(d.user_district)
        self._u_age = tl(d.user_age)
        self._u_plat = tl(d.user_platform)
        self._u_chan = tl(d.user_channel)
        self._u_city = tl(d.user_home_city)
        mains = d.mains
        self._i_cuisine = tl(d.item_cuisine[mains])
        self._i_rest = tl(d.item_rest[mains])
        text = load(self.text_backend, d.dir).astype(np.float32)[mains]
        text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-9)
        self._text = torch.as_tensor(text)

    def _user_feats(self, torch, u):
        return (u, self._u_district[u], self._u_age[u], self._u_plat[u], self._u_chan[u], self._u_city[u])

    def _item_feats(self, torch, i=None):
        if i is None:
            i = torch.arange(len(self._i_cuisine))
        return (i, self._i_cuisine[i], self._i_rest[i], self._text[i])

    def _train(self, torch) -> None:
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = len(self._pos_u)
        g = torch.Generator().manual_seed(self.seed)
        t0 = time.perf_counter()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            total = 0.0
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                u, i = self._pos_u[idx], self._pos_i[idx]
                id_keep = (torch.rand(len(idx), generator=g) >= self.id_dropout).float()
                uv = self.model.user_tower(self._user_feats(torch, u), id_keep)
                iv = self.model.item_tower(self._item_feats(torch, i))
                # Cosine logits need a temperature; without it the unit-vector dot products are
                # too flat for cross-entropy to push positives above negatives.
                logits = (uv @ iv.T) / self.temperature             # (B, B): in-batch negatives
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(idx)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            log.info("  two-tower epoch %d/%d loss %.4f", epoch + 1, self.epochs, total / n)
        self.train_seconds = time.perf_counter() - t0

    def user_vector(self, req: Request, known: bool = True) -> np.ndarray:
        import torch
        with torch.no_grad():
            u = torch.as_tensor([req.user], dtype=torch.long)
            id_keep = torch.ones(1) if known else torch.zeros(1)   # unknown user: profile features only
            return self.model.user_tower(self._user_feats(torch, u), id_keep).cpu().numpy()[0]

    def recommend(self, req: Request, cand: np.ndarray) -> np.ndarray:
        uv = self.user_vector(req)
        return (self._item_vecs[self.col_of[cand]] @ uv).astype(np.float32)

    def item_vectors(self) -> np.ndarray:
        """Learned item vectors in global item-id order (mains filled, non-mains zero)."""
        out = np.zeros((self.data.n_items, self.dim), np.float32)
        out[self.data.mains] = self._item_vecs
        return out


def _make_net(torch, n_users, n_age, n_district, n_plat, n_chan, n_items, n_cuisine, n_rest, text_dim, dim):
    nn = torch.nn
    F = torch.nn.functional
    e = 16

    def mlp(in_dim):
        return nn.Sequential(nn.Linear(in_dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.u_id = nn.Embedding(n_users, e)
            self.u_district = nn.Embedding(n_district, e)
            self.u_age = nn.Embedding(n_age, e)
            self.u_plat = nn.Embedding(n_plat, 4)
            self.u_chan = nn.Embedding(n_chan, 4)
            self.u_city = nn.Embedding(2, 4)
            self.u_mlp = mlp(e * 3 + 12)
            self.i_id = nn.Embedding(n_items, e)
            self.i_cuisine = nn.Embedding(n_cuisine, e)
            self.i_rest = nn.Embedding(n_rest, e)
            self.i_mlp = mlp(e * 3 + text_dim)

        def user_tower(self, f, id_keep=None):
            u, district, age, plat, chan, city = f
            uid = self.u_id(u)
            if id_keep is not None:                     # zero the id part for some rows (cold-start training)
                uid = uid * id_keep.unsqueeze(-1)
            x = torch.cat([uid, self.u_district(district), self.u_age(age),
                           self.u_plat(plat), self.u_chan(chan), self.u_city(city)], dim=-1)
            return F.normalize(self.u_mlp(x), dim=-1)

        def item_tower(self, f):
            i, cuisine, rest, text = f
            x = torch.cat([self.i_id(i), self.i_cuisine(cuisine), self.i_rest(rest), text], dim=-1)
            return F.normalize(self.i_mlp(x), dim=-1)

    return Net()
