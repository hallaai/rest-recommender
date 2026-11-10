"""DCNv2 ranker (Deep & Cross Network v2, PyTorch, CPU): the neural challenger to LightGBM.

DCNv2 scores a (request, dish) feature row with two parallel paths whose outputs are concatenated:
  - a cross network that learns explicit, bounded-degree feature interactions
    (x_{l+1} = x_0 * (W x_l + b) + x_l), so products like distance x is-dinner x cuisine form
    without being hand-crafted;
  - a deep MLP for everything else.

Trained with the same binary labels as the pointwise ranker (ordered = 1), on the same features,
so the comparison is like-for-like: does a neural model's learned interactions beat gradient-
boosted trees on these tabular ranking features? Often they do not, and showing that honestly
(with the cost) is the point of including it.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .features import FEATURE_NAMES

log = logging.getLogger(__name__)


class DCNv2:
    name = "ranker_dcnv2"

    def __init__(self, cross_layers: int = 3, deep: tuple[int, ...] = (128, 64), epochs: int = 8,
                 batch: int = 8192, lr: float = 3e-3, seed: int = 0) -> None:
        self.cross_layers, self.deep, self.epochs = cross_layers, deep, epochs
        self.batch, self.lr, self.seed = batch, lr, seed

    def fit(self, train) -> None:
        import torch
        torch.manual_seed(self.seed)
        X = torch.as_tensor(np.ascontiguousarray(train.X), dtype=torch.float32)
        y = torch.as_tensor(np.ascontiguousarray(train.y), dtype=torch.float32)
        self._mean = X.mean(0, keepdim=True)
        self._std = X.std(0, keepdim=True).clamp_min(1e-6)
        Xn = (X - self._mean) / self._std
        self.model = _DCN(X.shape[1], self.cross_layers, self.deep)
        pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        n = len(y)
        t0 = time.perf_counter()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            total = 0.0
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                out = self.model(Xn[idx]).squeeze(-1)
                loss = loss_fn(out, y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.detach()) * len(idx)
            log.info("  dcnv2 epoch %d/%d loss %.4f", epoch + 1, self.epochs, total / n)
        self.train_seconds = time.perf_counter() - t0
        self.model.eval()

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            Xn = (torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32) - self._mean) / self._std
            return torch.sigmoid(self.model(Xn).squeeze(-1)).cpu().numpy()


def _DCN(in_dim, cross_layers, deep):
    import torch
    nn = torch.nn

    class CrossNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(cross_layers)])

        def forward(self, x0):
            x = x0
            for lin in self.w:
                x = x0 * lin(x) + x          # DCNv2 cross: x0 ⊙ (W x + b) + x
            return x

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross = CrossNet()
            layers, d = [], in_dim
            for h in deep:
                layers += [nn.Linear(d, h), nn.ReLU()]
                d = h
            self.deep = nn.Sequential(*layers)
            self.head = nn.Linear(in_dim + deep[-1], 1)

        def forward(self, x):
            return self.head(torch.cat([self.cross(x), self.deep(x)], dim=-1))

    return Net()
