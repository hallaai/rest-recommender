"""Ground-truth oracle: the true probability a user orders each candidate dish.

Reconstructs the utility the simulator used (recsys/synth/truth.py) for one request over its
candidate main dishes, then the softmax choice probabilities among them. This lets the
evaluator score a recommended list by the demand it would actually capture, using every
dish's probability rather than the single dish that happened to be observed, so the metric is
far less noisy than hit-rate.

This module reads the hidden truth files. It is used ONLY by the evaluator; no recommender
imports it (a test enforces that).
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

from ..config import DATA_DIR
from ..data import Dataset, MIN_PER_DAY, Request
from ..geo import haversine_km
from ..synth import truth as T
from ..synth.users import UserTruth


class TruthOracle:
    def __init__(self, data: Dataset, data_dir: Path = DATA_DIR) -> None:
        self.data = data
        u = dict(np.load(data_dir / "truth" / "users.npz"))
        self.ut = UserTruth(log_aff=u["log_aff"], z=u["z"], price_sens=u["price_sens"],
                            dist_sens=u["dist_sens"], vegetarian=u["vegetarian"], spicy_tol=u["spicy_tol"],
                            habit=u["habit"], daypart_w=u["daypart_w"], weekly_rate=u["weekly_rate"],
                            conv_base=u["conv_base"], ref_price=u["ref_price"])
        it = dict(np.load(data_dir / "truth" / "items.npz"))
        self.item = T.ItemTruth(
            restaurant=data.item_rest, cuisine=data.item_cuisine, category=data.item_cat,
            log_price=np.log(data.item_price), vegetarian=data.item_veg, spicy=data.item_spicy,
            quality=it["quality"], z=it["z"])

    def choice_probs(self, req: Request, cand: np.ndarray) -> np.ndarray:
        """P(user orders dish | they order something this session) for each candidate main.

        Uses the same utility terms as the simulator, with habit counts taken from the user's
        orders before the request day (point-in-time), matching how the request was generated.
        """
        u = req.user
        it = self.item
        # static (taste, price, diet) + context (daypart, weather) + distance/fee
        util = T.static_utility(self.ut, u, it)[cand].astype(np.float64)
        util += T.context_utility(it, req.daypart, req.rain, req.hot)[cand]
        km = haversine_km(req.lat, req.lon, self.data.rest_lat[it.restaurant[cand]],
                          self.data.rest_lon[it.restaurant[cand]])
        util += T.distance_utility(self.ut, u, km)
        # habit: counts of this dish and this restaurant before the request day
        hist = self.data.history(u, req.day)
        n_item = _counts(hist.items, cand, self.data.n_items)
        n_rest = _counts(hist.rests, it.restaurant[cand], self.data.n_rest)
        util += T.habit_utility(self.ut, u, n_item, n_rest)
        return T.softmax(util)


def _counts(history_ids: np.ndarray, target_ids: np.ndarray, n: int) -> np.ndarray:
    """How many times each target id appears in history_ids (vectorised)."""
    if len(history_ids) == 0:
        return np.zeros(len(target_ids), np.float64)
    freq = np.bincount(history_ids, minlength=n)
    return freq[target_ids].astype(np.float64)
