"""The hidden ground-truth choice model.

The simulator draws behaviour from it; the evaluator uses it to compute, for any list of
recommended items, the true probability that the user orders one of them. No
recommendation model may import this module (a test enforces it).

Utility of main dish i for user u in context c:
    u = log_aff[u, cuisine_i] + z_u . z_i + quality_i
        - price_sens_u * log(price_i / ref_price_u)
        - dist_sens_u * km(u, restaurant_i) - 0.12 * fee
        + fit[cuisine_i, daypart] + rain/heat bonus
        - 4 * (vegetarian user and non-vegetarian dish) - 0.9 * (spicy dish and low tolerance)
        + habit_u * (0.8 * log1p(times ordered dish) + 0.6 * log1p(times ordered restaurant))
The user picks a main with softmax(u); whether they order at all depends on logsumexp(u).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .users import UserTruth
from .vocab import CUISINES

FIT = np.array([c.fit for c in CUISINES], dtype=np.float32)        # (n_cuisines, 5)
RAIN = np.array([c.rain for c in CUISINES], dtype=np.float32)
HEAT = np.array([c.heat for c in CUISINES], dtype=np.float32)
LSE_REF = 7.4           # logsumexp level at which a session converts at the user's base rate (calibrated: ~60% conversion)
EXPOSURE_BOOST = 1.0    # extra utility for dishes the feed showed (the logging policy's effect)


@dataclass(frozen=True)
class ItemTruth:
    """Hidden per-item parameters (row i = item i) plus the visible fields the utility uses."""
    restaurant: np.ndarray   # (I,) int
    cuisine: np.ndarray      # (I,) int
    category: np.ndarray     # (I,) int, 0 = main
    log_price: np.ndarray    # (I,)
    vegetarian: np.ndarray   # (I,) bool
    spicy: np.ndarray        # (I,) bool
    quality: np.ndarray      # (I,) dish quality + restaurant quality
    z: np.ndarray            # (I, LATENT_DIM)


def delivery_fee(km: np.ndarray) -> np.ndarray:
    """Visible delivery fee in EUR: 0.99 base + 0.45 per km, capped at 4.99."""
    return np.minimum(0.99 + 0.45 * km, 4.99)


def static_utility(ut: UserTruth, u: int, it: ItemTruth) -> np.ndarray:
    """Context-free part of the utility of every item for user u."""
    util = (ut.log_aff[u][it.cuisine] + it.z @ ut.z[u] + it.quality
            - ut.price_sens[u] * (it.log_price - np.log(ut.ref_price[u])))
    if ut.vegetarian[u]:
        util = util - 4.0 * (~it.vegetarian)
    if ut.spicy_tol[u] < -0.5:
        util = util - 0.9 * it.spicy
    return util.astype(np.float32)


def context_utility(it: ItemTruth, daypart: int, rain: bool, hot: bool) -> np.ndarray:
    util = FIT[it.cuisine, daypart].copy()
    if rain:
        util += RAIN[it.cuisine]
    if hot:
        util += HEAT[it.cuisine]
    return util


def distance_utility(ut: UserTruth, u: int, km: np.ndarray) -> np.ndarray:
    """Per-restaurant utility of distance and fee."""
    return (-ut.dist_sens[u] * km - 0.12 * delivery_fee(km)).astype(np.float32)


def habit_utility(ut: UserTruth, u: int, n_item: np.ndarray, n_rest: np.ndarray) -> np.ndarray:
    return ut.habit[u] * (0.8 * np.log1p(n_item) + 0.6 * np.log1p(n_rest))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def logsumexp(x: np.ndarray) -> float:
    m = float(x.max())
    return m + float(np.log(np.exp(x - m).sum()))


def conversion_prob(ut: UserTruth, u: int, lse: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(ut.conv_base[u] + 0.7 * (lse - LSE_REF)))))
