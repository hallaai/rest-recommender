"""Feature extraction for the ranking stage: ~40 numbers per (request, candidate dish) pair.

This is where retrieval scores and hard features (distance, time, price, history) sit side by
side as columns, so the ranker learns to weigh them against each other. Features are computed
strictly from information available at the request: the user's orders before the request day,
the dish and restaurant attributes, and the context (hour, weekday, weather, location).

A row is one candidate dish for one request. The label is 1 for the dish the user ordered, 0
otherwise (for pointwise/pairwise training). Groups are per request (for listwise ranking).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import Dataset, Request
from ..geo import haversine_km
from ..synth.truth import delivery_fee

FEATURE_NAMES = [
    "retr_buy_again", "retr_als", "retr_item_item", "retr_content", "retr_pop", "retr_geo_pop",
    "distance_km", "delivery_fee", "prep_min", "rest_rating", "price", "price_vs_user_avg",
    "is_reorder", "times_ordered_item", "times_ordered_rest", "days_since_item", "days_since_cuisine",
    "cuisine_count_7d", "cuisine_count_30d", "cuisine_count_90d", "cuisine_share_user",
    "is_veg", "is_spicy", "user_is_veg", "price_tier", "zone_pop_pctile",
    "hour_sin", "hour_cos", "is_weekend", "daypart", "rain", "hot",
    "user_n_orders", "user_avg_price", "same_city_as_home",
]


@dataclass
class FeatureBundle:
    X: np.ndarray            # (n_rows, n_features)
    y: np.ndarray            # (n_rows,) 1 if ordered
    groups: np.ndarray       # (n_requests,) rows per request, in order
    request_idx: np.ndarray  # (n_rows,) which request each row belongs to
    cand: np.ndarray         # (n_rows,) candidate item id


class FeatureBuilder:
    """Builds features for a request given candidate scores from the retrieval sources."""

    def __init__(self, data: Dataset) -> None:
        self.d = data
        pop_rows = data.o_main  # filled by caller via set_popularity
        self._zone_pop = None

    def set_popularity(self, cutoff_day: int) -> None:
        d = self.d
        rows = d.order_rows_before(cutoff_day)
        pop = np.bincount(d.o_main[rows], minlength=d.n_items).astype(np.float64)
        self._pop_pctile = pop.argsort().argsort() / max(d.n_items - 1, 1)

    def row(self, req: Request, cand: np.ndarray, retr: dict[str, np.ndarray]) -> np.ndarray:
        """One feature matrix (len(cand), n_features) for a request."""
        d = self.d
        hist = d.history(req.user, req.day)
        rest = d.item_rest[cand]
        km = haversine_km(req.lat, req.lon, d.rest_lat[rest], d.rest_lon[rest])
        price = d.item_price[cand]
        user_avg = float(np.mean(hist.prices)) if hist.n else float(np.mean(price))
        # history counts per item / restaurant / cuisine
        item_freq = np.bincount(hist.items, minlength=d.n_items)
        rest_freq = np.bincount(hist.rests, minlength=d.n_rest)
        cuis = d.item_cuisine[cand]
        cuis_hist = hist.cuisines
        def cuis_count(days):
            if not hist.n:
                return np.zeros(len(cand))
            recent = hist.cuisines[hist.days >= req.day - days]
            freq = np.bincount(recent, minlength=len(np.unique(d.item_cuisine))) if len(recent) else np.zeros(1)
            full = np.bincount(recent, minlength=int(d.item_cuisine.max()) + 1) if len(recent) else np.zeros(int(d.item_cuisine.max()) + 1)
            return full[cuis]
        days_since_item = np.full(len(cand), 999.0)
        days_since_cuis = np.full(len(cand), 999.0)
        if hist.n:
            last_item_day = {}
            for it, dy in zip(hist.items, hist.days):
                last_item_day[it] = dy
            days_since_item = np.array([req.day - last_item_day.get(int(c), req.day - 999) for c in cand], float)
            last_cuis_day = {}
            for cu, dy in zip(hist.cuisines, hist.days):
                last_cuis_day[cu] = dy
            days_since_cuis = np.array([req.day - last_cuis_day.get(int(c), req.day - 999) for c in cuis], float)
        cuis_share = (np.array([np.mean(cuis_hist == c) for c in cuis]) if hist.n else np.zeros(len(cand)))

        n = len(cand)
        f = np.zeros((n, len(FEATURE_NAMES)), np.float32)
        col = {name: k for k, name in enumerate(FEATURE_NAMES)}
        f[:, col["retr_buy_again"]] = retr.get("buy_again", np.zeros(n))
        f[:, col["retr_als"]] = retr.get("als", np.zeros(n))
        f[:, col["retr_item_item"]] = retr.get("item_item", np.zeros(n))
        f[:, col["retr_content"]] = retr.get("content", np.zeros(n))
        f[:, col["retr_pop"]] = retr.get("popularity", np.zeros(n))
        f[:, col["retr_geo_pop"]] = retr.get("geo_time_pop", np.zeros(n))
        f[:, col["distance_km"]] = km
        f[:, col["delivery_fee"]] = delivery_fee(km)
        f[:, col["prep_min"]] = d.rest_prep[rest]
        f[:, col["rest_rating"]] = d.rest_rating[rest]
        f[:, col["price"]] = price
        f[:, col["price_vs_user_avg"]] = price / max(user_avg, 1e-3)
        f[:, col["is_reorder"]] = (item_freq[cand] > 0).astype(np.float32)
        f[:, col["times_ordered_item"]] = item_freq[cand]
        f[:, col["times_ordered_rest"]] = rest_freq[rest]
        f[:, col["days_since_item"]] = np.clip(days_since_item, 0, 999)
        f[:, col["days_since_cuisine"]] = np.clip(days_since_cuis, 0, 999)
        f[:, col["cuisine_count_7d"]] = cuis_count(7)
        f[:, col["cuisine_count_30d"]] = cuis_count(30)
        f[:, col["cuisine_count_90d"]] = cuis_count(90)
        f[:, col["cuisine_share_user"]] = cuis_share
        f[:, col["is_veg"]] = d.item_veg[cand]
        f[:, col["is_spicy"]] = d.item_spicy[cand]
        f[:, col["user_is_veg"]] = 0.0        # visible proxy unknown; left 0 (diet not in profile)
        f[:, col["price_tier"]] = d.rest_tier[rest]
        f[:, col["zone_pop_pctile"]] = self._pop_pctile[cand]
        f[:, col["hour_sin"]] = np.sin(2 * np.pi * req.hour / 24)
        f[:, col["hour_cos"]] = np.cos(2 * np.pi * req.hour / 24)
        f[:, col["is_weekend"]] = 1.0 if req.dow >= 5 else 0.0
        f[:, col["daypart"]] = req.daypart
        f[:, col["rain"]] = float(req.rain)
        f[:, col["hot"]] = float(req.hot)
        f[:, col["user_n_orders"]] = hist.n
        f[:, col["user_avg_price"]] = user_avg
        f[:, col["same_city_as_home"]] = 1.0 if req.city == req.home_city else 0.0
        return f
