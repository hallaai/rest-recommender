"""Live recommendation service: holds fitted recommenders in memory and answers per-user
requests with a top-5 per algorithm, plus a plain-language reason for each pick.

Built for the Recommend page. It reuses the phase-2 baselines exactly as the leaderboard
evaluates them (fitted at the same cutoff), so what you see live matches what Compare scored.

A request is (user or a map point, hour, weekday, weather). Candidates are the main dishes
available there and then (open + delivers), the same availability rule the simulator used.
Business rules shape the final five: at most 2 dishes per restaurant, and only in-range.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date
from functools import cached_property

import numpy as np

from .data import Dataset, Request, VAL_END_DAY, MIN_PER_DAY, START
from .eval.base import FitContext
from .eval.baselines import ALL_BASELINES, _daypart
from .geo import CITIES, DISTRICTS, city_of, haversine_km, nearest_district
from .synth.vocab import CUISINES, DAYPARTS

log = logging.getLogger(__name__)
MAX_PER_RESTAURANT = 2
TOP_K = 5


@dataclass(frozen=True)
class Rec:
    item_id: int
    score: float
    reason: str


class RecommendService:
    """Thread-safe: recommenders are fitted once at startup, then only read."""

    def __init__(self, data: Dataset) -> None:
        self.data = data
        self._recs = {}
        self._fit_lock = threading.Lock()
        self._fitted = False
        self.fit_seconds: dict[str, float] = {}

    def fit(self) -> None:
        """Fit every baseline at the validation cutoff (train + val), like the leaderboard."""
        with self._fit_lock:
            if self._fitted:
                return
            ctx = FitContext(data=self.data, cutoff_day=VAL_END_DAY)
            for cls in ALL_BASELINES:
                rec = cls()
                t0 = time.perf_counter()
                rec.fit(ctx)
                self.fit_seconds[rec.name] = time.perf_counter() - t0
                self._recs[rec.name] = rec
            self._fitted = True
            log.info("Fitted %d recommenders for serving", len(self._recs))

    @property
    def algorithms(self) -> list[str]:
        return [cls.name for cls in ALL_BASELINES]

    # -- building a request from UI inputs --------------------------------------
    def make_request(self, user: int, lat: float, lon: float, hour: int, dow: int,
                     rain: bool, hot: bool) -> Request:
        city_code = city_of(np.array([lat]), np.array([lon]))[0]
        city = 0 if city_code == "BCN" else 1
        home_city = int(self.data.user_home_city[user]) if 0 <= user < self.data.n_users else city
        # a serving request is "now": a day just after the data, so history = all known orders
        day = VAL_END_DAY
        return Request(session_id=-1, user=user, ts=day * MIN_PER_DAY + hour * 60, day=day, hour=hour,
                       dow=dow, daypart=_daypart(np.array([hour]))[0], lat=lat, lon=lon, city=city,
                       home_city=home_city, rain=rain, hot=hot,
                       district=int(nearest_district(np.array([lat]), np.array([lon]))[0]),
                       target=-1, target_rest=-1)

    def recommend(self, algo: str, req: Request, k: int = TOP_K) -> dict:
        if algo not in self._recs:
            raise KeyError(algo)
        cand = self.data.available_mains(req.lat, req.lon, req.hour)
        t0 = time.perf_counter()
        if len(cand) == 0:
            return {"algorithm": algo, "items": [], "latency_ms": 0.0, "n_candidates": 0}
        scores = self._recs[algo].recommend(req, cand)
        latency = (time.perf_counter() - t0) * 1000
        # Personalised sources have no signal for an unknown user (a map point, user -1): they
        # rank by a user profile that does not exist. Only popularity sources are meaningful.
        personalised = {"buy_again", "item_item", "ease", "als"}
        unknown_user = not (0 <= req.user < self.data.n_users)
        # No signal (every candidate scores 0, or an unknown user for a personalised model)
        # means the algorithm has nothing to say for this request — say so, don't guess.
        if algo != "random" and ((unknown_user and algo in personalised) or not np.any(scores > 0)):
            return {"algorithm": algo, "items": [], "latency_ms": round(latency, 2),
                    "n_candidates": int(len(cand)), "no_signal": True}
        picks = self._apply_rules(cand, scores, k)
        hist = set(self.data.history(req.user, req.day).items.tolist()) if 0 <= req.user < self.data.n_users else set()
        items = [self._describe(int(cand[i]), float(scores[i]), algo, req, hist) for i in picks]
        return {"algorithm": algo, "items": items, "latency_ms": round(latency, 2),
                "n_candidates": int(len(cand)), "no_signal": False}

    def recommend_all(self, req: Request, k: int = TOP_K) -> list[dict]:
        return [self.recommend(a, req, k) for a in self.algorithms]

    def _apply_rules(self, cand: np.ndarray, scores: np.ndarray, k: int) -> list[int]:
        """Indices into cand for the final list: highest score first, at most 2 per restaurant."""
        order = np.argsort(-scores, kind="stable")
        per_rest: dict[int, int] = {}
        out = []
        for i in order:
            if scores[i] <= 0 and len(out) >= k:
                break
            r = int(self.data.item_rest[cand[i]])
            if per_rest.get(r, 0) >= MAX_PER_RESTAURANT:
                continue
            per_rest[r] = per_rest.get(r, 0) + 1
            out.append(int(i))
            if len(out) >= k:
                break
        return out

    # -- reasons and tooltip detail ---------------------------------------------
    @cached_property
    def _rest_name(self) -> dict[int, str]:
        rr = self.data.rest_df.set_index("restaurant_id")
        return {int(i): str(n) for i, n in rr["name"].items()}

    @cached_property
    def _item_name(self) -> np.ndarray:
        return self.data.items_df["name"].to_numpy()

    def _describe(self, item: int, score: float, algo: str, req: Request, hist: set[int]) -> dict:
        from .synth.truth import delivery_fee
        d = self.data
        rest = int(d.item_rest[item])
        km = float(haversine_km(req.lat, req.lon, d.rest_lat[rest], d.rest_lon[rest]))
        cuisine = CUISINES[int(d.item_cuisine[item])].label
        n_ordered = int(np.sum(d.history(req.user, req.day).items == item)) if 0 <= req.user < d.n_users else 0
        avg_basket = self._avg_basket(req.user, req.day)
        price = float(d.item_price[item])
        details = {
            "Algorithm score": round(float(score), 4),
            "Cuisine": cuisine,
            "Distance": f"{km:.2f} km",
            "Delivery fee": f"€{float(delivery_fee(np.array([km]))[0]):.2f}",
            "Price": f"€{price:.2f}",
            "Restaurant rating": f"★ {float(d.rest_rating[rest]):.1f}",
            "Times you ordered this": n_ordered,
            "Reorder": "yes" if item in hist else "no (new to you)",
        }
        if avg_basket:
            details["Price vs your average"] = f"{price / avg_basket:.2f}x"
        return {
            "item_id": item, "name": str(self._item_name[item]),
            "restaurant_id": rest, "restaurant": self._rest_name.get(rest, "?"),
            "cuisine": cuisine, "price": price, "distance_km": round(km, 2), "score": round(float(score), 4),
            "is_reorder": item in hist,
            "reason": self._reason(item, algo, req, hist, km, cuisine),
            "details": details,
        }

    def _avg_basket(self, user: int, before_day: int) -> float:
        if not (0 <= user < self.data.n_users):
            return 0.0
        items = self.data.history(user, before_day).items
        return float(np.mean(self.data.item_price[items])) if len(items) else 0.0

    def _reason(self, item: int, algo: str, req: Request, hist: set[int], km: float, cuisine: str) -> str:
        d = self.data
        if algo == "buy_again":
            n = int(np.sum(d.history(req.user, req.day).items == item))
            return f"ordered {n}x before" if n else "ordered before"
        if algo in ("popularity", "geo_time_pop"):
            where = CITIES[req.city].name if algo == "geo_time_pop" else "overall"
            when = DAYPARTS[req.daypart]
            return f"popular {cuisine.lower()} in {where} at {when}" if algo == "geo_time_pop" else f"popular {cuisine.lower()}"
        if algo in ("als", "ease"):
            return f"users with similar taste order this ({cuisine.lower()})"
        if algo == "item_item":
            return f"goes with what you order ({cuisine.lower()})"
        return f"{cuisine}, {km:.1f} km away"
