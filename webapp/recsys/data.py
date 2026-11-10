"""Offline dataset for training and evaluation: numpy arrays loaded from data/*.parquet.

Conventions
- Time: integer minutes since 2026-05-01 00:00 local time; day = minute // 1440.
- Point-in-time history: a request on day d sees only orders from days < d (a nightly
  feature snapshot). Models fitted with cutoff day c use only orders from days < c.
- Item ids are global (0..I-1). Recommenders rank main dishes only; `main_index` maps an
  item id to its position among mains (or -1).
- A request is an ordered session; its target is the main dish the user chose.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .geo import DISTRICTS, haversine_km, nearest_district
from .synth.restaurants import open_matrix
from .synth.vocab import CATEGORIES, CUISINE_INDEX
from .synth.world import SPLIT, WorldConfig

START = pd.Timestamp(WorldConfig().start)
MIN_PER_DAY = 1440
AGE_BANDS = ("18-24", "25-34", "35-44", "45-54", "55+")
PLATFORMS = ("android", "ios")
CHANNELS = ("organic", "referral", "paid_social", "promo_code")


def day_index(d: date) -> int:
    return (pd.Timestamp(d) - START).days


TRAIN_END_DAY = day_index(SPLIT.train_end) + 1     # first day after train
VAL_END_DAY = day_index(SPLIT.val_end) + 1         # first day after validation
TEST_END_DAY = day_index(SPLIT.test_end) + 1


@dataclass(frozen=True)
class History:
    """A user's main-dish orders before some day, oldest first."""
    items: np.ndarray      # main item ids
    rests: np.ndarray
    days: np.ndarray
    cuisines: np.ndarray
    prices: np.ndarray

    @property
    def n(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class Request:
    session_id: int
    user: int
    ts: int                # minutes since START
    day: int
    hour: int
    dow: int               # 0 = Monday
    daypart: int
    lat: float
    lon: float
    city: int              # 0 BCN, 1 MAD (where the user is now)
    home_city: int
    rain: bool
    hot: bool
    district: int          # nearest district of the request location
    target: int            # main item ordered (-1 when unknown)
    target_rest: int


class Dataset:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.dir = data_dir
        items = pd.read_parquet(data_dir / "items.parquet")
        rest = pd.read_parquet(data_dir / "restaurants.parquet")
        users = pd.read_parquet(data_dir / "users.parquet")
        self.items_df, self.rest_df, self.users_df = items, rest, users
        self.n_items, self.n_rest, self.n_users = len(items), len(rest), len(users)
        # items
        self.item_rest = items["restaurant_id"].to_numpy(np.int64)
        self.item_cat = items["category"].map({c: i for i, c in enumerate(CATEGORIES)}).to_numpy(np.int64)
        self.item_price = items["price"].to_numpy(np.float32)
        self.item_veg = items["vegetarian"].to_numpy(bool)
        self.item_spicy = items["spicy"].to_numpy(bool)
        self.item_sku = items["sku"].to_numpy()
        # restaurants
        self.rest_lat = rest["lat"].to_numpy()
        self.rest_lon = rest["lon"].to_numpy()
        self.rest_radius = rest["delivery_radius_km"].to_numpy()
        self.rest_city = (rest["city"] == "MAD").to_numpy().astype(np.int64)
        self.rest_cuisine = rest["cuisine"].map(CUISINE_INDEX).to_numpy(np.int64)
        self.rest_rating = rest["rating"].to_numpy(np.float32)
        self.rest_tier = rest["price_tier"].to_numpy(np.int64)
        self.rest_prep = rest["prep_min"].to_numpy(np.float32)
        self.rest_chain = rest["chain"].fillna("").to_numpy()
        self.rest_open = open_matrix([[tuple(w) for w in json.loads(h)] for h in rest["hours"]])
        self.item_cuisine = self.rest_cuisine[self.item_rest]
        # mains
        self.mains = np.flatnonzero(self.item_cat == 0)
        self.main_index = np.full(self.n_items, -1, np.int64)
        self.main_index[self.mains] = np.arange(len(self.mains))
        order = np.argsort(self.item_rest[self.mains], kind="stable")
        sorted_mains = self.mains[order]
        bounds = np.searchsorted(self.item_rest[sorted_mains], np.arange(self.n_rest + 1))
        self.rest_mains = [sorted_mains[bounds[r]:bounds[r + 1]] for r in range(self.n_rest)]
        # users
        self.user_home_city = (users["home_city"] == "MAD").to_numpy().astype(np.int64)
        self.user_home = users[["home_lat", "home_lon"]].to_numpy()
        self.user_district = users["district"].map({d.code: i for i, d in enumerate(DISTRICTS)}).to_numpy(np.int64)
        self.user_age = users["age_band"].map({a: i for i, a in enumerate(AGE_BANDS)}).to_numpy(np.int64)
        self.user_platform = users["platform"].map({p: i for i, p in enumerate(PLATFORMS)}).to_numpy(np.int64)
        self.user_channel = users["channel"].map({c: i for i, c in enumerate(CHANNELS)}).to_numpy(np.int64)
        self.user_signup_day = (pd.to_datetime(users["signup_date"]) - START).dt.days.to_numpy(np.int64)
        self._load_orders()
        self._load_sessions()

    def _load_orders(self) -> None:
        o = pd.read_parquet(self.dir / "orders.parquet")
        oi = pd.read_parquet(self.dir / "order_items.parquet")
        main = oi[oi["is_main"]].set_index("order_id")["item_id"]
        o = o.assign(main=main.loc[o["order_id"]].to_numpy(),
                     ts_min=((pd.to_datetime(o["ts"]) - START).dt.total_seconds() // 60).astype(np.int64))
        o = o.sort_values(["user_id", "ts_min", "order_id"]).reset_index(drop=True)
        self.o_user = o["user_id"].to_numpy(np.int64)
        self.o_ts = o["ts_min"].to_numpy(np.int64)
        self.o_day = self.o_ts // MIN_PER_DAY
        self.o_main = o["main"].to_numpy(np.int64)
        self.o_rest = o["restaurant_id"].to_numpy(np.int64)
        self.o_session = o["session_id"].to_numpy(np.int64)
        self.o_city = (o["city"] == "MAD").to_numpy().astype(np.int64)
        self.o_lat, self.o_lon = o["lat"].to_numpy(), o["lon"].to_numpy()
        self.o_district = nearest_district(self.o_lat, self.o_lon)
        self.o_hour = ((self.o_ts % MIN_PER_DAY) // 60).astype(np.int64)
        self.o_subtotal = o["subtotal"].to_numpy(np.float32)
        self.user_ptr = np.searchsorted(self.o_user, np.arange(self.n_users + 1))
        # all lines (add-ons too), used by the ground-truth habit counts
        oid_to_row = pd.Series(np.arange(len(o)), index=o["order_id"])
        self.l_row = oid_to_row.loc[oi["order_id"]].to_numpy(np.int64)
        self.l_item = oi["item_id"].to_numpy(np.int64)

    def _load_sessions(self) -> None:
        s = pd.read_parquet(self.dir / "sessions.parquet").sort_values("session_id").reset_index(drop=True)
        self.s_user = s["user_id"].to_numpy(np.int64)
        self.s_ts = ((pd.to_datetime(s["ts"]) - START).dt.total_seconds() // 60).astype(np.int64).to_numpy()
        self.s_lat, self.s_lon = s["lat"].to_numpy(), s["lon"].to_numpy()
        self.s_city = (s["city"] == "MAD").to_numpy().astype(np.int64)
        self.s_daypart = s["daypart"].to_numpy(np.int64)
        self.s_rain = s["rain"].to_numpy(bool)
        self.s_hot = s["hot"].to_numpy(bool)
        self.s_ordered = s["ordered"].to_numpy(bool)
        assert (s["session_id"].to_numpy() == np.arange(len(s))).all(), "session ids must be 0..n-1"

    # -- history -----------------------------------------------------------------
    def history(self, user: int, before_day: int) -> History:
        lo, hi = self.user_ptr[user], self.user_ptr[user + 1]
        cut = lo + int(np.searchsorted(self.o_day[lo:hi], before_day, side="left"))
        sl = slice(lo, cut)
        items = self.o_main[sl]
        return History(items, self.o_rest[sl], self.o_day[sl], self.item_cuisine[items], self.item_price[items])

    def order_rows_before(self, cutoff_day: int) -> np.ndarray:
        """Row indices of orders placed before `cutoff_day` (for fitting)."""
        return np.flatnonzero(self.o_day < cutoff_day)

    # -- requests ----------------------------------------------------------------
    def requests(self, first_day: int, end_day: int, sample: int | None = None, seed: int = 0) -> list[Request]:
        rows = np.flatnonzero((self.o_day >= first_day) & (self.o_day < end_day))
        if sample and len(rows) > sample:
            rows = np.sort(np.random.default_rng(seed).choice(rows, sample, replace=False))
        out = []
        for r in rows:
            sid = int(self.o_session[r])
            ts = int(self.o_ts[r])
            day = ts // MIN_PER_DAY
            out.append(Request(
                session_id=sid, user=int(self.o_user[r]), ts=ts, day=day, hour=int(self.o_hour[r]),
                dow=int((START.dayofweek + day) % 7), daypart=int(self.s_daypart[sid]),
                lat=float(self.o_lat[r]), lon=float(self.o_lon[r]), city=int(self.o_city[r]),
                home_city=int(self.user_home_city[self.o_user[r]]), rain=bool(self.s_rain[sid]),
                hot=bool(self.s_hot[sid]), district=int(self.o_district[r]),
                target=int(self.o_main[r]), target_rest=int(self.o_rest[r])))
        return out

    def available_mains(self, lat: float, lon: float, hour: int) -> np.ndarray:
        """Main dishes from restaurants open at `hour` that deliver to (lat, lon)."""
        km = haversine_km(lat, lon, self.rest_lat, self.rest_lon)
        rs = np.flatnonzero((km <= self.rest_radius) & self.rest_open[hour % 24])
        if not len(rs):
            return np.zeros(0, np.int64)
        return np.concatenate([self.rest_mains[r] for r in rs])

    def rest_km(self, lat: float, lon: float) -> np.ndarray:
        return haversine_km(lat, lon, self.rest_lat, self.rest_lon)

    @cached_property
    def cold_start_profile(self) -> np.ndarray:
        """Visible sign-up profile per user as small ints: (district, age, platform, channel, home city)."""
        return np.stack([self.user_district, self.user_age, self.user_platform, self.user_channel,
                         self.user_home_city], axis=1)
