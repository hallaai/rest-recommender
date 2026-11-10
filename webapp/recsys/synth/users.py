"""Users: a visible profile (what the app knows) and a hidden taste (the ground truth).

Visible: home city/district/location, work location, signup date, platform, channel, age band.
Hidden (data/truth/, never loaded by models): cuisine affinities, latent dish taste, price
and distance sensitivity, diet, spice tolerance, habit strength, meal-time mix, activity,
trips to the other city. Some hidden traits correlate with visible ones (age -> cuisine,
district income and platform -> price sensitivity) so cold-start features carry signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..geo import CITIES, DISTRICTS, sample_points
from .vocab import CUISINES
from .world import SEGMENTS, SPLIT, WorldConfig

LATENT_DIM = 8
AGE_BANDS = ("18-24", "25-34", "35-44", "45-54", "55+")
YOUNG_CUISINES = ("burgers", "kebab", "chicken", "korean", "poke", "sushi")
OLDER_CUISINES = ("tapas", "italian", "grill", "chinese", "peruvian")


@dataclass(frozen=True)
class UserTruth:
    """Hidden per-user parameters as column arrays (row i = user i)."""
    log_aff: np.ndarray        # (U, n_cuisines)
    z: np.ndarray              # (U, LATENT_DIM)
    price_sens: np.ndarray     # (U,)
    dist_sens: np.ndarray      # (U,) utility per km
    vegetarian: np.ndarray     # (U,) bool
    spicy_tol: np.ndarray      # (U,)
    habit: np.ndarray          # (U,) strength of reordering
    daypart_w: np.ndarray      # (U, 5)
    weekly_rate: np.ndarray    # (U,) sessions per week when active
    conv_base: np.ndarray      # (U,) logit of ordering in a session
    ref_price: np.ndarray      # (U,) EUR the user considers normal for a main


def _segments(rng: np.random.Generator, n: int) -> np.ndarray:
    p = np.array([SEGMENTS.regular, SEGMENTS.light, SEGMENTS.growth, SEGMENTS.new_in_test])
    return rng.choice(np.array(SEGMENTS.names), size=n, p=p / p.sum())


def _signup(rng: np.random.Generator, seg: str, cfg: WorldConfig) -> date:
    if seg in ("regular", "light"):
        return cfg.start - timedelta(days=int(rng.integers(30, 900)))
    if seg == "growth":
        return cfg.start + timedelta(days=int(rng.integers(0, (SPLIT.val_end - cfg.start).days)))
    return SPLIT.val_end + timedelta(days=int(rng.integers(1, 29)))


def generate_users(cfg: WorldConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, UserTruth]:
    n = cfg.users
    city = np.where(rng.random(n) < cfg.user_city_share_bcn, "BCN", "MAD")
    seg = _segments(rng, n)
    age = rng.choice(len(AGE_BANDS), size=n, p=[0.17, 0.33, 0.25, 0.15, 0.10])
    rows = []
    income = np.empty(n)
    for i in range(n):
        idx = [k for k, d in enumerate(DISTRICTS) if d.city == city[i]]
        w = np.array([DISTRICTS[k].density for k in idx])
        d = DISTRICTS[rng.choice(idx, p=w / w.sum())]
        hlat, hlon = sample_points(rng, d, 1)
        income[i] = d.income
        has_work = rng.random() < 0.6
        if has_work:
            wd = DISTRICTS[rng.choice(idx, p=w / w.sum())]
            wlat, wlon = sample_points(rng, wd, 1)
        ios = rng.random() < np.clip(0.35 + 0.25 * (d.income - 1), 0.15, 0.7)
        rows.append({
            "user_id": i, "home_city": city[i], "district": d.code,
            "home_lat": float(hlat[0]), "home_lon": float(hlon[0]),
            "work_lat": float(wlat[0]) if has_work else np.nan,
            "work_lon": float(wlon[0]) if has_work else np.nan,
            "signup_date": _signup(rng, seg[i], cfg), "platform": "ios" if ios else "android",
            "channel": rng.choice(["organic", "referral", "paid_social", "promo_code"], p=[0.5, 0.15, 0.2, 0.15]),
            "age_band": AGE_BANDS[age[i]], "segment": seg[i],
        })
    users = pd.DataFrame(rows)
    return users, _truth(rng, users, age, income)


def _truth(rng: np.random.Generator, users: pd.DataFrame, age: np.ndarray, income: np.ndarray) -> UserTruth:
    n, k = len(users), len(CUISINES)
    city_i = (users["home_city"] == "MAD").to_numpy().astype(int)
    base = np.log(np.array([[c.popularity * c.city_mult[ci] for c in CUISINES] for ci in (0, 1)]))
    log_aff = base[city_i] + rng.normal(0, 0.9, (n, k))
    favs = rng.integers(0, k, (n, 3))
    np.put_along_axis(log_aff, favs, np.take_along_axis(log_aff, favs, 1) + 1.0, axis=1)
    young = (age <= 1).astype(float)[:, None]
    for key in YOUNG_CUISINES:
        log_aff[:, [c.key for c in CUISINES].index(key)] += 0.5 * young[:, 0] - 0.3 * (age >= 3)
    for key in OLDER_CUISINES:
        log_aff[:, [c.key for c in CUISINES].index(key)] += 0.4 * (age >= 2) - 0.2 * young[:, 0]
    ios = (users["platform"] == "ios").to_numpy()
    price_sens = (1.1 * income ** -0.9 * np.exp(rng.normal(0, 0.3, n)) * np.where(age == 0, 1.25, 1.0)
                  * np.where(ios, 0.85, 1.0))
    seg = users["segment"].to_numpy()
    rate = np.minimum(np.exp(rng.normal(np.log(1.8), 0.5, n)), 5.5)   # cap: ~4 sessions/week after scaling
    rate = np.where(seg == "light", rng.uniform(0.08, 0.3, n), rate)
    rate = np.where(seg == "growth", rate * 0.7, rate)
    dayp = rng.dirichlet(np.array([0.08, 0.33, 0.08, 0.43, 0.08]) * 12, n)
    owl = rng.random(n) < 0.15
    dayp[owl, 4] += 0.25
    brk = rng.random(n) < 0.10
    dayp[brk, 0] += 0.25
    dayp /= dayp.sum(1, keepdims=True)
    return UserTruth(
        log_aff=log_aff.astype(np.float32),
        z=(rng.normal(0, 0.6, (n, LATENT_DIM))).astype(np.float32),
        price_sens=price_sens.astype(np.float32),
        dist_sens=np.exp(rng.normal(np.log(0.22), 0.4, n)).astype(np.float32),
        vegetarian=rng.random(n) < 0.08,
        spicy_tol=rng.normal(0, 1, n).astype(np.float32),
        habit=(rng.beta(2, 2, n) * 2.2).astype(np.float32),
        daypart_w=dayp.astype(np.float32),
        weekly_rate=rate.astype(np.float32),
        conv_base=np.where(seg == "light", -0.4, 0.4).astype(np.float32) + rng.normal(0, 0.4, n).astype(np.float32),
        ref_price=(11.0 * income ** 0.5 * np.exp(rng.normal(0, 0.15, n))).astype(np.float32),
    )


@dataclass(frozen=True)
class Trip:
    user_id: int
    start: date
    days: int
    lat: float
    lon: float


def generate_trips(cfg: WorldConfig, rng: np.random.Generator, users: pd.DataFrame) -> list[Trip]:
    """A share of users visit the other city 1-3 times; they stay near its centre."""
    trips: list[Trip] = []
    centres = {"BCN": [d for d in DISTRICTS if d.code in ("BCN01", "BCN02", "BCN06")],
               "MAD": [d for d in DISTRICTS if d.code in ("MAD01", "MAD04", "MAD07")]}
    for u in users.itertuples():
        if rng.random() >= cfg.traveller_share:
            continue
        other = "MAD" if u.home_city == "BCN" else "BCN"
        first = max(cfg.start, u.signup_date)
        span = (cfg.end - first).days
        if span < 5:
            continue
        for _ in range(int(rng.integers(1, 4))):
            d = centres[other][int(rng.integers(0, 3))]
            lat, lon = sample_points(rng, d, 1)
            trips.append(Trip(int(u.user_id), first + timedelta(days=int(rng.integers(0, span - 4))),
                              int(rng.integers(2, 6)), float(lat[0]), float(lon[0])))
    return trips


CITY_NAMES = {c.code: c.name for c in CITIES}
