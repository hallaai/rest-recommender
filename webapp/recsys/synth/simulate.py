"""Sessions, feed impressions, clicks and orders, drawn from the ground-truth model.

Each user is simulated independently with a seed derived from (world seed, user id), so the
output is identical for any number of worker processes. The feed shown in each session
comes from a simple logging policy (popular nearby + the user's past dishes + random);
shown dishes get an exposure boost (truth.EXPOSURE_BOOST), so about a fifth of orders come
from the feed and the rest from browsing or search, as in a real app. Impression logs
therefore cover only feed orders: a click/order model trained on them sees that selection bias.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ..geo import haversine_km
from . import truth as T
from .users import UserTruth

DOW_FACTOR = np.array([0.85, 0.85, 0.9, 1.0, 1.25, 1.3, 1.2])      # Mon..Sun
DAYPART_HOURS = (
    (np.array([8, 9, 10, 11]), np.array([0.2, 0.35, 0.3, 0.15])),
    (np.array([12, 13, 14, 15, 16]), np.array([0.1, 0.3, 0.35, 0.18, 0.07])),
    (np.array([17, 18, 19]), np.array([0.3, 0.4, 0.3])),
    (np.array([20, 21, 22]), np.array([0.3, 0.45, 0.25])),
    (np.array([23, 24, 25]), np.array([0.5, 0.35, 0.15])),          # 24, 25 = 00h, 01h next day
)
ADDON_P = {1: 0.35, 2: 0.45, 3: 0.18}                               # side, drink, dessert


@dataclass(frozen=True)
class SimContext:
    """Everything the per-user simulation reads; built once, shared with worker processes."""
    seed: int
    start: date
    n_days: int
    impressions: int
    regular_rate_scale: float
    it: T.ItemTruth
    ut: UserTruth
    item_price: np.ndarray
    rest_lat: np.ndarray
    rest_lon: np.ndarray
    rest_radius: np.ndarray
    open_hr: np.ndarray              # (24, R) bool
    rest_city: np.ndarray            # (R,) 0 BCN / 1 MAD
    prior_pop: np.ndarray            # (I,) logging-policy popularity score
    ctx_util: np.ndarray             # (5, 2, 2, I) daypart x rain x hot
    rain: np.ndarray                 # (2, n_days) bool
    hot: np.ndarray                  # (2, n_days) bool
    home: np.ndarray                 # (U, 2)
    work: np.ndarray                 # (U, 2), NaN when none
    home_city: np.ndarray            # (U,) 0/1
    first_day: np.ndarray            # (U,) day index from which the user is active
    trips: dict[int, list[tuple[int, int, float, float, int]]]   # user -> (start, days, lat, lon, city)


_CTX: SimContext | None = None
_REST_INDEX: tuple[list[np.ndarray], list[dict[int, np.ndarray]]] | None = None


def _init(ctx: SimContext) -> None:
    global _CTX, _REST_INDEX
    _CTX = ctx
    _REST_INDEX = _rest_index(ctx)


@dataclass
class _Location:
    mains: np.ndarray        # main-dish item ids within delivery range
    main_rest: np.ndarray    # their restaurant ids
    dist_u: np.ndarray       # distance utility per main
    km: np.ndarray           # km per main


def _location(c: SimContext, u: int, lat: float, lon: float, by_rest_main: list[np.ndarray]) -> _Location:
    km_r = haversine_km(lat, lon, c.rest_lat, c.rest_lon)
    rests = np.flatnonzero(km_r <= c.rest_radius)
    mains = np.concatenate([by_rest_main[r] for r in rests]) if len(rests) else np.zeros(0, np.int64)
    mr = c.it.restaurant[mains]
    return _Location(mains, mr, T.distance_utility(c.ut, u, km_r[mr]), km_r[mr])


def _feed(rng: np.random.Generator, c: SimContext, cand: np.ndarray, n_item: np.ndarray) -> np.ndarray:
    """Logging policy: indices into cand. Popular nearby, the user's past dishes, random."""
    k = min(c.impressions, len(cand))
    score = c.prior_pop[cand] + rng.gumbel(0, 0.6, len(cand))
    chosen = list(np.argsort(-score)[: k // 2])
    past = np.flatnonzero(n_item[cand] > 0)
    rng.shuffle(past)
    chosen += [p for p in past[:2] if p not in chosen]
    rest = np.setdiff1d(np.arange(len(cand)), chosen)
    chosen += list(rng.choice(rest, size=max(0, min(k - len(chosen), len(rest))), replace=False))
    chosen = np.array(chosen[:k])
    return chosen[np.argsort(-(c.prior_pop[cand[chosen]] + rng.normal(0, 0.8, len(chosen))))]


def simulate_user(u: int) -> dict[str, list]:
    c = _CTX
    assert c is not None
    rng = np.random.default_rng([c.seed, 7, u])
    assert _REST_INDEX is not None
    by_rest_main, by_rest_cat = _REST_INDEX
    static = T.static_utility(c.ut, u, c.it)
    n_item = np.zeros(len(c.item_price), np.float32)
    n_rest = np.zeros(len(c.rest_lat), np.float32)
    locs: dict[tuple, _Location] = {}
    out: dict[str, list] = {"sessions": [], "impressions": [], "lines": []}
    trip_days = {}
    for (s, days, lat, lon, city) in c.trips.get(u, []):
        for d in range(s, min(s + days, c.n_days)):
            trip_days[d] = (lat, lon, city)
    start_dow = c.start.weekday()
    for day in range(int(c.first_day[u]), c.n_days):
        dow = (start_dow + day) % 7
        season = 0.85 if (c.start + timedelta(days=day)).month == 8 else 1.0
        trip = trip_days.get(day)
        city = trip[2] if trip else int(c.home_city[u])
        rate = (c.ut.weekly_rate[u] / 7 * DOW_FACTOR[dow] * season * (1.25 if c.rain[city, day] else 1.0)
                * (1.6 if trip else 1.0) * c.regular_rate_scale)
        for _ in range(min(int(rng.poisson(rate)), 2)):
            _session(rng, c, u, day, dow, trip, city, static, n_item, n_rest, locs, by_rest_main,
                     by_rest_cat, out)
    return out


def _session(rng, c: SimContext, u, day, dow, trip, city, static, n_item, n_rest, locs, by_rest_main,
             by_rest_cat, out) -> None:
    dp = int(rng.choice(5, p=c.ut.daypart_w[u]))
    hours, hp = DAYPART_HOURS[dp]
    hour = int(rng.choice(hours, p=hp))
    if day == c.n_days - 1:
        hour = min(hour, 23)          # keep the last day's late sessions inside the window
    if trip:
        key, (lat, lon), kind = ("trip", trip[0]), (trip[0], trip[1]), 2
    elif dp == 1 and dow < 5 and not np.isnan(c.work[u, 0]) and rng.random() < 0.75:
        key, (lat, lon), kind = ("work",), tuple(c.work[u]), 1
    else:
        key, (lat, lon), kind = ("home",), tuple(c.home[u]), 0
    if key not in locs:
        locs[key] = _location(c, u, lat, lon, by_rest_main)
    loc = locs[key]
    rain, hot = bool(c.rain[city, day]), bool(c.hot[city, day])
    is_open = c.open_hr[hour % 24][loc.main_rest]
    cand, cand_rest = loc.mains[is_open], loc.main_rest[is_open]
    sid = len(out["sessions"])
    minute = int(rng.integers(0, 60))
    ts_min = (day * 24 + hour) * 60 + minute
    if len(cand) == 0:
        out["sessions"].append((u, ts_min, lat, lon, kind, city, dp, rain, hot, 0, False))
        return
    util = (static[cand] + c.ctx_util[dp, int(rain), int(hot)][cand] + loc.dist_u[is_open]
            + T.habit_utility(c.ut, u, n_item[cand], n_rest[cand_rest]))
    feed = _feed(rng, c, cand, n_item)
    shown = np.zeros(len(cand), bool)
    shown[feed] = True
    boosted = util + T.EXPOSURE_BOOST * shown
    ordered = rng.random() < T.conversion_prob(c.ut, u, T.logsumexp(boosted))
    pick = int(rng.choice(len(cand), p=T.softmax(boosted))) if ordered else -1
    for pos, j in enumerate(feed):
        clicked = j == pick or rng.random() < 1 / (1 + np.exp(-(util[j] - 2.2 - 0.12 * pos)))
        out["impressions"].append((sid, int(cand[j]), pos, bool(clicked), j == pick))
    out["sessions"].append((u, ts_min, lat, lon, kind, city, dp, rain, hot, len(cand), bool(ordered)))
    if not ordered:
        return
    item, r = int(cand[pick]), int(cand_rest[pick])
    qty = 2 if rng.random() < 0.15 else 1
    out["lines"].append((sid, item, qty, float(c.item_price[item]), True))
    n_item[item] += 1
    n_rest[r] += 1
    for cat, p in ADDON_P.items():
        opts = by_rest_cat[r][cat]
        if len(opts) and rng.random() < p:
            a = int(rng.choice(opts, p=T.softmax(static[opts] + c.ctx_util[dp, int(rain), int(hot)][opts])))
            out["lines"].append((sid, a, 1, float(c.item_price[a]), False))
            n_item[a] += 1


def _rest_index(c: SimContext) -> tuple[list[np.ndarray], list[dict[int, np.ndarray]]]:
    """Item ids per restaurant: mains, and add-ons by category."""
    n_r = len(c.rest_lat)
    order = np.argsort(c.it.restaurant, kind="stable")
    bounds = np.searchsorted(c.it.restaurant[order], np.arange(n_r + 1))
    mains, cats = [], []
    for r in range(n_r):
        ids = order[bounds[r]:bounds[r + 1]]
        mains.append(ids[c.it.category[ids] == 0])
        cats.append({k: ids[c.it.category[ids] == k] for k in (1, 2, 3)})
    return mains, cats


def run(ctx: SimContext, workers: int) -> list[dict[str, list]]:
    """Per-user results in user order (map preserves order, so any worker count gives the same output)."""
    users = range(len(ctx.home))
    if workers <= 1:
        _init(ctx)
        return [simulate_user(u) for u in users]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(ctx,)) as ex:
        return list(ex.map(simulate_user, users, chunksize=64))
