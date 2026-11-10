"""Restaurants: location by district density, cuisine by city taste, price tier by district
income, Spanish opening hours, and outlets of fictional chains in both cities.

`quality` is hidden ground truth; `rating` is what the app shows (quality plus noise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geo import CITIES, DISTRICTS, sample_points
from .vocab import CHAINS, CUISINES
from .world import WorldConfig


def _opening_windows(rng: np.random.Generator, hours: tuple[int, int]) -> list[tuple[int, int]]:
    """List of (open, close) hours; close may exceed 24 (after midnight). 25% of
    lunch-and-dinner places close between services, as many Spanish restaurants do."""
    o, c = hours
    o = int(np.clip(o + rng.integers(-1, 2), 7, 14))
    c = int(np.clip(c + rng.integers(-1, 2), o + 6, 26))
    if o <= 13 and c >= 23 and rng.random() < 0.25:
        return [(o, 16), (20, c)]
    return [(o, c)]


def _price_tier(rng: np.random.Generator, income: float) -> int:
    p_high = np.clip(0.12 + 0.35 * (income - 1.0), 0.03, 0.55)
    p_low = np.clip(0.35 - 0.35 * (income - 1.0), 0.08, 0.6)
    u = rng.random()
    return 3 if u < p_high else (1 if u < p_high + p_low else 2)


def _cuisine_probs(city_idx: int) -> np.ndarray:
    w = np.array([c.popularity * c.city_mult[city_idx] for c in CUISINES])
    return w / w.sum()


def _district_probs(city: str) -> tuple[list[int], np.ndarray]:
    idx = [i for i, d in enumerate(DISTRICTS) if d.city == city]
    w = np.array([DISTRICTS[i].density for i in idx])
    return idx, w / w.sum()


def generate_restaurants(cfg: WorldConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict] = []
    for ci, city in enumerate(CITIES):
        dist_idx, dist_p = _district_probs(city.code)
        chain_slots = [(ch, ch.outlets[ci]) for ch in CHAINS]
        n_chain = sum(n for _, n in chain_slots)
        n_indep = cfg.restaurants[ci] - n_chain
        plan: list[tuple[str, str, int | None]] = []   # (cuisine, chain key or "", tier)
        for ch, n in chain_slots:
            plan += [(ch.cuisine, ch.key, ch.price_tier)] * n
        cuis = rng.choice(len(CUISINES), size=n_indep, p=_cuisine_probs(ci))
        plan += [(CUISINES[k].key, "", None) for k in cuis]
        districts = rng.choice(dist_idx, size=len(plan), p=dist_p)
        for (cuisine, chain, tier), di in zip(plan, districts):
            d = DISTRICTS[di]
            lat, lon = sample_points(rng, d, 1)
            ck = next(c for c in CUISINES if c.key == cuisine)
            quality = rng.normal(0.0, 0.35) + (0.1 if chain else 0.0)
            rows.append({
                "city": city.code, "district": d.code, "lat": float(lat[0]), "lon": float(lon[0]),
                "cuisine": cuisine, "chain": chain,
                "price_tier": tier if tier else _price_tier(rng, d.income),
                "quality": quality,
                "rating": float(np.clip(np.round(4.3 + 0.5 * quality + rng.normal(0, 0.12), 1), 3.2, 4.9)),
                "n_ratings": int(rng.lognormal(5.0, 1.0)) + 5,
                "delivery_radius_km": float(np.round(rng.uniform(3.0, 5.0) * (1.15 if city.code == "MAD" else 1.0), 1)),
                "prep_min": int(rng.integers(8, 28)),
                "hours": _opening_windows(rng, ck.hours),
            })
    df = pd.DataFrame(rows)
    df.insert(0, "restaurant_id", np.arange(len(df), dtype=np.int32))
    return df


def open_mask(hours: list[list[tuple[int, int]]], hour: int) -> np.ndarray:
    """Which restaurants are open at a given hour (0-23); windows past midnight wrap."""
    out = np.zeros(len(hours), dtype=bool)
    for i, windows in enumerate(hours):
        for o, c in windows:
            if o <= hour < c or (c > 24 and hour < c - 24):
                out[i] = True
                break
    return out


def open_matrix(hours: list[list[tuple[int, int]]]) -> np.ndarray:
    """(24, n_restaurants) bool: open at each hour."""
    return np.stack([open_mask(hours, h) for h in range(24)])
