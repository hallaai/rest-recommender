"""Items from menus, hidden item parameters, and weather."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .menus import MenuSpec
from .users import LATENT_DIM
from .vocab import CATEGORIES, CHAINS, CUISINE_INDEX, CUISINES
from .world import WorldConfig

CHAIN_NAMES = {c.key: c.name for c in CHAINS}


def build_items(restaurants: pd.DataFrame, menus: dict[str, dict], specs: list[MenuSpec],
                rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Returns (restaurants with names, items, hidden per-item arrays).

    Chain outlets share the chain's menu: same dish names and `sku`, so the same dish in
    two cities has the same hidden taste vector but different item ids.
    """
    sku_latent: dict[str, tuple[np.ndarray, float]] = {}
    rows, names, taglines, seen = [], [], [], set()
    for r in restaurants.itertuples():
        source = r.chain or f"r{r.restaurant_id}"
        menu = menus[source]
        name = CHAIN_NAMES.get(r.chain) or menu.get("name") or f"Restaurant {r.restaurant_id}"
        if not r.chain and name.lower() in seen:
            name = f"{name} ({r.district})"
        seen.add(name.lower())
        names.append(name)
        taglines.append(menu.get("tagline", ""))
        for k, it in enumerate(menu["items"]):
            sku = f"{source}:{k}"
            if sku not in sku_latent:
                sku_latent[sku] = (rng.normal(0, 0.5, LATENT_DIM).astype(np.float32), float(rng.normal(0, 0.3)))
            rows.append({"restaurant_id": r.restaurant_id, "sku": sku, **it})
    rest = restaurants.assign(name=names, tagline=taglines)
    items = pd.DataFrame(rows)
    items.insert(0, "item_id", np.arange(len(items), dtype=np.int32))
    items["tags"] = items["tags"].apply(lambda t: list(t))
    z = np.stack([sku_latent[s][0] for s in items["sku"]])
    dish_q = np.array([sku_latent[s][1] for s in items["sku"]], dtype=np.float32)
    rq = rest.set_index("restaurant_id").loc[items["restaurant_id"], "quality"].to_numpy(np.float32)
    hidden = {"z": z, "quality": dish_q + rq}
    return rest, items, hidden


def item_arrays(rest: pd.DataFrame, items: pd.DataFrame) -> dict[str, np.ndarray]:
    """Visible item fields as arrays in item_id order (used by truth and simulation)."""
    cuisine_of_rest = rest.set_index("restaurant_id")["cuisine"]
    return {
        "restaurant": items["restaurant_id"].to_numpy(np.int64),
        "cuisine": np.array([CUISINE_INDEX[cuisine_of_rest[r]] for r in items["restaurant_id"]], np.int64),
        "category": items["category"].map({c: i for i, c in enumerate(CATEGORIES)}).to_numpy(np.int64),
        "log_price": np.log(items["price"].to_numpy(np.float32)),
        "vegetarian": items["vegetarian"].to_numpy(bool),
        "spicy": items["spicy"].to_numpy(bool),
    }


# Rain probability and mean daily max temperature (C) by month, May..Oct.
_RAIN = {"BCN": (0.25, 0.15, 0.08, 0.12, 0.25, 0.30), "MAD": (0.20, 0.10, 0.05, 0.05, 0.15, 0.30)}
_TMAX = {"BCN": (22, 26, 29, 29, 26, 22), "MAD": (24, 30, 34, 33, 28, 21)}


def generate_weather(cfg: WorldConfig, rng: np.random.Generator) -> pd.DataFrame:
    days = (cfg.end - cfg.start).days + 1
    rows = []
    for city in ("BCN", "MAD"):
        prev_rain = False
        for d in range(days):
            day = cfg.start + timedelta(days=d)
            m = day.month - 5
            p = _RAIN[city][m] * (1.8 if prev_rain else 1.0)       # rainy spells
            rain = bool(rng.random() < min(p, 0.8))
            tmax = _TMAX[city][m] + rng.normal(0, 2.5) - (3 if rain else 0)
            rows.append({"date": day, "city": city, "rain": rain, "temp_max_c": round(float(tmax), 1)})
            prev_rain = rain
    return pd.DataFrame(rows)


def prior_popularity(items: pd.DataFrame, rest: pd.DataFrame, quality: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """The old production feed's score (the logging policy): noisy quality plus cuisine popularity."""
    pop = {c.key: np.log(c.popularity) for c in CUISINES}
    rr = rest.set_index("restaurant_id")
    cuis = rr.loc[items["restaurant_id"], "cuisine"].to_numpy()
    rating = rr.loc[items["restaurant_id"], "rating"].to_numpy()
    return (0.5 * quality + 0.5 * np.array([pop[c] for c in cuis]) + 0.6 * (rating - 4.3)
            + rng.normal(0, 0.3, len(items))).astype(np.float32)
