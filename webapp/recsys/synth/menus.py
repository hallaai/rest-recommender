"""Menus: one Gemini call per independent restaurant and one per chain (outlets share it).

Replies are validated (closed category set, prices clamped to a plausible range, booleans
coerced) and cached in data/menus_cache.json keyed by menu source, so the world is
generated once and rebuilt for free. Without Gemini a template menu is used (tests, offline).
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..geo import DISTRICTS, DISTRICT_INDEX
from ..llm import Gemini, LLMError
from .vocab import CATEGORIES, CHAINS, CUISINES, FALLBACK_ADDONS

log = logging.getLogger(__name__)
TIER_MULT = {1: 0.8, 2: 1.0, 3: 1.35}
TIER_SIGN = {1: "€", 2: "€€", 3: "€€€"}


@dataclass(frozen=True)
class MenuSpec:
    source: str                 # "r<id>" for an independent restaurant, chain key otherwise
    cuisine: str
    city: str
    district: str
    price_tier: int
    counts: tuple[int, int, int, int]   # main, side, drink, dessert
    price_lo: float
    price_hi: float


def build_specs(restaurants: pd.DataFrame, rng: np.random.Generator) -> list[MenuSpec]:
    specs: list[MenuSpec] = []
    seen_chains: set[str] = set()
    for r in restaurants.itertuples():
        if r.chain and r.chain in seen_chains:
            continue
        source = r.chain or f"r{r.restaurant_id}"
        if r.chain:
            seen_chains.add(r.chain)
        cz = next(c for c in CUISINES if c.key == r.cuisine)
        m = TIER_MULT[r.price_tier]
        n_main = int(rng.integers(9, 17))
        counts = (n_main, int(rng.integers(2, 6)), int(rng.integers(3, 7)), int(rng.integers(1, 4)))
        if r.cuisine in ("desserts", "bakery"):
            counts = (n_main, counts[1], counts[2] + 2, 0)
        specs.append(MenuSpec(source, r.cuisine, r.city, DISTRICTS[DISTRICT_INDEX[r.district]].name,
                              int(r.price_tier), counts, round(cz.main_price[0] * m, 1),
                              round(cz.main_price[1] * m, 1)))
    return specs


def _prompt(spec: MenuSpec) -> str:
    n_main, n_side, n_drink, n_dessert = spec.counts
    cz = next(c for c in CUISINES if c.key == spec.cuisine)
    chain = next((c for c in CHAINS if c.key == spec.source), None)
    who = (f'the fictional restaurant chain "{chain.name}"' if chain else
           f"a fictional {cz.label} restaurant in {spec.district}, {'Barcelona' if spec.city == 'BCN' else 'Madrid'}, Spain")
    name_rule = (f'Use "{chain.name}" as the name.' if chain else
                 "Invent a believable, fictional restaurant name (not a real brand or a real restaurant).")
    return f"""Write the delivery-app menu of {who}. Cuisine: {cz.label}. Price level: {TIER_SIGN[spec.price_tier]}.
{name_rule}
Return JSON only:
{{"name": str, "tagline": str (max 12 words),
 "items": [{{"name": str, "description": str (8-20 words, appetising, English),
            "category": "main"|"side"|"drink"|"dessert", "price_eur": number,
            "vegetarian": bool, "vegan": bool, "spicy": bool, "tags": [str, max 3]}}]}}
Include exactly {n_main} items with category "main" (prices {spec.price_lo}-{spec.price_hi} EUR),
{n_side} "side" (starters, sides), {n_drink} "drink", {n_dessert} "dessert".
Dish names may be in Spanish, Catalan or the cuisine's language where natural. Vary the dishes;
no duplicates. About {int(cz.veg_share * 100)}% of mains vegetarian, {int(cz.spicy_share * 100)}% spicy."""


def _clean_item(raw: dict[str, Any], spec: MenuSpec) -> dict[str, Any] | None:
    cat = str(raw.get("category", "")).strip().lower()
    name = str(raw.get("name", "")).strip()[:80]
    if cat not in CATEGORIES or not name:
        return None
    try:
        price = float(raw.get("price_eur", 0))
    except (TypeError, ValueError):
        return None
    lo, hi = (spec.price_lo * 0.7, spec.price_hi * 1.4) if cat == "main" else (0.8, 12.0)
    if not np.isfinite(price) or price <= 0:
        return None
    vegan = bool(raw.get("vegan", False))
    tags = [str(t)[:24] for t in (raw.get("tags") or []) if isinstance(t, (str, int))][:3]
    return {"name": name, "description": str(raw.get("description", "")).strip()[:240],
            "category": cat, "price": round(float(np.clip(price, lo, hi)), 2),
            "vegetarian": bool(raw.get("vegetarian", False)) or vegan, "vegan": vegan,
            "spicy": bool(raw.get("spicy", False)), "tags": tags}


def clean_menu(raw: Any, spec: MenuSpec) -> dict[str, Any] | None:
    """Validated menu, or None when the reply is unusable (fewer than 4 mains)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        return None
    items, names = [], set()
    for it in raw["items"]:
        if isinstance(it, dict) and (c := _clean_item(it, spec)) and c["name"].lower() not in names:
            names.add(c["name"].lower())
            items.append(c)
    if sum(i["category"] == "main" for i in items) < 4:
        return None
    return {"name": str(raw.get("name") or "").strip()[:60] or None,
            "tagline": str(raw.get("tagline") or "").strip()[:120], "items": items, "source": "gemini"}


def fallback_menu(spec: MenuSpec, rng: np.random.Generator) -> dict[str, Any]:
    cz = next(c for c in CUISINES if c.key == spec.cuisine)
    chain = next((c for c in CHAINS if c.key == spec.source), None)
    items = []
    styles = ("Classic", "House", "Deluxe", "Spicy", "Mini", "Family", "Chef's", "Garden")
    for k in range(spec.counts[0]):
        base = cz.examples[k % len(cz.examples)]
        name = base if k < len(cz.examples) else f"{styles[k % len(styles)]} {base.lower()}"
        items.append({"name": name, "description": f"{cz.label} favourite: {base.lower()}.",
                      "category": "main", "price": round(float(rng.uniform(spec.price_lo, spec.price_hi)), 2),
                      "vegetarian": bool(rng.random() < cz.veg_share), "vegan": False,
                      "spicy": bool(rng.random() < cz.spicy_share), "tags": [cz.key]})
    for cat, n in zip(("side", "drink", "dessert"), spec.counts[1:]):
        for j, (nm, p) in enumerate(FALLBACK_ADDONS[cat][:n]):
            items.append({"name": nm, "description": nm, "category": cat, "price": p,
                          "vegetarian": cat != "side" or j != 2, "vegan": cat == "drink",
                          "spicy": False, "tags": []})
    name = chain.name if chain else f"{cz.label.split(' ')[0]} {spec.district.split(' ')[0]} {spec.source[1:]}"
    return {"name": name, "tagline": "", "items": items, "source": "template"}


def fingerprint(spec: MenuSpec) -> str:
    """What a cached menu must match to be reused for a restaurant."""
    return f"{spec.cuisine}|{spec.city}|{spec.price_tier}|{spec.counts}"


def generate_menus(specs: list[MenuSpec], gemini: Gemini | None, cache_path: Path,
                   rng: np.random.Generator, workers: int = 8) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    # Drop entries made for a different restaurant spec (seed or world size changed).
    cache = {s.source: cache[s.source] for s in specs
             if s.source in cache and cache[s.source].get("spec") == fingerprint(s)}
    todo = [s for s in specs if s.source not in cache]
    if gemini and todo:
        log.info("Generating %d menus with Gemini (%d cached)", len(todo), len(cache))

        def one(spec: MenuSpec) -> tuple[str, dict[str, Any] | None]:
            try:
                menu = clean_menu(gemini.json(_prompt(spec), purpose="menu_generation", max_tokens=6000), spec)
                return spec.source, ({**menu, "spec": fingerprint(spec)} if menu else None)
            except Exception as e:   # one bad menu must not lose the batch; it falls back to a template
                log.warning("Menu %s failed: %s", spec.source, e if isinstance(e, LLMError) else type(e).__name__)
                return spec.source, None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, (src, menu) in enumerate(ex.map(one, todo), 1):
                if menu:
                    cache[src] = menu
                if i % 25 == 0:
                    log.info("  menus %d/%d", i, len(todo))
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    # Fallbacks are not cached: a later run with Gemini replaces them.
    return {s.source: cache.get(s.source) or fallback_menu(s, rng) for s in specs}
