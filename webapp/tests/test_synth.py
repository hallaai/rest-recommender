"""Invariants of the synthetic world: it must look like a real log, with no impossible events."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from recsys.geo import haversine_km, is_land
from recsys.synth.generate import build_world
from recsys.synth.menus import MenuSpec, clean_menu, fallback_menu
from recsys.synth.restaurants import open_matrix
from recsys.synth.world import SPLIT

from .conftest import SMALL


def test_haversine_known_distance():
    # Barcelona (Plaça Catalunya) to Madrid (Puerta del Sol) is about 505 km.
    assert 495 < float(haversine_km(41.3870, 2.1700, 40.4169, -3.7035)) < 515
    assert float(haversine_km(41.0, 2.0, 41.0, 2.0)) == 0.0


def test_sea_and_parks_are_not_land():
    assert not is_land(np.array([41.37]), np.array([2.21]))[0]        # Mediterranean off Barceloneta
    assert not is_land(np.array([40.418]), np.array([-3.755]))[0]     # Casa de Campo
    assert is_land(np.array([41.3888]), np.array([2.1620]))[0]        # Eixample


def test_tables_are_consistent(small_world):
    v, _, _ = small_world
    items, rest, orders, oi = v["items"], v["restaurants"], v["orders"], v["order_items"]
    assert set(items["restaurant_id"]) <= set(rest["restaurant_id"])
    assert (items["price"] > 0).all()
    assert set(oi["order_id"]) == set(orders["order_id"])
    assert (oi.groupby("order_id")["is_main"].sum() == 1).all()       # exactly one main per order
    mains = oi[oi["is_main"]].merge(items[["item_id", "category"]], on="item_id")
    assert (mains["category"] == "main").all()
    # add-ons come from the same restaurant as the main
    with_rest = oi.merge(items[["item_id", "restaurant_id"]], on="item_id")
    assert (with_rest.groupby("order_id")["restaurant_id"].nunique() == 1).all()
    assert orders["subtotal"].gt(0).all()


def test_orders_only_from_open_restaurants_in_range(small_world, restaurant_hours):
    v, _, _ = small_world
    o = v["orders"].merge(v["restaurants"][["restaurant_id", "lat", "lon", "delivery_radius_km"]],
                          on="restaurant_id", suffixes=("", "_r"))
    km = haversine_km(o["lat"], o["lon"], o["lat_r"], o["lon_r"])
    assert (km <= o["delivery_radius_km"] + 1e-6).all()
    is_open = open_matrix(restaurant_hours)
    hours = pd.to_datetime(o["ts"]).dt.hour.to_numpy()
    # A 00h/01h order belongs to the previous evening's window; open_matrix handles the wrap.
    assert is_open[hours, o["restaurant_id"].to_numpy()].all()


def test_time_window_and_signup(small_world):
    v, _, _ = small_world
    s = v["sessions"].merge(v["users"][["user_id", "signup_date"]], on="user_id")
    ts = pd.to_datetime(s["ts"])
    assert ts.min() >= pd.Timestamp(SMALL.start)
    assert ts.max() < pd.Timestamp(SPLIT.test_end) + pd.Timedelta(days=1)
    assert (ts.dt.normalize() >= pd.to_datetime(s["signup_date"])).all()


def test_hidden_truth_is_not_in_visible_tables(small_world):
    v, _, _ = small_world
    assert "segment" not in v["users"].columns
    assert "quality" not in v["restaurants"].columns


def test_output_identical_for_any_worker_count(small_world, tmp_path):
    v1, _, _ = small_world
    v2, _, _ = build_world(replace(SMALL, workers=2), None, tmp_path)
    for name in ("sessions", "orders", "order_items", "impressions"):
        pd.testing.assert_frame_equal(v1[name].reset_index(drop=True), v2[name].reset_index(drop=True))


def test_realistic_behaviour(small_world):
    v, truth, _ = small_world
    s = v["sessions"]
    assert 0.4 < s["ordered"].mean() < 0.8                       # conversion
    lines = v["order_items"].merge(v["orders"][["order_id", "user_id", "ts"]], on="order_id")
    mains = lines[lines["is_main"]].sort_values("ts")
    repeat = mains.duplicated(["user_id", "item_id"]).mean()
    assert 0.1 < repeat < 0.6                                     # people reorder, but not only
    assert {"BCN", "MAD"} <= set(v["orders"]["city"])


def _spec(**kw) -> MenuSpec:
    base = dict(source="r1", cuisine="ramen", city="BCN", district="Gràcia", price_tier=2,
                counts=(5, 2, 2, 1), price_lo=12.0, price_hi=16.0)
    return MenuSpec(**{**base, **kw})


def test_clean_menu_validates_reply():
    raw = {"name": "Ramen Bar", "items": [
        {"name": f"Ramen {i}", "category": "main", "price_eur": 14, "vegetarian": False} for i in range(4)] + [
        {"name": "Bad", "category": "breakfast", "price_eur": 3},            # unknown category
        {"name": "Ramen 0", "category": "main", "price_eur": 14},             # duplicate
        {"name": "Gold ramen", "category": "main", "price_eur": 999},         # clamped
        {"name": "Vegan gyoza", "category": "side", "price_eur": 5, "vegan": True}]}
    m = clean_menu(raw, _spec())
    names = [i["name"] for i in m["items"]]
    assert "Bad" not in names and names.count("Ramen 0") == 1
    gold = next(i for i in m["items"] if i["name"] == "Gold ramen")
    assert gold["price"] <= 16.0 * 1.4
    assert next(i for i in m["items"] if i["name"] == "Vegan gyoza")["vegetarian"] is True


def test_clean_menu_rejects_too_few_mains():
    assert clean_menu({"items": [{"name": "x", "category": "main", "price_eur": 10}]}, _spec()) is None
    assert clean_menu("not json", _spec()) is None


def test_fallback_menu_matches_counts():
    m = fallback_menu(_spec(), np.random.default_rng(0))
    cats = pd.Series([i["category"] for i in m["items"]]).value_counts()
    assert cats["main"] == 5 and cats["side"] == 2 and cats["drink"] == 2 and cats["dessert"] == 1


def test_menu_cache_is_invalidated_when_spec_changes(tmp_path):
    import json
    from recsys.synth.menus import fingerprint, generate_menus
    spec = _spec()
    cached = {**fallback_menu(_spec(cuisine="sushi"), np.random.default_rng(1)), "spec": fingerprint(_spec(cuisine="sushi"))}
    (tmp_path / "menus.json").write_text(json.dumps({"r1": cached}), encoding="utf-8")
    menus = generate_menus([spec], None, tmp_path / "menus.json", np.random.default_rng(0))
    assert menus["r1"]["items"][0]["name"] != cached["items"][0]["name"]     # stale sushi menu not reused
