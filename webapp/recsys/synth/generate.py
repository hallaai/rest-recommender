"""Build the synthetic world and write it to data/ (visible) and data/truth/ (hidden).

    python -m recsys.synth.generate                # Gemini menus (cached), 8 workers
    python -m recsys.synth.generate --no-gemini    # template menus, no API calls
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, load_settings, quiet_http_logs
from ..geo import city_of, haversine_km
from ..llm import Gemini
from . import truth as T
from .catalog import build_items, generate_weather, item_arrays, prior_popularity
from .menus import build_specs, generate_menus
from .restaurants import generate_restaurants, open_matrix
from .simulate import SimContext, run
from .users import generate_trips, generate_users
from .world import WorldConfig

log = logging.getLogger(__name__)
LOCATION_KINDS = np.array(["home", "work", "trip"])
CITY_CODES = np.array(["BCN", "MAD"])


def build_context(cfg: WorldConfig, rest: pd.DataFrame, items: pd.DataFrame, hidden: dict,
                  users: pd.DataFrame, ut, trips, weather: pd.DataFrame, prior: np.ndarray) -> tuple[SimContext, T.ItemTruth]:
    arr = item_arrays(rest, items)
    it = T.ItemTruth(restaurant=arr["restaurant"], cuisine=arr["cuisine"], category=arr["category"],
                     log_price=arr["log_price"], vegetarian=arr["vegetarian"], spicy=arr["spicy"],
                     quality=hidden["quality"], z=hidden["z"])
    n_days = (cfg.end - cfg.start).days + 1
    ctx_util = np.stack([np.stack([np.stack([T.context_utility(it, dp, bool(r), bool(h)) for h in (0, 1)])
                                   for r in (0, 1)]) for dp in range(5)])
    w = weather.assign(day=(pd.to_datetime(weather["date"]) - pd.Timestamp(cfg.start)).dt.days)
    rain = np.zeros((2, n_days), bool)
    hot = np.zeros((2, n_days), bool)
    for ci, city in enumerate(CITY_CODES):
        ww = w[w["city"] == city]
        rain[ci, ww["day"]] = ww["rain"]
        hot[ci, ww["day"]] = ww["temp_max_c"] > 30
    trip_map: dict[int, list] = {}
    for t in trips:
        trip_map.setdefault(t.user_id, []).append(
            ((t.start - cfg.start).days, t.days, t.lat, t.lon, int(city_of(t.lat, t.lon)[0] == "MAD")))
    first_day = np.maximum((pd.to_datetime(users["signup_date"]) - pd.Timestamp(cfg.start)).dt.days, 0)
    ctx = SimContext(
        seed=cfg.seed, start=cfg.start, n_days=n_days, impressions=cfg.impressions_per_session,
        regular_rate_scale=cfg.regular_sessions_per_week / 1.8, it=it, ut=ut,
        item_price=items["price"].to_numpy(np.float32),
        rest_lat=rest["lat"].to_numpy(), rest_lon=rest["lon"].to_numpy(),
        rest_radius=rest["delivery_radius_km"].to_numpy(), open_hr=open_matrix(list(rest["hours"])),
        rest_city=(rest["city"] == "MAD").to_numpy().astype(int), prior_pop=prior, ctx_util=ctx_util,
        rain=rain, hot=hot, home=users[["home_lat", "home_lon"]].to_numpy(),
        work=users[["work_lat", "work_lon"]].to_numpy(),
        home_city=(users["home_city"] == "MAD").to_numpy().astype(int),
        first_day=first_day.to_numpy(), trips=trip_map)
    return ctx, it


def assemble(cfg: WorldConfig, results: list[dict], rest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-user outputs -> global tables with session_id / order_id assigned in user order."""
    s_rows, i_rows, l_rows = [], [], []
    offset = 0
    for res in results:
        s_rows += res["sessions"]
        i_rows += [(sid + offset, *rest_) for sid, *rest_ in res["impressions"]]
        l_rows += [(sid + offset, *rest_) for sid, *rest_ in res["lines"]]
        offset += len(res["sessions"])
    s = pd.DataFrame(s_rows, columns=["user_id", "ts_min", "lat", "lon", "kind", "city", "daypart",
                                      "rain", "hot", "n_available", "ordered"])
    s.insert(0, "session_id", np.arange(len(s), dtype=np.int64))
    s["ts"] = pd.Timestamp(cfg.start) + pd.to_timedelta(s.pop("ts_min"), unit="min")
    s["location_kind"] = LOCATION_KINDS[s.pop("kind")]
    s["city"] = CITY_CODES[s["city"]]
    s = s.sort_values(["ts", "session_id"]).reset_index(drop=True)
    imp = pd.DataFrame(i_rows, columns=["session_id", "item_id", "position", "clicked", "ordered"])
    lines = pd.DataFrame(l_rows, columns=["session_id", "item_id", "qty", "unit_price", "is_main"])
    ordered = s[s["ordered"]].sort_values(["ts", "session_id"])
    orders = ordered[["session_id", "user_id", "ts", "lat", "lon", "city"]].copy()
    orders.insert(0, "order_id", np.arange(len(orders), dtype=np.int64))
    main = lines[lines["is_main"]].set_index("session_id")["item_id"]
    return {"sessions": s, "impressions": imp, "orders": orders,
            "order_items": lines.merge(orders[["session_id", "order_id"]], on="session_id")
                               .drop(columns="session_id")[["order_id", "item_id", "qty", "unit_price", "is_main"]],
            "_main_item": main}


def finish_orders(tables: dict, items: pd.DataFrame, rest: pd.DataFrame) -> pd.DataFrame:
    o = tables["orders"]
    rid = items.set_index("item_id").loc[tables["_main_item"].loc[o["session_id"]].to_numpy(), "restaurant_id"]
    o = o.assign(restaurant_id=rid.to_numpy())
    rr = rest.set_index("restaurant_id").loc[o["restaurant_id"]]
    km = haversine_km(o["lat"].to_numpy(), o["lon"].to_numpy(), rr["lat"].to_numpy(), rr["lon"].to_numpy())
    oi = tables["order_items"]
    sub = (oi["qty"] * oi["unit_price"]).groupby(oi["order_id"]).sum()
    return o.assign(distance_km=np.round(km, 2), delivery_fee=np.round(T.delivery_fee(km), 2),
                    subtotal=np.round(sub.loc[o["order_id"]].to_numpy(), 2))


def write(out: Path, tables: dict[str, pd.DataFrame], truth: dict[str, dict[str, np.ndarray]], meta: dict) -> None:
    (out / "truth").mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    for name, arrs in truth.items():
        np.savez_compressed(out / "truth" / f"{name}.npz", **arrs)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def build_world(cfg: WorldConfig, gemini: Gemini | None, out: Path) -> tuple[dict, dict, dict]:
    """Generate everything; returns (visible tables, hidden truth arrays, meta). `out` holds the menu cache."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    rest = generate_restaurants(cfg, rng)
    specs = build_specs(rest, rng)
    out.mkdir(parents=True, exist_ok=True)
    menus = generate_menus(specs, gemini, out / "menus_cache.json", np.random.default_rng(cfg.seed + 1))
    rest, items, hidden = build_items(rest, menus, specs, np.random.default_rng(cfg.seed + 2))
    users, ut = generate_users(cfg, np.random.default_rng(cfg.seed + 3))
    trips = generate_trips(cfg, np.random.default_rng(cfg.seed + 4), users)
    weather = generate_weather(cfg, np.random.default_rng(cfg.seed + 5))
    prior = prior_popularity(items, rest, hidden["quality"], np.random.default_rng(cfg.seed + 6))
    ctx, it = build_context(cfg, rest, items, hidden, users, ut, trips, weather, prior)
    log.info("World: %d restaurants, %d items, %d users, %d trips. Simulating...",
             len(rest), len(items), len(users), len(trips))
    t1 = time.perf_counter()
    tables = assemble(cfg, run(ctx, cfg.workers), rest)
    orders = finish_orders(tables, items, rest)
    log.info("Simulated in %.1fs", time.perf_counter() - t1)
    visible = {
        "restaurants": rest.drop(columns=["quality"]).assign(hours=rest["hours"].apply(json.dumps)),
        "items": items, "users": users.drop(columns=["segment"]), "weather": weather,
        "sessions": tables["sessions"], "impressions": tables["impressions"], "orders": orders,
        "order_items": tables["order_items"],
    }
    truth = {
        "users": {**{k: v for k, v in asdict(ut).items()}, "segment": users["segment"].to_numpy().astype(str)},
        "items": {**{k: getattr(it, k) for k in ("quality", "z")}, "prior_pop": prior},
        "restaurants": {"quality": rest["quality"].to_numpy(np.float32)},
        "trips": {"user_id": np.array([t.user_id for t in trips]),
                  "start": np.array([t.start.isoformat() for t in trips]),
                  "days": np.array([t.days for t in trips])},
    }
    meta = {"config": asdict(cfg), "generated_at": datetime.now().isoformat(timespec="seconds"),
            "menu_sources": {k: v.get("source") for k, v in menus.items()},
            "seconds": round(time.perf_counter() - t0, 1)}
    return visible, truth, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gemini", action="store_true")
    ap.add_argument("--users", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_http_logs()
    cfg = WorldConfig()
    cfg = replace(cfg, users=args.users or cfg.users, workers=args.workers or cfg.workers)
    gemini = None if args.no_gemini else Gemini(load_settings())
    visible, truth, meta = build_world(cfg, gemini, args.out)
    write(args.out, visible, truth, meta)
    log.info("Wrote %s: %d sessions, %d orders, %d impressions in %.1fs", args.out, len(visible["sessions"]),
             len(visible["orders"]), len(visible["impressions"]), meta["seconds"])


if __name__ == "__main__":
    main()
