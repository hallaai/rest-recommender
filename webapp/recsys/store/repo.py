"""Read-only Postgres access for the Explore pages and the live vector search.

All SQL is parameterised; the schema name comes from settings (validated identifier).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ..store.loader import VECTOR_SPACES
from ..synth.world import SPLIT

MAX_PAGE = 200
log = logging.getLogger(__name__)


def _like(q: str) -> str:
    """Escape LIKE wildcards so a search is literal."""
    return q.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


class Repo:
    def __init__(self, database_url: str, schema: str) -> None:
        if not database_url:
            raise RuntimeError("DATABASE_URL is not set (webapp/.env).")
        self._schema = schema
        # The managed server drops idle connections: check each one before lending it, recycle
        # idle ones after 5 minutes, and keep TCP keepalives on.
        self._pool = ConnectionPool(
            database_url, min_size=1, max_size=6, open=True, check=ConnectionPool.check_connection,
            max_idle=300, max_lifetime=1800, timeout=10,
            kwargs={"row_factory": dict_row, "autocommit": True, "keepalives": 1, "keepalives_idle": 60,
                    "options": f"-c search_path={schema},public -c statement_timeout=15000"})
        self._pool.wait(timeout=10)     # fail at startup, not on the first request

    def close(self) -> None:
        self._pool.close()

    def _all(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        """Run a read query; one retry on a dropped connection (the pool discards the broken one)."""
        for attempt in (1, 2):
            try:
                with self._pool.connection() as c:
                    return c.execute(sql, params).fetchall()
            except psycopg.OperationalError:
                if attempt == 2:
                    raise
                log.warning("Database connection dropped; retrying once")
        raise AssertionError("unreachable")

    def _one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    # -- overview ------------------------------------------------------------------
    def overview(self) -> dict[str, Any]:
        counts = self._one("""
            SELECT (SELECT count(*) FROM restaurants) AS restaurants, (SELECT count(*) FROM items) AS items,
                   (SELECT count(*) FROM users) AS users, (SELECT count(*) FROM sessions) AS sessions,
                   (SELECT count(*) FROM orders) AS orders,
                   (SELECT round(avg(subtotal), 2) FROM orders) AS avg_basket_eur,
                   (SELECT round(avg(ordered::int)::numeric, 3) FROM sessions) AS conversion""")
        by_city = self._all("""
            SELECT r.city, count(DISTINCT r.restaurant_id) AS restaurants, count(i.item_id) AS items
            FROM restaurants r JOIN items i USING (restaurant_id) GROUP BY r.city ORDER BY r.city""")
        users_city = self._all("SELECT home_city AS city, count(*) AS users FROM users GROUP BY 1 ORDER BY 1")
        monthly = self._all("""
            SELECT to_char(date_trunc('month', ts), 'YYYY-MM') AS month, city, count(*) AS orders
            FROM orders GROUP BY 1, 2 ORDER BY 1, 2""")
        cuisine = self._all("""
            SELECT r.cuisine, count(*) AS orders FROM orders o JOIN restaurants r USING (restaurant_id)
            GROUP BY 1 ORDER BY 2 DESC""")
        hourly = self._all("""
            SELECT extract(hour FROM ts)::int AS hour, count(*) AS orders FROM orders GROUP BY 1 ORDER BY 1""")
        history = self._all("""
            WITH n AS (SELECT u.user_id, count(o.order_id) AS k FROM users u
                       LEFT JOIN orders o ON o.user_id = u.user_id AND o.ts::date <= %(val_end)s GROUP BY u.user_id)
            SELECT CASE WHEN k = 0 THEN 'no history' WHEN k <= 3 THEN '1-3 orders'
                        WHEN k <= 15 THEN '4-15 orders' ELSE '16+ orders' END AS bucket, count(*) AS users
            FROM n GROUP BY 1""", {"val_end": SPLIT.val_end})
        return {"counts": counts, "by_city": by_city, "users_by_city": users_city, "monthly": monthly,
                "cuisine": cuisine, "hourly": hourly, "history_before_test": history}

    # -- map -------------------------------------------------------------------------
    def map_restaurants(self) -> list[dict[str, Any]]:
        return self._all("""
            SELECT r.restaurant_id AS id, r.name, r.city, r.cuisine, r.chain, r.price_tier, r.rating,
                   r.lat, r.lon, r.delivery_radius_km, coalesce(o.n, 0) AS orders
            FROM restaurants r LEFT JOIN (SELECT restaurant_id, count(*) AS n FROM orders GROUP BY 1) o
              USING (restaurant_id) ORDER BY r.restaurant_id""")

    def map_users(self) -> list[dict[str, Any]]:
        return self._all("""
            SELECT u.user_id AS id, u.home_city AS city, u.home_lat AS lat, u.home_lon AS lon,
                   coalesce(o.n, 0) AS orders
            FROM users u LEFT JOIN (SELECT user_id, count(*) AS n FROM orders GROUP BY 1) o USING (user_id)""")

    def districts(self) -> list[dict[str, Any]]:
        return self._all("SELECT code, city, name, lat, lon, density, income FROM districts ORDER BY code")

    def restaurants_near(self, lat: float, lon: float) -> list[dict[str, Any]]:
        """Restaurants that deliver to a point (PostGIS distance vs each restaurant's radius).
        PostGIS measures on the spheroid and the simulator on a sphere: they can disagree by
        about 0.3% of the distance for a restaurant right at its radius edge."""
        return self._all("""
            SELECT restaurant_id AS id, name, cuisine,
                   round((ST_Distance(geog, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)::geography) / 1000)::numeric, 2) AS km
            FROM restaurants
            WHERE ST_DWithin(geog, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)::geography, delivery_radius_km * 1000)
            ORDER BY km""", {"lat": lat, "lon": lon})

    # -- restaurants & items -----------------------------------------------------------
    def restaurants(self, city: str | None, cuisine: str | None, q: str | None,
                    limit: int, offset: int) -> dict[str, Any]:
        where, p = ["TRUE"], {"limit": min(limit, MAX_PAGE), "offset": offset}
        if city:
            where.append("r.city = %(city)s")
            p["city"] = city
        if cuisine:
            where.append("r.cuisine = %(cuisine)s")
            p["cuisine"] = cuisine
        if q:
            where.append("r.name ILIKE %(q)s ESCAPE '\'")
            p["q"] = f"%{_like(q)}%"
        w = " AND ".join(where)
        rows = self._all(f"""
            SELECT r.restaurant_id AS id, r.name, r.city, d.name AS district, r.cuisine, r.chain, r.price_tier,
                   r.rating, r.n_ratings, (SELECT count(*) FROM items i WHERE i.restaurant_id = r.restaurant_id) AS items,
                   coalesce(o.n, 0) AS orders, count(*) OVER () AS total
            FROM restaurants r JOIN districts d ON d.code = r.district
            LEFT JOIN (SELECT restaurant_id, count(*) AS n FROM orders GROUP BY 1) o USING (restaurant_id)
            WHERE {w} ORDER BY orders DESC, r.restaurant_id LIMIT %(limit)s OFFSET %(offset)s""", p)
        return {"total": rows[0]["total"] if rows else 0, "rows": [{k: v for k, v in r.items() if k != "total"} for r in rows]}

    def restaurant(self, rid: int) -> dict[str, Any] | None:
        r = self._one("""
            SELECT r.restaurant_id AS id, r.name, r.tagline, r.city, d.name AS district, r.cuisine, r.chain,
                   r.price_tier, r.rating, r.n_ratings, r.lat, r.lon, r.delivery_radius_km, r.prep_min, r.hours
            FROM restaurants r JOIN districts d ON d.code = r.district WHERE r.restaurant_id = %s""", (rid,))
        if not r:
            return None
        r["menu"] = self._all("""
            SELECT i.item_id AS id, i.name, i.description, i.category, i.price, i.vegetarian, i.vegan, i.spicy,
                   i.tags, coalesce(s.n, 0) AS orders
            FROM items i LEFT JOIN (SELECT item_id, sum(qty) AS n FROM order_items GROUP BY 1) s USING (item_id)
            WHERE i.restaurant_id = %s
            ORDER BY array_position(ARRAY['main','side','drink','dessert'], i.category), orders DESC""", (rid,))
        r["orders_by_hour"] = self._all("""
            SELECT extract(hour FROM ts)::int AS hour, count(*) AS orders FROM orders
            WHERE restaurant_id = %s GROUP BY 1 ORDER BY 1""", (rid,))
        return r

    def items(self, city: str | None, cuisine: str | None, category: str | None, q: str | None,
              vegetarian: bool | None, limit: int, offset: int) -> dict[str, Any]:
        where, p = ["TRUE"], {"limit": min(limit, MAX_PAGE), "offset": offset}
        for key, col in (("city", "r.city"), ("cuisine", "r.cuisine"), ("category", "i.category")):
            val = {"city": city, "cuisine": cuisine, "category": category}[key]
            if val:
                where.append(f"{col} = %({key})s")
                p[key] = val
        if vegetarian is not None:
            where.append("i.vegetarian = %(veg)s")
            p["veg"] = vegetarian
        if q:
            where.append("(i.name ILIKE %(q)s ESCAPE '\' OR i.description ILIKE %(q)s ESCAPE '\')")
            p["q"] = f"%{_like(q)}%"
        rows = self._all(f"""
            SELECT i.item_id AS id, i.name, i.description, i.category, i.price, i.vegetarian, i.spicy,
                   r.restaurant_id, r.name AS restaurant, r.city, r.cuisine, coalesce(s.n, 0) AS orders,
                   count(*) OVER () AS total
            FROM items i JOIN restaurants r USING (restaurant_id)
            LEFT JOIN (SELECT item_id, sum(qty) AS n FROM order_items GROUP BY 1) s USING (item_id)
            WHERE {' AND '.join(where)} ORDER BY orders DESC, i.item_id LIMIT %(limit)s OFFSET %(offset)s""", p)
        return {"total": rows[0]["total"] if rows else 0, "rows": [{k: v for k, v in r.items() if k != "total"} for r in rows]}

    # -- users -----------------------------------------------------------------------
    def users(self, city: str | None, activity: str | None, limit: int, offset: int) -> dict[str, Any]:
        p: dict[str, Any] = {"limit": min(limit, MAX_PAGE), "offset": offset}
        where = ["TRUE"]
        if city:
            where.append("u.home_city = %(city)s")
            p["city"] = city
        having = {"none": "coalesce(o.n, 0) = 0", "light": "coalesce(o.n, 0) BETWEEN 1 AND 3",
                  "regular": "coalesce(o.n, 0) BETWEEN 4 AND 15", "heavy": "coalesce(o.n, 0) > 15",
                  "traveller": "coalesce(o.away, 0) > 0"}.get(activity or "", "TRUE")
        rows = self._all(f"""
            SELECT u.user_id AS id, u.home_city, d.name AS district, u.signup_date, u.platform, u.age_band,
                   coalesce(o.n, 0) AS orders, coalesce(o.away, 0) AS orders_away, o.last_order,
                   count(*) OVER () AS total
            FROM users u JOIN districts d ON d.code = u.district
            LEFT JOIN (SELECT user_id, count(*) AS n, max(ts) AS last_order,
                              count(*) FILTER (WHERE o.city <> uu.home_city) AS away
                       FROM orders o JOIN users uu USING (user_id) GROUP BY 1) o USING (user_id)
            WHERE {' AND '.join(where)} AND {having}
            ORDER BY orders DESC, u.user_id LIMIT %(limit)s OFFSET %(offset)s""", p)
        return {"total": rows[0]["total"] if rows else 0, "rows": [{k: v for k, v in r.items() if k != "total"} for r in rows]}

    def user(self, uid: int) -> dict[str, Any] | None:
        u = self._one("""
            SELECT u.user_id AS id, u.home_city, d.name AS district, u.home_lat, u.home_lon, u.work_lat, u.work_lon,
                   u.signup_date, u.platform, u.channel, u.age_band
            FROM users u JOIN districts d ON d.code = u.district WHERE u.user_id = %s""", (uid,))
        if not u:
            return None
        u["orders"] = self._all("""
            SELECT o.order_id AS id, o.ts, o.city, o.subtotal, o.delivery_fee, o.distance_km, o.lat, o.lon,
                   r.restaurant_id, r.name AS restaurant, r.cuisine, s.location_kind,
                   json_agg(json_build_object('id', i.item_id, 'name', i.name, 'qty', oi.qty,
                                              'category', i.category) ORDER BY oi.is_main DESC) AS items
            FROM orders o JOIN restaurants r USING (restaurant_id) JOIN sessions s USING (session_id)
            JOIN order_items oi USING (order_id) JOIN items i ON i.item_id = oi.item_id
            WHERE o.user_id = %s GROUP BY o.order_id, r.restaurant_id, s.location_kind ORDER BY o.ts DESC LIMIT 300""", (uid,))
        u["sessions"] = self._one("""
            SELECT count(*) AS sessions, round(avg(ordered::int)::numeric, 3) AS conversion,
                   count(*) FILTER (WHERE location_kind = 'trip') AS trip_sessions
            FROM sessions WHERE user_id = %s""", (uid,))
        return u

    # -- vector search ---------------------------------------------------------------
    def nearest_items(self, space: str, query: np.ndarray, restaurant_ids: list[int], k: int,
                      ef_search: int = 100) -> list[tuple[int, float]]:
        """Top-k items by cosine similarity among the given restaurants (HNSW + iterative scan)."""
        if space not in VECTOR_SPACES:
            raise ValueError("unknown vector space")
        vec = "[" + ",".join(f"{x:.6f}" for x in query) + "]"
        with self._pool.connection() as c:
            with c.transaction():
                c.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
                c.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                rows = c.execute(f"""
                    SELECT v.item_id, 1 - (v.v <=> %(q)s::vector) AS sim
                    FROM vec_{space} v JOIN items i USING (item_id)
                    WHERE i.restaurant_id = ANY(%(r)s) AND i.category = 'main'
                    ORDER BY v.v <=> %(q)s::vector LIMIT %(k)s""",
                                 {"q": vec, "r": restaurant_ids, "k": k}).fetchall()
        return [(r["item_id"], float(r["sim"])) for r in rows]

    def ann_search(self, space: str, query: np.ndarray, k: int, *, strategy: str = "hnsw",
                   ef_search: int = 100, probes: int = 10) -> list[tuple[int, float]]:
        """Top-k items over the WHOLE catalogue by cosine, with a chosen index strategy.

        strategy: 'exact' (no index: planner forced to sequential scan — the correctness
        reference), 'hnsw' (ef_search knob) or 'ivfflat' (probes knob). Used by the ANN
        benchmark to measure recall vs latency; not on the serving path.
        """
        if space not in VECTOR_SPACES:
            raise ValueError("unknown vector space")
        if strategy not in ("exact", "hnsw", "ivfflat"):
            raise ValueError("unknown strategy")
        vec = "[" + ",".join(f"{x:.6f}" for x in query) + "]"
        with self._pool.connection() as c:
            with c.transaction():
                if strategy == "exact":
                    c.execute("SET LOCAL enable_indexscan = off")
                    c.execute("SET LOCAL enable_indexonlyscan = off")
                elif strategy == "hnsw":
                    c.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
                else:
                    c.execute(f"SET LOCAL ivfflat.probes = {int(probes)}")
                rows = c.execute(f"""
                    SELECT item_id, 1 - (v <=> %(q)s::vector) AS sim
                    FROM vec_{space}
                    ORDER BY v <=> %(q)s::vector LIMIT %(k)s""",
                                 {"q": vec, "k": k}).fetchall()
        return [(r["item_id"], float(r["sim"])) for r in rows]
