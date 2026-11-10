"""Load the generated world (data/*.parquet) and item vectors into Postgres.

    python -m recsys.store.loader                  # tables + every vector space found
    python -m recsys.store.loader --vectors-only   # (re)load vector tables only

Recreates the schema from scratch (DROP SCHEMA <schema> CASCADE) in one transaction, so it
only touches its own schema (config.validate_schema refuses public / pg_*) and a failed load
leaves the previous data in place. Bulk inserts use COPY. Each vector space gets a table vec_<space>
(item_id, v vector(dim)) with an HNSW cosine index.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from ..config import DATA_DIR, load_settings
from ..geo import CITIES, DISTRICTS

log = logging.getLogger(__name__)
SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
VECTOR_SPACES = ("gemini", "bge", "tfidf", "two_tower")

TABLE_COLUMNS = {
    "restaurants": ["restaurant_id", "city", "district", "name", "tagline", "cuisine", "chain", "price_tier",
                    "rating", "n_ratings", "lat", "lon", "delivery_radius_km", "prep_min", "hours"],
    "items": ["item_id", "restaurant_id", "sku", "name", "description", "category", "price", "vegetarian",
              "vegan", "spicy", "tags"],
    "users": ["user_id", "home_city", "district", "home_lat", "home_lon", "work_lat", "work_lon",
              "signup_date", "platform", "channel", "age_band"],
    "weather": ["date", "city", "rain", "temp_max_c"],
    "sessions": ["session_id", "user_id", "ts", "lat", "lon", "city", "location_kind", "daypart", "rain",
                 "hot", "n_available", "ordered"],
    "orders": ["order_id", "session_id", "user_id", "ts", "restaurant_id", "city", "lat", "lon",
               "distance_km", "delivery_fee", "subtotal"],
    "order_items": ["order_id", "item_id", "qty", "unit_price", "is_main"],
}


def _rows(df: pd.DataFrame, cols: list[str]) -> Iterable[tuple]:
    sub = df[cols].astype(object).where(df[cols].notna(), None)
    for row in sub.itertuples(index=False, name=None):
        yield tuple(list(v) if isinstance(v, np.ndarray) else v for v in row)


def _copy(conn: psycopg.Connection, table: str, cols: list[str], rows: Iterable[tuple]) -> int:
    n = 0
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as cp:
            for r in rows:
                cp.write_row(r)
                n += 1
    return n


def load_tables(conn: psycopg.Connection, schema: str, data_dir: Path) -> None:
    conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.execute(SCHEMA_SQL.replace("{schema}", schema))
    _copy(conn, "cities", ["code", "name", "lat", "lon"], ((c.code, c.name, c.lat, c.lon) for c in CITIES))
    _copy(conn, "districts", ["code", "city", "name", "lat", "lon", "density", "income"],
          ((d.code, d.city, d.name, d.lat, d.lon, d.density, d.income) for d in DISTRICTS))
    for table, cols in TABLE_COLUMNS.items():
        t0 = time.perf_counter()
        df = pd.read_parquet(data_dir / f"{table}.parquet")
        if table == "restaurants":
            df = df.assign(chain=df["chain"].replace("", None))
        n = _copy(conn, table, cols, _rows(df, cols))
        log.info("  %-12s %8d rows in %.1fs", table, n, time.perf_counter() - t0)
    conn.execute("UPDATE restaurants SET geog = ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography")
    for sql in (
        "CREATE INDEX ON restaurants USING gist (geog)",
        "CREATE INDEX ON restaurants (city, cuisine)",
        "CREATE INDEX ON items (restaurant_id)",
        "CREATE INDEX ON users (home_city)",
        "CREATE INDEX ON sessions (user_id, ts)",
        "CREATE INDEX ON orders (user_id, ts)",
        "CREATE INDEX ON orders (restaurant_id)",
        "CREATE INDEX ON order_items (order_id)",
        "CREATE INDEX ON order_items (item_id)",
    ):
        conn.execute(sql)
    conn.execute("ANALYZE")


def load_vectors(conn: psycopg.Connection, space: str, vecs: np.ndarray) -> float:
    """(Re)create vec_<space> and its HNSW index; returns index build seconds."""
    if space not in VECTOR_SPACES:
        raise ValueError(f"unknown vector space {space}")
    dim = int(vecs.shape[1])
    conn.execute(f"DROP TABLE IF EXISTS vec_{space}")
    conn.execute(f"CREATE TABLE vec_{space} (item_id int PRIMARY KEY REFERENCES items(item_id), v vector({dim}) NOT NULL)")
    rows = ((i, "[" + ",".join(f"{x:.6f}" for x in v) + "]") for i, v in enumerate(vecs))
    _copy(conn, f"vec_{space}", ["item_id", "v"], rows)
    t0 = time.perf_counter()
    conn.execute(f"CREATE INDEX vec_{space}_hnsw ON vec_{space} USING hnsw (v vector_cosine_ops) "
                 "WITH (m = 16, ef_construction = 64)")
    conn.execute(f"ANALYZE vec_{space}")
    return time.perf_counter() - t0


def read_fingerprint(database_url: str, schema: str) -> dict | None:
    """What the database currently holds (written by load()), or None if never loaded."""
    with psycopg.connect(database_url, autocommit=True, connect_timeout=10) as conn:
        exists = conn.execute("SELECT to_regclass(%s)", (f"{schema}.build_info",)).fetchone()[0]
        if not exists:
            return None
        row = conn.execute(f"SELECT value FROM {schema}.build_info WHERE key = 'fingerprint'").fetchone()
        return row[0] if row else None


def load(database_url: str, schema: str, data_dir: Path, fingerprint: dict,
         tables: bool = True, spaces: tuple[str, ...] | None = None) -> None:
    """Load tables (atomically) and/or vector spaces, then record the fingerprint."""
    with psycopg.connect(database_url, autocommit=True) as conn:
        if tables:
            log.info("Loading tables into schema %s ...", schema)
            with conn.transaction():          # readers see the old schema until the new one is complete
                load_tables(conn, schema, data_dir)
        conn.execute(f"SET search_path TO {schema}, public")
        for space in spaces if spaces is not None else VECTOR_SPACES:
            path = data_dir / "embeddings" / f"{space}.npy"
            if path.exists():
                secs = load_vectors(conn, space, np.load(path))
                log.info("  vec_%s loaded, HNSW built in %.1fs", space, secs)
        conn.execute("CREATE TABLE IF NOT EXISTS build_info (key text PRIMARY KEY, value jsonb NOT NULL)")
        conn.execute("INSERT INTO build_info VALUES ('fingerprint', %s) ON CONFLICT (key) DO UPDATE "
                      "SET value = EXCLUDED.value", (Jsonb(fingerprint),))


def main() -> None:
    from ..build import data_fingerprint
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors-only", action="store_true")
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = load_settings()
    load(s.database_url, s.db_schema, args.data, data_fingerprint(args.data), tables=not args.vectors_only)


if __name__ == "__main__":
    main()
