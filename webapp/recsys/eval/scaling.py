"""Scaling projection: what does this pipeline cost at 10x / 100x the catalogue?

    python -m recsys.eval.scaling          # measure unit costs, project, write artifacts/scaling.json

Managers ask "this demo runs on 8.9k dishes and 6k users — what happens at real scale?" Rather
than guess, we measure the cost of each stage at today's size and project it along each stage's
known complexity, stating the assumption for every line so the projection is auditable, not magic.

Stages and how they scale:
  * embedding    : one vector per dish            -> O(items)      (and a one-off $ for the API)
  * retrieval fit: ALS/EASE factorise user x item -> ~O(orders)    (orders grow with users)
  * ranker train : rows = requests x candidates   -> O(train_rows) (linear in training requests)
  * serving/req  : retrieve + ~35 feats x ~100 cand + score
                   ANN is sublinear in items (HNSW), feature build is O(candidates), not items
  * vector index : HNSW build is ~O(items log items); storage is O(items x dim x 4 bytes)

The projection holds the per-user order rate and the candidate pool fixed (both realistic), and
scales items and users independently. Absolute numbers are this machine's; the shape is general.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from ..config import ARTIFACT_DIR, DATA_DIR, load_settings
from ..data import Dataset

log = logging.getLogger("recsys.scaling")

# Scenarios to project to (items, users). The first row is measured; the rest are projected.
SCENARIOS = [
    ("demo (measured)", 1, 1),
    ("city launch", 3, 5),
    ("national", 12, 30),
    ("marketplace", 60, 100),
]


def _measure(data: Dataset) -> dict:
    """Measure the quantities the projection scales from, at today's size."""
    n_items = int(data.n_items)
    n_users = int(data.n_users)
    n_orders = int(len(data.o_user))
    dim = 768  # gemini/bge embedding width (the stored serving vectors)

    # Serving latency per request: retrieve + feature-build + score on a fixed candidate pool.
    # We read it from the ranking artifact if present (measured there), else mark unknown.
    art = ARTIFACT_DIR / "ranking.json"
    serve_ms = None
    train_s = None
    train_rows = None
    if art.exists():
        r = json.loads(art.read_text(encoding="utf-8"))
        live = next((p for p in r["pipelines"] if p["ranker"] == "listwise"), r["pipelines"][0])
        serve_ms = live["latency_ms_per_request"]
        train_s = live["train_seconds"]
        train_rows = r["n_train_rows"]

    # Vector storage: items x dim x 4 bytes (float32), per embedding space.
    bytes_per_space = n_items * dim * 4
    return {"n_items": n_items, "n_users": n_users, "n_orders": n_orders, "dim": dim,
            "serve_ms": serve_ms, "train_seconds": train_s, "train_rows": train_rows,
            "vector_mb_per_space": round(bytes_per_space / 1e6, 1)}


def _project(base: dict) -> list[dict]:
    rows = []
    for name, item_mult, user_mult in SCENARIOS:
        items = int(base["n_items"] * item_mult)
        users = int(base["n_users"] * user_mult)
        # orders scale with users (same per-user rate); ranker rows scale with training requests,
        # which scale with users; candidate pool and feature count are held fixed.
        orders = int(base["n_orders"] * user_mult)
        train_rows = int(base["train_rows"] * user_mult) if base["train_rows"] else None
        # ranker training ~ linear in rows.
        train_s = round(base["train_seconds"] * user_mult, 1) if base["train_seconds"] else None
        # serving latency: feature build is O(candidates) (fixed); ANN retrieval over items is
        # ~O(log items) with HNSW. So per-request latency grows very slowly — model it as
        # the fixed part plus a log(items) term on a small ANN share (~30% of the request).
        serve = None
        if base["serve_ms"]:
            ann_share = 0.30
            fixed = base["serve_ms"] * (1 - ann_share)
            ann = base["serve_ms"] * ann_share * (np.log2(max(items, 2)) / np.log2(max(base["n_items"], 2)))
            serve = round(fixed + ann, 1)
        rows.append({
            "scenario": name, "item_mult": item_mult, "user_mult": user_mult,
            "items": items, "users": users, "orders": orders,
            "train_rows": train_rows, "train_seconds": train_s,
            "serve_ms_per_request": serve,
            "vector_mb_per_space": round(base["vector_mb_per_space"] * item_mult, 1),
            "embedding_one_off_usd": round(items / 1000 * 0.00013 * 1000 / 1000, 4),  # ~$0.00013/1k tok, ~1 vec/dish
        })
    return rows


def run(out: Path) -> dict:
    data = Dataset(DATA_DIR)
    base = _measure(data)
    rows = _project(base)
    notes = [
        "Serving latency per request grows only with log(items): HNSW retrieval is sublinear and "
        "feature-building runs over a fixed ~100-candidate pool, not the whole catalogue.",
        "Ranker training time and training rows scale linearly with the number of users (more "
        "requests), not with catalogue size.",
        "Vector storage is items x dim x 4 bytes per embedding space; three spaces are stored.",
        "The ranker is retrained offline on a schedule (hours), so its cost is a batch job, not a "
        "per-request cost — it never sits on the serving latency.",
        "Absolute numbers are this machine (single CPU); the growth shape is hardware-independent.",
    ]
    result = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "measured": base,
              "scenarios": rows, "notes": notes}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Wrote %s", out)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR / "scaling.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(args.out)


if __name__ == "__main__":
    main()
