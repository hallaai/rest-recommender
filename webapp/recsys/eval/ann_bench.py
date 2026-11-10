"""ANN benchmark: does approximate vector search cost us quality, and how much faster is it?

    python -m recsys.eval.ann_bench            # benchmark gemini space, write artifacts/ann.json

Answers a concrete question managers ask about pgvector: "if we use an approximate index instead
of scanning every dish, how many of the true nearest dishes do we miss, and how much latency do
we save?" We take real query vectors (user taste vectors over the catalogue), compute the exact
top-k by brute force in numpy (ground truth), then query three strategies in Postgres —

  * exact        : ORDER BY distance with no index (sequential scan) — the correctness reference
  * hnsw         : the shipped HNSW index at several ef_search settings (recall/latency trade-off)
  * ivfflat      : an IVFFlat index at several probe settings — the other common pgvector index

and report recall@k (fraction of the true top-k found) and median/p95 latency for each. Recall is
measured against exact *numpy* truth, so even the DB "exact" row is checked, not assumed.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from ..config import ARTIFACT_DIR, DATA_DIR, load_settings
from ..data import Dataset, TRAIN_END_DAY, VAL_END_DAY
from ..embed import load
from ..store.repo import Repo
from .content import Content

log = logging.getLogger("recsys.ann")
K = 20                       # neighbours per query
HNSW_EF = [20, 40, 80, 160]  # ef_search sweep
IVF_PROBES = [1, 4, 10, 20]  # probes sweep


def _query_vectors(data: Dataset, backend: str, n: int, seed: int = 0) -> np.ndarray:
    """Real query vectors: user taste vectors (time-weighted avg of ordered-dish embeddings)."""
    from .base import FitContext
    content = Content(backend)
    content.fit(FitContext(data=data, cutoff_day=VAL_END_DAY))
    reqs = data.requests(TRAIN_END_DAY, VAL_END_DAY, sample=n * 3, seed=seed)
    vecs = []
    for r in reqs:
        v = content.user_vector(r)
        if v is not None:
            vecs.append(v)
        if len(vecs) >= n:
            break
    return np.asarray(vecs, np.float32)


def _exact_topk(item_vecs: np.ndarray, item_ids: np.ndarray, q: np.ndarray, k: int) -> set[int]:
    """Brute-force cosine top-k in numpy — the ground truth recall is measured against."""
    sims = item_vecs @ q
    idx = np.argpartition(-sims, k)[:k]
    return set(int(item_ids[i]) for i in idx)


def _time_queries(fn, queries, repeats: int = 1) -> tuple[list[set[int]], float, float]:
    """Run fn(q) over queries (repeated for stable timing); return results, median ms, p95 ms."""
    latencies = []
    results = []
    for rep in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            res = fn(q)
            latencies.append((time.perf_counter() - t0) * 1000)
            if rep == 0:
                results.append(res)
    lat = np.array(latencies)
    return results, float(np.median(lat)), float(np.percentile(lat, 95))


def _recall(approx: list[set[int]], truth: list[set[int]], k: int) -> float:
    return float(np.mean([len(a & t) / k for a, t in zip(approx, truth)]))


def _ensure_ivfflat(repo: Repo, space: str, dim: int) -> None:
    """Create an IVFFlat index for this space if absent (lists ~ sqrt(n_rows))."""
    with repo._pool.connection() as c:
        n = c.execute(f"SELECT count(*) n FROM vec_{space}").fetchone()["n"]
        lists = max(int(n ** 0.5), 1)
        exists = c.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname=%s AND indexname=%s",
            (repo._schema, f"vec_{space}_ivf")).fetchone()
        if not exists:
            log.info("Building IVFFlat index (lists=%d) for %s ...", lists, space)
            c.execute(f"CREATE INDEX vec_{space}_ivf ON vec_{space} "
                      f"USING ivfflat (v vector_cosine_ops) WITH (lists = {lists})")
            c.execute("COMMIT")


def run(backend: str, n_queries: int, out: Path) -> dict:
    s = load_settings()
    data = Dataset(DATA_DIR)
    repo = Repo(s.database_url, s.db_schema)

    item_vecs = load(backend, data.dir).astype(np.float32)
    item_vecs /= np.maximum(np.linalg.norm(item_vecs, axis=1, keepdims=True), 1e-9)
    item_ids = data.items_df["item_id"].to_numpy()
    dim = item_vecs.shape[1]

    log.info("Building %d query vectors (%s, dim=%d) ...", n_queries, backend, dim)
    queries = _query_vectors(data, backend, n_queries)
    truth = [_exact_topk(item_vecs, item_ids, q, K) for q in queries]
    log.info("  %d query vectors, exact top-%d computed", len(queries), K)

    _ensure_ivfflat(repo, backend, dim)

    def exact_db(q):
        return set(i for i, _ in repo.ann_search(backend, q, K, strategy="exact"))

    strategies = []
    # exact (sequential scan) — recall should be 1.0, this validates the harness
    res, med, p95 = _time_queries(exact_db, queries)
    strategies.append({"name": "exact (scan)", "param": None, "recall": _recall(res, truth, K),
                       "median_ms": round(med, 2), "p95_ms": round(p95, 2)})
    log.info("  exact scan: recall %.3f  median %.2fms", strategies[-1]["recall"], med)

    for ef in HNSW_EF:
        fn = lambda q, ef=ef: set(i for i, _ in repo.ann_search(backend, q, K, strategy="hnsw", ef_search=ef))
        res, med, p95 = _time_queries(fn, queries)
        strategies.append({"name": "HNSW", "param": f"ef_search={ef}", "recall": _recall(res, truth, K),
                           "median_ms": round(med, 2), "p95_ms": round(p95, 2)})
        log.info("  HNSW ef=%d: recall %.3f  median %.2fms", ef, strategies[-1]["recall"], med)

    for probes in IVF_PROBES:
        fn = lambda q, p=probes: set(i for i, _ in repo.ann_search(backend, q, K, strategy="ivfflat", probes=p))
        res, med, p95 = _time_queries(fn, queries)
        strategies.append({"name": "IVFFlat", "param": f"probes={probes}", "recall": _recall(res, truth, K),
                           "median_ms": round(med, 2), "p95_ms": round(p95, 2)})
        log.info("  IVFFlat probes=%d: recall %.3f  median %.2fms", probes, strategies[-1]["recall"], med)

    repo._pool.close()
    result = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": backend,
              "dim": dim, "n_items": len(item_ids), "n_queries": len(queries), "k": K,
              "strategies": strategies}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Wrote %s", out)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gemini", choices=["tfidf", "bge", "gemini"])
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR / "ann.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(args.backend, args.queries, args.out)


if __name__ == "__main__":
    main()
