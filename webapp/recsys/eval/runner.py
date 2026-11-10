"""Run every recommender over the test month, measure quality, behaviour and cost.

    python -m recsys.eval.runner                 # all baselines, full October test set
    python -m recsys.eval.runner --sample 3000   # faster: a user-stratified sample of requests

Writes artifacts/leaderboard.json: for each algorithm, the metrics with confidence intervals,
the cost (fit time, memory, latency p50/p95), and per-segment breakdowns; plus pairwise
overlap and paired differences. The Compare and Costs pages read this file.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..config import ARTIFACT_DIR, DATA_DIR, load_settings
from ..data import Dataset, Request, TEST_END_DAY, VAL_END_DAY
from .base import FitContext, Recommender
from .baselines import ALL_BASELINES
from .metrics import (CAND_K, K, RequestEval, bootstrap_ci, catalog_coverage, gini, mean_overlap,
                      ndcg_at_k, paired_diff_ci, pct_of_best, revenue_per_1000)
from .oracle import TruthOracle

log = logging.getLogger("recsys.eval")
POINT_FIELDS = ("hit", "ndcg", "rr", "recall100", "rest_hit", "demand", "is_reorder",
                "n_rest_top5", "n_cuisine_top5", "pop_pctile")


def segment_of(data: Dataset, req: Request) -> dict[str, str]:
    n = data.history(req.user, req.day).n
    hist = "none" if n == 0 else "1-3" if n <= 3 else "4-15" if n <= 15 else "16+"
    return {"history": hist, "city": ("BCN", "MAD")[req.city],
            "traveller": "yes" if req.city != req.home_city else "no",
            "daypart": ("breakfast", "lunch", "afternoon", "dinner", "late")[req.daypart]}


def _eval_one(data: Dataset, oracle: TruthOracle, pop_pctile: np.ndarray, req: Request,
              cand: np.ndarray, order: np.ndarray) -> RequestEval:
    """Build the metric record for one request given the candidate order (best first).

    `order` indexes into `cand`; `probs` is aligned with `cand`, so top-K probabilities are
    just probs[order[:K]] — no per-dish dict lookups."""
    probs = oracle.choice_probs(req, cand)                   # aligned with cand
    topk_idx = order[:K]
    topk = cand[topk_idx]
    price = data.item_price
    demand = float(probs[topk_idx].sum())
    best_probs = np.sort(probs)[::-1]
    best_demand = float(best_probs[:K].sum())
    revenue = float((probs[topk_idx] * price[topk]).sum())
    best_order_idx = np.argsort(-probs)[:K]
    best_revenue = float((probs[best_order_idx] * price[cand[best_order_idx]]).sum())
    # rank of the ordered dish in the full ranked list (1-based), 0 if absent
    target_pos = np.flatnonzero(cand[order] == req.target)
    r = int(target_pos[0]) + 1 if len(target_pos) else 0
    hist_items = data.history(req.user, req.day).items
    rests = data.item_rest[topk]
    return RequestEval(
        user=req.user,
        hit=float(req.target in topk),
        ndcg=ndcg_at_k(r),
        rr=1.0 / r if r else 0.0,
        recall100=float(1 <= r <= CAND_K),
        rest_hit=float(req.target_rest in rests),
        demand=demand, best_demand=best_demand, revenue=revenue, best_revenue=best_revenue,
        is_reorder=float(np.isin(topk[0], hist_items)) if len(topk) else 0.0,
        n_rest_top5=float(len(np.unique(rests))),
        n_cuisine_top5=float(len(np.unique(data.item_cuisine[topk]))),
        pop_pctile=float(np.mean(pop_pctile[topk])) if len(topk) else 0.0,
        top5=tuple(int(i) for i in topk))


def run_recommender(rec: Recommender, data: Dataset, oracle: TruthOracle, requests: list[Request],
                    cands: list[np.ndarray], pop_pctile: np.ndarray) -> dict:
    # Retrieval models fit on train + validation (everything before the first October day).
    ctx = FitContext(data=data, cutoff_day=VAL_END_DAY)
    tracemalloc.start()
    t0 = time.perf_counter()
    rec.fit(ctx)
    fit_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    evals: list[RequestEval] = []
    latencies = np.empty(len(requests))
    for i, (req, cand) in enumerate(zip(requests, cands)):
        t = time.perf_counter()
        scores = rec.recommend(req, cand)
        order = np.argsort(-scores, kind="stable")
        latencies[i] = (time.perf_counter() - t) * 1000      # ms
        evals.append(_eval_one(data, oracle, pop_pctile, req, cand, order))
    return {"evals": evals, "fit_s": fit_s, "peak_mb": peak / 1e6,
            "lat_p50_ms": float(np.percentile(latencies, 50)),
            "lat_p95_ms": float(np.percentile(latencies, 95))}


def summarise(name: str, result: dict, data: Dataset, segments: list[dict], vcpu_hour_usd: float) -> dict:
    evals = result["evals"]
    overall = {f: bootstrap_ci(evals, f) for f in POINT_FIELDS}
    overall["pct_of_best"] = {"mean": pct_of_best(evals)}
    overall["revenue_per_1000"] = {"mean": revenue_per_1000(evals)}
    # behaviour
    exposure = np.bincount(np.concatenate([np.array(e.top5) for e in evals]) if evals else np.zeros(0, int),
                           minlength=data.n_items)
    rest_exposure = np.bincount(data.item_rest[np.flatnonzero(exposure > 0)],
                                weights=exposure[exposure > 0], minlength=data.n_rest) if exposure.sum() else np.zeros(data.n_rest)
    behaviour = {"catalog_coverage": catalog_coverage(evals, len(data.mains)),
                 "restaurant_gini": gini(rest_exposure)}
    # segments
    seg_out: dict[str, dict] = {}
    for key in ("history", "traveller", "city", "daypart"):
        buckets: dict[str, list[RequestEval]] = {}
        for e, s in zip(evals, segments):
            buckets.setdefault(s[key], []).append(e)
        seg_out[key] = {v: {"n": len(b), "demand": bootstrap_ci(b, "demand"),
                            "hit": bootstrap_ci(b, "hit"), "pct_of_best": pct_of_best(b)}
                        for v, b in sorted(buckets.items())}
    cost = {"fit_s": result["fit_s"], "peak_mb": result["peak_mb"],
            "lat_p50_ms": result["lat_p50_ms"], "lat_p95_ms": result["lat_p95_ms"],
            "fit_usd": result["fit_s"] / 3600 * vcpu_hour_usd,
            "usd_per_million_recs": result["lat_p50_ms"] / 1000 / 3600 * vcpu_hour_usd * 1e6}
    return {"name": name, "metrics": overall, "behaviour": behaviour, "segments": seg_out, "cost": cost}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, help="evaluate on a user-stratified sample of requests")
    ap.add_argument("--only", nargs="+", help="only these model names (default: all)")
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR / "leaderboard.json")
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    s = load_settings()
    data = Dataset(args.data)
    oracle = TruthOracle(data, args.data)
    log.info("Loading test requests ...")
    requests = data.requests(VAL_END_DAY, TEST_END_DAY, sample=args.sample)
    cands = [data.available_mains(r.lat, r.lon, r.hour) for r in requests]
    keep = [i for i, c in enumerate(cands) if len(c) >= K]
    requests = [requests[i] for i in keep]
    cands = [cands[i] for i in keep]
    segments = [segment_of(data, r) for r in requests]
    pop_rows = data.order_rows_before(VAL_END_DAY)
    pop = np.bincount(data.o_main[pop_rows], minlength=data.n_items)
    pop_pctile = pop.argsort().argsort() / max(data.n_items - 1, 1)
    log.info("%d test requests, median %d candidates", len(requests),
             int(np.median([len(c) for c in cands])))

    from .models import retriever_factories, RETRIEVER_ORDER
    factories = retriever_factories()
    chosen = [n for n in RETRIEVER_ORDER if n in factories]
    if args.only:
        chosen = [n for n in chosen if n in args.only]
    results: dict[str, dict] = {}
    rows = []
    for name in chosen:
        rec = factories[name]()
        log.info("Fitting + scoring %s ...", rec.name)
        res = run_recommender(rec, data, oracle, requests, cands, pop_pctile)
        results[rec.name] = res
        rows.append(summarise(rec.name, res, data, segments, s.vcpu_hour_usd))
        m = rows[-1]["metrics"]
        log.info("  %-14s hit@5 %.3f  demand %.3f  %%best %.1f  fit %.1fs  p95 %.2fms", rec.name,
                 m["hit"]["mean"], m["demand"]["mean"], m["pct_of_best"]["mean"] * 100,
                 res["fit_s"], res["lat_p95_ms"])

    names = [r["name"] for r in rows]
    overlap = [[round(mean_overlap(results[a]["evals"], results[b]["evals"]), 3) for b in names] for a in names]
    best = max(rows, key=lambda r: r["metrics"]["demand"]["mean"])["name"]
    diffs = {r["name"]: paired_diff_ci(results[r["name"]]["evals"], results[best]["evals"], "demand")
             for r in rows if r["name"] != best}
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "k": K,
           "n_requests": len(requests), "sample": args.sample,
           "algorithms": rows, "overlap": {"names": names, "jaccard": overlap},
           "best_by_demand": best, "paired_vs_best": diffs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
