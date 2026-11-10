"""Stage-2 ranking evaluation and the full two-stage pipeline.

    python -m recsys.eval.rank_runner                 # train rankers, evaluate the pipeline

Pipeline: retrieval sources (fitted on May-Aug) propose candidates; they are pooled to the top
CAND_K by a simple blend; features are built; a ranker (trained on September) scores them; the
top 5 after business rules (<=2 per restaurant) is the recommendation. Compares the three
LightGBM losses and DCNv2 against the best single retriever, and writes artifacts/ranking.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from ..config import ARTIFACT_DIR, DATA_DIR, load_settings
from ..data import Dataset, TRAIN_END_DAY, VAL_END_DAY, TEST_END_DAY
from .base import FitContext
from .baselines import Popularity, GeoTimePopularity, BuyAgain, ItemItem, ALS
from .content import Content
from .features import FEATURE_NAMES, FeatureBuilder
from .metrics import CAND_K, K, RequestEval, bootstrap_ci, ndcg_at_k, pct_of_best, revenue_per_1000
from .oracle import TruthOracle
from .ranker import LGBMRanker, TrainSet, build_training_data

log = logging.getLogger("recsys.rank")
MAX_PER_RESTAURANT = 2
# Retrieval sources used as both candidate generators and ranker features.
RETRIEVER_NAMES = ["buy_again", "als", "item_item", "content_gemini", "popularity", "geo_time_pop"]


def make_retrievers():
    return {"buy_again": BuyAgain(), "als": ALS(), "item_item": ItemItem(),
            "content_gemini": Content("gemini"), "popularity": Popularity(), "geo_time_pop": GeoTimePopularity()}


def pool_candidates(retr_scores: dict[str, np.ndarray], cand: np.ndarray, k: int) -> np.ndarray:
    """Union of the top-k from each source (rank-normalised blend for ordering), capped at CAND_K."""
    blended = np.zeros(len(cand))
    for s in retr_scores.values():
        if np.any(s > 0):
            r = s.argsort().argsort() / max(len(s) - 1, 1)
            blended += r
    return np.argsort(-blended)[:k]


def apply_rules(data: Dataset, cand: np.ndarray, order: np.ndarray, k: int) -> np.ndarray:
    per_rest: dict[int, int] = {}
    out = []
    for i in order:
        r = int(data.item_rest[cand[i]])
        if per_rest.get(r, 0) >= MAX_PER_RESTAURANT:
            continue
        per_rest[r] = per_rest.get(r, 0) + 1
        out.append(int(i))
        if len(out) >= k:
            break
    return np.array(out, int)


def eval_pipeline(data, oracle, fb, requests, cands, retrievers, ranker, pop_pctile) -> list[RequestEval]:
    evals = []
    for req, cand in zip(requests, cands):
        retr = {name: rec.recommend(req, cand) for name, rec in retrievers.items()}
        pooled = pool_candidates(retr, cand, CAND_K)
        pc = cand[pooled]
        feat_in = {name: retr[name][pooled] for name in retr}
        feat_in["content"] = feat_in.pop("content_gemini", np.zeros(len(pc)))
        X = fb.row(req, pc, feat_in)
        scores = ranker.score(X)
        order = np.argsort(-scores, kind="stable")
        picks = apply_rules(data, pc, order, K)
        evals.append(_eval(data, oracle, pop_pctile, req, pc, picks))
    return evals


def _eval(data, oracle, pop_pctile, req, cand, picks) -> RequestEval:
    probs = oracle.choice_probs(req, cand)
    topk = cand[picks]
    r = int(np.flatnonzero(topk == req.target)[0]) + 1 if req.target in topk else 0
    best = np.sort(probs)[::-1]
    price = data.item_price
    hist = data.history(req.user, req.day).items
    return RequestEval(
        user=req.user, hit=float(req.target in topk), ndcg=ndcg_at_k(r), rr=1.0 / r if r else 0.0,
        recall100=float(req.target in cand), rest_hit=float(req.target_rest in data.item_rest[topk]),
        demand=float(probs[picks].sum()), best_demand=float(best[:K].sum()),
        revenue=float((probs[picks] * price[topk]).sum()), best_revenue=float((best[:K] * 0).sum()),
        is_reorder=float(np.isin(topk[0], hist)) if len(topk) else 0.0,
        n_rest_top5=float(len(np.unique(data.item_rest[topk]))),
        n_cuisine_top5=float(len(np.unique(data.item_cuisine[topk]))),
        pop_pctile=float(np.mean(pop_pctile[topk])) if len(topk) else 0.0,
        top5=tuple(int(i) for i in topk))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int)
    ap.add_argument("--train-sample", type=int, default=6000)
    ap.add_argument("--rankers", nargs="+", default=["pointwise", "pairwise", "listwise", "dcnv2"])
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR / "ranking.json")
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    s = load_settings()
    data = Dataset(args.data)
    oracle = TruthOracle(data, args.data)
    fb = FeatureBuilder(data)

    # Retrievers fit on train only (May-Aug), so September is clean for the ranker.
    log.info("Fitting retrievers on May-Aug ...")
    retrievers = make_retrievers()
    for rec in retrievers.values():
        rec.fit(FitContext(data=data, cutoff_day=TRAIN_END_DAY))
    fb.set_popularity(TRAIN_END_DAY)

    log.info("Building ranker training data on September ...")
    tr_reqs = data.requests(TRAIN_END_DAY, VAL_END_DAY, sample=args.train_sample)
    tr_cands = [data.available_mains(r.lat, r.lon, r.hour) for r in tr_reqs]
    keep = [i for i, c in enumerate(tr_cands) if len(c) >= K and tr_reqs[i].target in set(c.tolist())]
    tr_reqs = [tr_reqs[i] for i in keep]
    tr_cands = [tr_cands[i] for i in keep]
    train = build_training_data(data, retrievers, fb, tr_reqs, tr_cands)
    log.info("  %d training rows over %d requests", len(train.y), len(train.groups))

    log.info("Loading October test requests ...")
    te_reqs = data.requests(VAL_END_DAY, TEST_END_DAY, sample=args.sample)
    te_cands = [data.available_mains(r.lat, r.lon, r.hour) for r in te_reqs]
    keep = [i for i, c in enumerate(te_cands) if len(c) >= K]
    te_reqs = [te_reqs[i] for i in keep]
    te_cands = [te_cands[i] for i in keep]
    pop = np.bincount(data.o_main[data.order_rows_before(VAL_END_DAY)], minlength=data.n_items)
    pop_pctile = pop.argsort().argsort() / max(data.n_items - 1, 1)

    rows = []
    for name in args.rankers:
        log.info("Training ranker %s ...", name)
        if name == "dcnv2":
            from .dcn import DCNv2
            rk = DCNv2()
        else:
            rk = LGBMRanker(loss=name)
        rk.fit(train)
        t0 = time.perf_counter()
        evals = eval_pipeline(data, oracle, fb, te_reqs, te_cands, retrievers, rk, pop_pctile)
        lat = (time.perf_counter() - t0) / max(len(te_reqs), 1) * 1000
        row = {"name": f"pipeline_{name}", "ranker": name,
               "metrics": {f: bootstrap_ci(evals, f) for f in ("demand", "hit", "ndcg", "rr", "rest_hit")},
               "pct_of_best": pct_of_best(evals), "revenue_per_1000": revenue_per_1000(evals),
               "cold_demand": bootstrap_ci([e for e, r in zip(evals, te_reqs)
                                            if data.history(r.user, r.day).n == 0], "demand"),
               "train_seconds": rk.train_seconds, "latency_ms_per_request": round(lat, 2),
               "feature_importance": getattr(rk, "feature_importance", None)}
        rows.append(row)
        log.info("  pipeline_%-10s demand %.3f  hit %.3f  ndcg %.3f  %%best %.1f  cold %.3f  train %.1fs",
                 name, row["metrics"]["demand"]["mean"], row["metrics"]["hit"]["mean"],
                 row["metrics"]["ndcg"]["mean"], row["pct_of_best"] * 100,
                 row["cold_demand"]["mean"], rk.train_seconds)

    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_test": len(te_reqs),
           "n_train_rows": len(train.y), "features": FEATURE_NAMES, "pipelines": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
