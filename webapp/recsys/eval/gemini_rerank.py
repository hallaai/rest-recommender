"""Gemini LLM reranker: does a language model improve the top of the list, and at what cost?

    python -m recsys.eval.gemini_rerank --requests 60   # eval, write artifacts/gemini_rerank.json

The LightGBM ranker scores (user, dish) pairs on ~35 numeric features. It cannot read a dish's
*text* — that "a rainy cold Tuesday night suits a warm ramen over a cold poke bowl" is the kind
of commonsense an LLM carries. This experiment asks Gemini to rerank the pipeline's top-N
candidates, given the request context and each dish's text, and measures whether the reorder
captures more true demand than the LightGBM order — against its latency and dollar cost.

Honesty and cost control:
  * Compared head-to-head on the SAME candidate set the pipeline produced, so this isolates the
    rerank step. Reward is the ground-truth oracle's captured demand.
  * Runs on a small request sample with a cheap flash-lite model; every call goes through the
    spend ledger, and the run stops cleanly if the budget is exhausted (reporting what it had).
  * The likely finding for managers: a measurable but small lift at a latency and $/request that
    only makes sense for the very top slots, if at all. The lab quantifies the trade-off.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from ..config import ARTIFACT_DIR, DATA_DIR, load_settings
from ..data import Dataset, Request, TRAIN_END_DAY, TEST_END_DAY, VAL_END_DAY
from ..eval.base import FitContext
from ..eval.features import FeatureBuilder
from ..eval.metrics import K
from ..eval.oracle import TruthOracle
from ..eval.rank_runner import apply_rules, make_retrievers, pool_candidates
from ..eval.ranker import LGBMRanker, build_training_data
from ..llm import BudgetExceeded, Gemini, LLMError
from ..synth.vocab import CUISINES, DAYPARTS

log = logging.getLogger("recsys.gemini_rerank")
RERANK_N = 10            # how many top candidates to hand the LLM


def _dish_line(data: Dataset, item: int) -> str:
    name = str(data.items_df["name"].to_numpy()[item])
    cuisine = CUISINES[int(data.item_cuisine[item])].label
    price = float(data.item_price[item])
    veg = "veg " if data.item_veg[item] else ""
    spicy = "spicy " if data.item_spicy[item] else ""
    return f"{name} — {veg}{spicy}{cuisine}, €{price:.1f}"


def _prompt(data: Dataset, req: Request, items: np.ndarray) -> str:
    ctx = (f"A food-delivery customer is ordering now. Context: {DAYPARTS[req.daypart]} on "
           f"{'a weekend' if req.dow >= 5 else 'a weekday'}, "
           f"{'rainy' if req.rain else 'clear'}{' and hot' if req.hot else ''}, "
           f"in {'Madrid' if req.city == 1 else 'Barcelona'}.")
    lines = "\n".join(f"{i}. {_dish_line(data, int(it))}" for i, it in enumerate(items))
    return (f"{ctx}\nHere are {len(items)} candidate dishes:\n{lines}\n\n"
            f"Rank them from most to least likely to be ordered in this exact context, using "
            f"common sense about weather, time of day and the dishes themselves. "
            f'Reply as JSON: {{"order": [list of the {len(items)} indices, best first]}}.')


def _rerank_order(reply: object, n: int) -> list[int] | None:
    """Validate the LLM's order: a permutation of 0..n-1. None if malformed."""
    if not isinstance(reply, dict) or "order" not in reply:
        return None
    order = reply["order"]
    if not isinstance(order, list):
        return None
    seen = [int(x) for x in order if isinstance(x, (int, float)) and 0 <= int(x) < n]
    # keep first occurrence, then append any missing indices so it stays a full permutation
    uniq = list(dict.fromkeys(seen))
    uniq += [i for i in range(n) if i not in uniq]
    return uniq[:n] if len(uniq) == n else None


def run(n_requests: int, out: Path, seed: int = 0) -> dict:
    s = load_settings()
    data = Dataset(DATA_DIR)
    oracle = TruthOracle(data, DATA_DIR)
    fb = FeatureBuilder(data)
    gem = Gemini(s)

    log.info("Fitting retrievers + ranker (same as the pipeline) ...")
    retrievers = make_retrievers()
    for rec in retrievers.values():
        rec.fit(FitContext(data=data, cutoff_day=TRAIN_END_DAY))
    fb.set_popularity(TRAIN_END_DAY)
    tr_reqs = data.requests(TRAIN_END_DAY, VAL_END_DAY, sample=4000, seed=seed)
    tr_cands = [data.available_mains(r.lat, r.lon, r.hour) for r in tr_reqs]
    keep = [i for i, c in enumerate(tr_cands) if len(c) >= K and tr_reqs[i].target in set(c.tolist())]
    tr_reqs, tr_cands = [tr_reqs[i] for i in keep], [tr_cands[i] for i in keep]
    ranker = LGBMRanker(loss="listwise")
    ranker.fit(build_training_data(data, retrievers, fb, tr_reqs, tr_cands))

    reqs = data.requests(VAL_END_DAY, TEST_END_DAY, sample=n_requests * 3, seed=seed)
    reqs = [r for r in reqs if len(data.available_mains(r.lat, r.lon, r.hour)) >= RERANK_N][:n_requests]
    log.info("Reranking %d requests with Gemini (%s) ...", len(reqs), s.model_text)

    d_lgbm, d_gem, lat_ms = [], [], []
    n_llm_ok = n_llm_bad = 0
    stopped = None
    for req in reqs:
        cand = data.available_mains(req.lat, req.lon, req.hour)
        retr = {name: rec.recommend(req, cand) for name, rec in retrievers.items()}
        pooled = pool_candidates(retr, cand, 100)
        pc = cand[pooled]
        feat_in = {name: retr[name][pooled] for name in retr}
        feat_in["content"] = feat_in.pop("content_gemini", np.zeros(len(pc)))
        scores = ranker.score(fb.row(req, pc, feat_in))
        top_idx = np.argsort(-scores, kind="stable")[:RERANK_N]
        top_items = pc[top_idx]

        probs = oracle.choice_probs(req, cand)
        prob_by_item = {int(c): float(p) for c, p in zip(cand, probs)}

        # LightGBM order: apply business rules to the ranker order, take top K.
        lgbm_pick = apply_rules(data, pc, np.argsort(-scores, kind="stable"), K)
        d_lgbm.append(float(sum(prob_by_item.get(int(i), 0.0) for i in pc[lgbm_pick])))

        # Gemini rerank of the top-N, then business rules on its order.
        try:
            t0 = time.perf_counter()
            reply = gem.json(_prompt(data, req, top_items), purpose="rerank",
                             max_tokens=256, temperature=0.2)
            lat_ms.append((time.perf_counter() - t0) * 1000)
        except BudgetExceeded as e:
            stopped = f"budget exhausted after {len(d_gem)} requests: {e}"
            log.warning(stopped)
            break
        except LLMError as e:
            log.warning("rerank call failed: %s", e)
            n_llm_bad += 1
            d_gem.append(d_lgbm[-1])          # fall back to the LightGBM order for this request
            continue
        order = _rerank_order(reply, len(top_items))
        if order is None:
            n_llm_bad += 1
            d_gem.append(d_lgbm[-1])
            continue
        n_llm_ok += 1
        reordered = top_items[order]
        gem_pick = apply_rules(data, reordered, np.arange(len(reordered)), K)
        d_gem.append(float(sum(prob_by_item.get(int(i), 0.0) for i in reordered[gem_pick])))

    n = len(d_gem)
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": s.model_text,
        "n_requests": n, "rerank_n": RERANK_N, "llm_ok": n_llm_ok, "llm_malformed": n_llm_bad,
        "stopped": stopped,
        "lgbm_demand": round(float(np.mean(d_lgbm[:n])), 4) if n else None,
        "gemini_demand": round(float(np.mean(d_gem)), 4) if n else None,
        "lift_pct": round((np.mean(d_gem) / np.mean(d_lgbm[:n]) - 1) * 100, 1) if n and np.mean(d_lgbm[:n]) else None,
        "median_llm_latency_ms": round(float(np.median(lat_ms)), 1) if lat_ms else None,
        "spend_usd": round(sum(r.usd for r in gem.ledger.records() if r.purpose == "rerank"), 4),
        "notes": [
            "Same candidate set as the LightGBM pipeline, so this isolates the rerank step.",
            "Reward is the ground-truth oracle's captured demand; both policies scored identically.",
            "Cost and latency are per request; an LLM in the serving path is far slower and paid, so "
            "it only makes sense (if at all) for the top few slots, computed asynchronously.",
            "A malformed or budget-blocked LLM reply falls back to the LightGBM order, never an error.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Wrote %s  (gemini %.4f vs lgbm %.4f, lift %s%%, $%.4f)", out,
             result["gemini_demand"] or 0, result["lgbm_demand"] or 0,
             result["lift_pct"], result["spend_usd"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=60)
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR / "gemini_rerank.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(args.requests, args.out)


if __name__ == "__main__":
    main()
