"""Cold-start bandit: can online exploration beat the static pipeline for no-history users?

    python -m recsys.eval.bandit        # simulate, write artifacts/bandit.json

The static pipeline treats a cold user with a fixed policy: popular-nearby dishes, ranked once.
It never learns from how that user responds. A bandit does: it treats each *cuisine* as an arm,
keeps a Beta posterior over "does this user's segment convert on this cuisine here," and uses
Thompson sampling to balance trying new cuisines against exploiting what has worked. As rounds
accumulate for a segment, it should converge on the cuisines that segment actually likes —
discovering preference the cold user never declared.

Honesty of the simulation:
  * Reward is the ground-truth oracle's demand (true order probability) for the dishes shown —
    the same signal the leaderboard scores, not a hand-set number.
  * Segments are the user profile the business genuinely has at signup: (home_city, district,
    age_band). Users within a segment share a posterior, so a new user benefits from what the
    bandit already learned about similar new users — this is what makes cold start tractable.
  * The static baseline is the pipeline's own cold-start policy (geo+time popularity), scored
    on the identical requests, so the comparison is apples to apples.

This is a demonstration of the *mechanism and its trade-off*, not a production policy. The point
for managers: exploration earns its keep only if the cold segment is large and revisited.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from ..config import ARTIFACT_DIR, DATA_DIR, load_settings
from ..data import Dataset, TRAIN_END_DAY, TEST_END_DAY, VAL_END_DAY
from ..eval.base import FitContext
from ..eval.baselines import GeoTimePopularity
from ..eval.metrics import K
from ..eval.oracle import TruthOracle
from ..synth.vocab import CUISINES

log = logging.getLogger("recsys.bandit")
N_CUISINES = len(CUISINES)


def _segment(data: Dataset, user: int, granularity: str = "coarse") -> tuple:
    """The signup-time profile a cold user genuinely has.

    'coarse' = (home_city, age_band): few segments, each revisited often, so posteriors learn fast
    — the realistic cold-start grouping. 'fine' = + district: more targeted but data-starved here.
    """
    if granularity == "fine":
        return (int(data.user_home_city[user]), int(data.user_district[user]), int(data.user_age[user]))
    return (int(data.user_home_city[user]), int(data.user_age[user]))


class ThompsonCuisineBandit:
    """One Beta(alpha, beta) posterior per (segment, cuisine). Thompson sampling picks a cuisine."""

    def __init__(self) -> None:
        self.alpha: dict[tuple, np.ndarray] = {}
        self.beta: dict[tuple, np.ndarray] = {}

    def _post(self, seg: tuple) -> tuple[np.ndarray, np.ndarray]:
        if seg not in self.alpha:
            self.alpha[seg] = np.ones(N_CUISINES)
            self.beta[seg] = np.ones(N_CUISINES)
        return self.alpha[seg], self.beta[seg]

    def choose(self, seg: tuple, available_cuisines: np.ndarray, rng: np.random.Generator) -> int:
        a, b = self._post(seg)
        sample = rng.beta(a, b)
        mask = np.full(N_CUISINES, -1.0)
        mask[available_cuisines] = sample[available_cuisines]
        return int(np.argmax(mask))

    def update(self, seg: tuple, cuisine: int, reward: float) -> None:
        """Reward in [0,1] (demand captured). Beta update with a continuous reward."""
        a, b = self._post(seg)
        a[cuisine] += reward
        b[cuisine] += 1.0 - reward


def _top_by_rules(data: Dataset, cand: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    """Top-k candidates by score, ≤2 per restaurant."""
    order = cand[np.argsort(-score, kind="stable")]
    per_rest: dict[int, int] = {}
    out = []
    for it in order:
        r = int(data.item_rest[it])
        if per_rest.get(r, 0) >= 2:
            continue
        per_rest[r] = per_rest.get(r, 0) + 1
        out.append(int(it))
        if len(out) >= k:
            break
    return np.array(out, int)


def run(n_requests: int, out: Path, seed: int = 0, granularity: str = "coarse") -> dict:
    data = Dataset(DATA_DIR)
    oracle = TruthOracle(data, DATA_DIR)
    rng = np.random.default_rng(seed)

    # Static cold-start baseline = the pipeline's cold policy, fit on train+val like serving.
    static = GeoTimePopularity()
    static.fit(FitContext(data=data, cutoff_day=VAL_END_DAY))

    # Cold users only: no order history before the test window. Sample broadly, then cap, so we
    # have enough repeat visits per segment for the posteriors to actually learn.
    reqs = data.requests(VAL_END_DAY, TEST_END_DAY, sample=n_requests * 6, seed=seed)
    cold = [r for r in reqs if data.history(r.user, r.day).n == 0]
    rng.shuffle(cold)
    cold = cold[:n_requests]
    log.info("Simulating %d cold-start requests ...", len(cold))

    bandit = ThompsonCuisineBandit()
    # Track cumulative mean demand over rounds for both policies (the learning curve).
    curve_bandit, curve_static = [], []
    sum_b = sum_s = 0.0
    # The bandit boosts its chosen cuisine on top of the static score, so it can only re-tilt a
    # strong prior, never discard it. CUISINE_BOOST is how much a chosen-cuisine dish is favoured.
    CUISINE_BOOST = 2.0
    for t, req in enumerate(cold, 1):
        cand = data.available_mains(req.lat, req.lon, req.hour)
        if len(cand) < K:
            continue
        probs = oracle.choice_probs(req, cand)         # true order prob per candidate
        prob_by_item = {int(c): float(p) for c, p in zip(cand, probs)}

        # Static policy: geo+time popularity, ranked once. The cold-start prior.
        s_scores = np.asarray(static.recommend(req, cand), float)
        s_top = _top_by_rules(data, cand, s_scores, K)
        d_static = float(sum(prob_by_item.get(int(i), 0.0) for i in s_top))

        # Bandit: pick a cuisine arm for this segment, then re-rank the SAME static scores with a
        # multiplicative boost on that cuisine. Reward = demand captured; update the arm's posterior.
        seg = _segment(data, req.user, granularity)
        avail_cuisines = np.unique(data.item_cuisine[cand])
        arm = bandit.choose(seg, avail_cuisines, rng)
        boost = np.where(data.item_cuisine[cand] == arm, CUISINE_BOOST, 1.0)
        b_scores = s_scores * boost
        b_top = _top_by_rules(data, cand, b_scores, K)
        d_bandit = float(sum(prob_by_item.get(int(i), 0.0) for i in b_top))
        bandit.update(seg, arm, min(max(d_bandit, 0.0), 1.0))

        sum_b += d_bandit
        sum_s += d_static
        curve_bandit.append(sum_b / t)
        curve_static.append(sum_s / t)

    # Downsample the curve for the chart (<=60 points).
    n = len(curve_bandit)
    step = max(n // 60, 1)
    idx = list(range(0, n, step))
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_requests": n, "n_cuisines": N_CUISINES, "segment_granularity": granularity,
        "n_segments": len(bandit.alpha),
        "final_bandit_demand": round(curve_bandit[-1], 4) if n else None,
        "final_static_demand": round(curve_static[-1], 4) if n else None,
        "curve": [{"round": i + 1, "bandit": round(curve_bandit[i], 4),
                   "static": round(curve_static[i], 4)} for i in idx],
        "notes": [
            "Reward is the ground-truth oracle's captured demand, the same signal the leaderboard scores.",
            "Arms are cuisines; posteriors are shared within a (city, district, age) segment, so a new "
            "cold user inherits what the bandit learned about similar new users.",
            "The static line is the pipeline's own cold-start policy (geo+time popularity) on the same "
            "requests. The bandit must explore first, so it can start below static and cross over as it learns.",
            "This demonstrates the exploration/exploitation trade-off; it is not a drop-in production policy.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Wrote %s  (bandit %.3f vs static %.3f over %d rounds)",
             out, result["final_bandit_demand"] or 0, result["final_static_demand"] or 0, n)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=4000)
    ap.add_argument("--granularity", default="coarse", choices=["coarse", "fine"])
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR / "bandit.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(args.requests, args.out, granularity=args.granularity)


if __name__ == "__main__":
    main()
