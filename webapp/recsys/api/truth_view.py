"""Read-only view of the simulation's hidden ground truth for the UI.

Shown on the user page, clearly labelled, so people can judge recommendations against what
the simulated user really likes. Recommendation models never read these files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import json

from ..synth.vocab import CUISINES, DAYPARTS
from ..synth.world import WorldConfig


class TruthView:
    def __init__(self, data_dir: Path) -> None:
        path = data_dir / "truth" / "users.npz"
        self._u = dict(np.load(path)) if path.exists() else None
        trips = data_dir / "truth" / "trips.npz"
        self._t = dict(np.load(trips)) if trips.exists() else None
        # The simulator scales the stored base rate (see simulate.SimContext.regular_rate_scale);
        # read the factor from the world that was actually generated.
        meta = data_dir / "meta.json"
        per_week = (json.loads(meta.read_text(encoding="utf-8"))["config"]["regular_sessions_per_week"]
                    if meta.exists() else WorldConfig().regular_sessions_per_week)
        self._rate_scale = per_week / 1.8

    @property
    def available(self) -> bool:
        return self._u is not None

    def user(self, uid: int) -> dict[str, Any] | None:
        if self._u is None or not 0 <= uid < len(self._u["segment"]):
            return None
        u = self._u
        aff = u["log_aff"][uid]
        top = np.argsort(-aff)[:5]
        share = np.exp(aff - aff.max())
        share = share / share.sum()
        ps = u["price_sens"]
        trips = []
        if self._t is not None:
            m = self._t["user_id"] == uid
            trips = [{"start": str(s), "days": int(d)} for s, d in zip(self._t["start"][m], self._t["days"][m])]
        return {
            "segment": str(u["segment"][uid]),
            "top_cuisines": [{"cuisine": CUISINES[k].key, "label": CUISINES[k].label,
                              "share": round(float(share[k]), 3)} for k in top],
            "vegetarian": bool(u["vegetarian"][uid]),
            "avoids_spicy": bool(u["spicy_tol"][uid] < -0.5),
            "price_sensitivity_pct": int(round(100 * float((ps < ps[uid]).mean()))),
            "habit_pct": int(round(100 * float((u["habit"] < u["habit"][uid]).mean()))),
            "reference_price_eur": round(float(u["ref_price"][uid]), 2),
            "sessions_per_week": round(float(u["weekly_rate"][uid]) * self._rate_scale, 2),
            "meal_times": {DAYPARTS[i]: round(float(w), 3) for i, w in enumerate(u["daypart_w"][uid])},
            "trips": trips,
        }
