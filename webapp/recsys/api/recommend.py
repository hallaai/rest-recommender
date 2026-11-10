"""Live recommendation endpoints for the Recommend page.

The recommenders are fitted on first use (ALS takes ~2 min), then kept in memory. While
fitting, requests get 503 + Retry-After so the UI can show a "warming up" state. Fitting runs
in a background thread so the first request returns immediately with that status.
"""
from __future__ import annotations

import logging
import threading
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request

from ..config import DATA_DIR
from ..synth.vocab import DAYPARTS

router = APIRouter(prefix="/api")
log = logging.getLogger(__name__)
_state: dict[str, Any] = {"service": None, "status": "cold", "error": None}
# The fused pipeline warms independently of the fast baselines: baselines show instantly,
# the pipeline row appears once its ranker is trained (tens of seconds later).
_pipe: dict[str, Any] = {"service": None, "status": "cold", "error": None}
_lock = threading.Lock()


def _start_fit() -> None:
    def work():
        try:
            from ..data import Dataset
            from ..serve import RecommendService
            svc = RecommendService(Dataset(DATA_DIR))
            svc.fit()
            with _lock:
                _state["service"] = svc
                _state["status"] = "ready"
            _start_pipeline_fit(svc)                  # warm the heavy pipeline next, reusing the data
        except Exception as e:                       # data missing, etc.
            log.exception("Recommender fitting failed")
            with _lock:
                _state["status"] = "error"
                _state["error"] = type(e).__name__

    with _lock:
        if _state["status"] in ("cold", "error"):
            _state["status"] = "fitting"
            _state["error"] = None
            threading.Thread(target=work, daemon=True).start()


def _start_pipeline_fit(base) -> None:
    """Fit the fused two-stage pipeline in the background, reusing the baseline service's data."""
    def work():
        try:
            from ..pipeline_serve import PipelineService
            svc = PipelineService(base.data, base=base)
            svc.fit()
            with _lock:
                _pipe["service"] = svc
                _pipe["status"] = "ready"
        except Exception as e:
            log.exception("Pipeline fitting failed")
            with _lock:
                _pipe["status"] = "error"
                _pipe["error"] = type(e).__name__

    with _lock:
        if _pipe["status"] in ("cold", "error"):
            _pipe["status"] = "fitting"
            _pipe["error"] = None
            threading.Thread(target=work, daemon=True).start()


def _service():
    with _lock:
        status, svc = _state["status"], _state["service"]
    if status == "ready":
        return svc
    if status in ("cold", "error"):
        _start_fit()
    raise HTTPException(503, "Recommenders are warming up (fitting models, up to ~2 min). Retry shortly.",
                        headers={"Retry-After": "15"})


@router.get("/recommend/status")
def status() -> dict[str, Any]:
    with _lock:
        st = _state["status"]
        pst = _pipe["status"]
    if st in ("cold", "error"):
        _start_fit()
        st = "fitting"
    return {"status": st, "pipeline_status": pst}


@router.get("/recommend")
def recommend(request: Request,
              user: Annotated[int, Query(ge=0, le=10_000_000)],
              hour: Annotated[int, Query(ge=0, le=23)] = 20,
              dow: Annotated[int, Query(ge=0, le=6)] = 4,
              rain: bool = False, hot: bool = False) -> dict[str, Any]:
    svc = _service()
    if not 0 <= user < svc.data.n_users:
        raise HTTPException(404, "Unknown user.")
    u = svc.data.users_df.iloc[user]
    req = svc.make_request(user, float(u["home_lat"]), float(u["home_lon"]), hour, dow, rain, hot)
    return _payload(svc, req, user_known=True)


@router.get("/recommend/at")
def recommend_at(request: Request,
                 lat: Annotated[float, Query(ge=36.0, le=44.0)],
                 lon: Annotated[float, Query(ge=-9.5, le=3.5)],
                 hour: Annotated[int, Query(ge=0, le=23)] = 20,
                 dow: Annotated[int, Query(ge=0, le=6)] = 4,
                 rain: bool = False, hot: bool = False) -> dict[str, Any]:
    """A brand-new user at a map point: no history, so only cold-start sources have signal."""
    svc = _service()
    req = svc.make_request(-1, lat, lon, hour, dow, rain, hot)
    return _payload(svc, req, user_known=False)


def _payload(svc, req, user_known: bool) -> dict[str, Any]:
    results = svc.recommend_all(req)
    with _lock:
        pipe, pstatus = _pipe["service"], _pipe["status"]
    if pipe is not None and pstatus == "ready":
        try:
            results.append(pipe.recommend(req))
        except Exception:                            # a live scoring error must not break the page
            log.exception("Pipeline scoring failed for a request")
    _attach_demand(svc, req, results)
    return {
        "context": {"lat": req.lat, "lon": req.lon, "hour": req.hour, "daypart": DAYPARTS[req.daypart],
                    "dow": req.dow, "rain": req.rain, "hot": req.hot, "city": ("BCN", "MAD")[req.city],
                    "user_known": user_known, "n_candidates": results[0]["n_candidates"] if results else 0,
                    "pipeline_status": pstatus, "has_oracle": _oracle(svc) is not None},
        "results": results,
    }


# -- two demand metrics shown per card --------------------------------------------
# 1) leaderboard_demand: the method's AVERAGE demand@5 over the October test set (from the
#    offline eval artifacts) — a global quality number, not about this request.
# 2) live_demand: the TRUE demand@5 of the 5 dishes this method just showed THIS user, from the
#    ground-truth oracle. Exact for the shown list, but the oracle reads hidden truth, so this is
#    a demo-only lens (production can't see it). Absent when truth files aren't present.
_oracle_state: dict[str, Any] = {"loaded": False, "oracle": None}


def _oracle(svc):
    if not _oracle_state["loaded"]:
        _oracle_state["loaded"] = True
        try:
            from ..eval.oracle import TruthOracle
            _oracle_state["oracle"] = TruthOracle(svc.data, DATA_DIR)
        except Exception:                            # truth files missing: no live metric, no error
            _oracle_state["oracle"] = None
    return _oracle_state["oracle"]


def _leaderboard_demand() -> dict[str, float]:
    """Average demand@5 per algorithm from the offline eval (leaderboard + the fused pipeline)."""
    out: dict[str, float] = {}
    from ..config import ARTIFACT_DIR
    import json
    lb = ARTIFACT_DIR / "leaderboard.json"
    if lb.exists():
        try:
            for a in json.loads(lb.read_text(encoding="utf-8")).get("algorithms", []):
                out[a["name"]] = a["metrics"]["demand"]["mean"]
        except Exception:
            pass
    rk = ARTIFACT_DIR / "ranking.json"
    if rk.exists():
        try:
            pipes = json.loads(rk.read_text(encoding="utf-8")).get("pipelines", [])
            best = max(pipes, key=lambda p: p["metrics"]["demand"]["mean"], default=None)
            if best:
                out["pipeline"] = best["metrics"]["demand"]["mean"]
        except Exception:
            pass
    return out


def _attach_demand(svc, req, results: list[dict]) -> None:
    import numpy as np
    oracle = _oracle(svc)
    lb = _leaderboard_demand()
    probs_by_item: dict[int, float] = {}
    if oracle is not None:
        cand = svc.data.available_mains(req.lat, req.lon, req.hour)
        if len(cand):
            probs = oracle.choice_probs(req, cand)
            probs_by_item = {int(c): float(p) for c, p in zip(cand, probs)}
    for r in results:
        r["leaderboard_demand"] = lb.get(r["algorithm"])
        if oracle is not None:
            r["live_demand"] = round(sum(probs_by_item.get(it["item_id"], 0.0)
                                         for it in r.get("items", [])), 4)
