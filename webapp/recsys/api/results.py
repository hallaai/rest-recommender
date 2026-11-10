"""Serve the evaluation results (leaderboard) to the Compare and Costs pages.

The leaderboard is produced offline by `python -m recsys.eval.runner` and read from
artifacts/leaderboard.json. The endpoint returns 404 with a clear message until it exists.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import ARTIFACT_DIR

router = APIRouter(prefix="/api")


@router.get("/leaderboard")
def leaderboard() -> dict:
    path = ARTIFACT_DIR / "leaderboard.json"
    if not path.exists():
        raise HTTPException(404, "No evaluation yet. Run: python -m recsys.eval.runner")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/ranking")
def ranking() -> dict:
    path = ARTIFACT_DIR / "ranking.json"
    if not path.exists():
        raise HTTPException(404, "No ranking evaluation yet. Run: python -m recsys.eval.rank_runner")
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(name: str, how: str) -> dict:
    path = ARTIFACT_DIR / name
    if not path.exists():
        raise HTTPException(404, f"Not generated yet. Run: {how}")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/ann")
def ann() -> dict:
    return _artifact("ann.json", "python -m recsys.eval.ann_bench")


@router.get("/scaling")
def scaling() -> dict:
    return _artifact("scaling.json", "python -m recsys.eval.scaling")


@router.get("/bandit")
def bandit() -> dict:
    return _artifact("bandit.json", "python -m recsys.eval.bandit")


@router.get("/gemini_rerank")
def gemini_rerank() -> dict:
    return _artifact("gemini_rerank.json", "python -m recsys.eval.gemini_rerank")
