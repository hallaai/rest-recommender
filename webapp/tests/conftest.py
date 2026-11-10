"""Shared fixtures: a small synthetic world generated once per test session (template menus,
no API calls, no database)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from recsys.synth.generate import build_world
from recsys.synth.world import WorldConfig

SMALL = replace(WorldConfig(), users=150, restaurants=(60, 70), workers=1)


@pytest.fixture(scope="session")
def small_world(tmp_path_factory) -> tuple[dict, dict, dict]:
    out: Path = tmp_path_factory.mktemp("world")
    return build_world(SMALL, None, out)


@pytest.fixture(scope="session")
def restaurant_hours(small_world) -> list[list[tuple[int, int]]]:
    visible, _, _ = small_world
    return [[tuple(w) for w in json.loads(h)] for h in visible["restaurants"]["hours"]]
