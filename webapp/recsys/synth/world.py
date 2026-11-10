"""Configuration of the synthetic world and the time split shared by every stage.

Time convention: all timestamps are naive local time (Europe/Madrid, both cities share it).
The window is 2026-05-01 .. 2026-10-31; train = May-Aug, validation = Sep, test = Oct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class WorldConfig:
    seed: int = 42
    restaurants: tuple[int, int] = (180, 220)        # independent + chain outlets, (BCN, MAD)
    users: int = 6000
    user_city_share_bcn: float = 0.45
    start: date = date(2026, 5, 1)
    end: date = date(2026, 10, 31)                   # inclusive
    impressions_per_session: int = 10
    regular_sessions_per_week: float = 1.3     # median for regular users (scales the 1.8 base)
    traveller_share: float = 0.10
    workers: int = 8


@dataclass(frozen=True)
class Split:
    """Time split: retrieval models fit on train, the ranker learns on val, all report on test."""
    train_end: date = date(2026, 8, 31)   # inclusive
    val_end: date = date(2026, 9, 30)     # inclusive
    test_end: date = date(2026, 10, 31)   # inclusive


SPLIT = Split()


@dataclass(frozen=True)
class Segments:
    """User mix by lifecycle. Shares sum to 1."""
    regular: float = 0.55        # signed up before the window, order regularly
    light: float = 0.15          # signed up before the window, rarely order
    growth: float = 0.12         # sign up during May-Sep
    new_in_test: float = 0.18    # sign up in October (cold start in the test month)
    names: tuple[str, ...] = field(default=("regular", "light", "growth", "new_in_test"))


SEGMENTS = Segments()
