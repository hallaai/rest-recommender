from __future__ import annotations

from dataclasses import replace

import pytest

from recsys.config import load_settings
from recsys.llm import BudgetExceeded, SpendLedger, parse_json, LLMError


@pytest.fixture
def ledger(tmp_path):
    s = replace(load_settings(), budget_total_usd=1.0, budget_daily_usd=0.5,
                pricing={"m": (1.0, 10.0)})
    return SpendLedger(s, tmp_path / "spend.jsonl")


def test_cost_per_million_tokens(ledger):
    assert ledger.cost("m", 1_000_000, 100_000) == pytest.approx(2.0)


def test_caps_refuse_before_spending(ledger):
    ledger.record("m", "menus", 100_000, 30_000, 1.0)      # $0.40
    ledger.check("m", 10_000, 1_000)                        # +$0.02: fine
    with pytest.raises(BudgetExceeded):
        ledger.check("m", 100_000, 10_000)                  # +$0.20 crosses the $0.50 daily cap
    assert ledger.summary()["menus"]["calls"] == 1


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_reservations_stop_parallel_overshoot(ledger):
    # Each call is estimated at $0.20; the daily cap is $0.50: only two may be in flight.
    r1 = ledger.check("m", 100_000, 10_000)
    r2 = ledger.check("m", 100_000, 10_000)
    with pytest.raises(BudgetExceeded):
        ledger.check("m", 100_000, 10_000)
    ledger.release(r1)
    ledger.release(r2)
    ledger.check("m", 100_000, 10_000)


def test_torn_ledger_line_is_skipped(ledger, tmp_path):
    ledger.record("m", "menus", 1000, 100, 0.1)
    with (tmp_path / "spend.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"ts": "2026-')
    assert len(ledger.records()) == 1


@pytest.mark.parametrize("bad", ["public", "pg_catalog", "information_schema", "Recsys", "a b", "x;drop", "é"])
def test_schema_name_is_restricted(bad):
    from recsys.config import validate_schema
    with pytest.raises(RuntimeError):
        validate_schema(bad)


def test_lab_schema_name_is_allowed():
    from recsys.config import validate_schema
    validate_schema("recsys")
