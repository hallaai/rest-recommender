"""Gemini gateway: JSON generation and embeddings, with a spend ledger.

Every call reserves its estimated cost against the caps before it runs (atomically, so
parallel threads cannot overshoot) and records the real cost after it. The ledger is a
JSON-lines file in artifacts/ so it works without the database. Days are UTC.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ARTIFACT_DIR, Settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Provider failure or unusable output. Message is safe to show."""


class BudgetExceeded(LLMError):
    pass


@dataclass(frozen=True)
class SpendRecord:
    ts: str
    model: str
    purpose: str
    tokens_in: int
    tokens_out: int
    usd: float
    seconds: float


class SpendLedger:
    """Append-only spend log with total and daily caps."""

    def __init__(self, settings: Settings, path: Path | None = None) -> None:
        self._s = settings
        self._path = path or ARTIFACT_DIR / "llm_spend.jsonl"
        self._lock = threading.Lock()
        self._pending = 0.0      # cost reserved by calls in flight

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        pin, pout = self._s.pricing.get(model, (1.0, 5.0))  # unknown model: assume pricey
        return (tokens_in * pin + tokens_out * pout) / 1e6

    def records(self) -> list[SpendRecord]:
        with self._lock:
            return self._read()

    def _read(self) -> list[SpendRecord]:
        if not self._path.exists():
            return []
        out = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(SpendRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue      # blank or torn line
        return out

    def check(self, model: str, est_in: int, est_out: int) -> float:
        """Refuse if the call could cross a cap; otherwise reserve its estimated cost and
        return it. The caller must release() the reservation when the call ends."""
        nxt = self.cost(model, est_in, est_out)
        with self._lock:
            recs = self._read()
            today = datetime.now(timezone.utc).date().isoformat()
            total = sum(r.usd for r in recs) + self._pending
            daily = sum(r.usd for r in recs if r.ts.startswith(today)) + self._pending
            if total + nxt > self._s.budget_total_usd:
                raise BudgetExceeded(f"Total Gemini budget ${self._s.budget_total_usd} would be exceeded.")
            if daily + nxt > self._s.budget_daily_usd:
                raise BudgetExceeded(f"Daily Gemini budget ${self._s.budget_daily_usd} would be exceeded.")
            self._pending += nxt
        return nxt

    def release(self, reserved: float) -> None:
        with self._lock:
            self._pending = max(0.0, self._pending - reserved)

    def record(self, model: str, purpose: str, tin: int, tout: int, seconds: float) -> SpendRecord:
        rec = SpendRecord(datetime.now(timezone.utc).isoformat(timespec="seconds"), model, purpose,
                          tin, tout, round(self.cost(model, tin, tout), 6), round(seconds, 3))
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__) + "\n")
        return rec

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for r in self.records():
            s = out.setdefault(r.purpose, {"calls": 0, "usd": 0.0, "tokens_in": 0, "tokens_out": 0})
            s["calls"] += 1
            s["usd"] = round(s["usd"] + r.usd, 6)
            s["tokens_in"] += r.tokens_in
            s["tokens_out"] += r.tokens_out
        return out


def parse_json(text: str) -> Any:
    """Parse a JSON reply, tolerating ```json fences."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError("Model returned malformed JSON.") from e


def _status_hint(e: Exception) -> str:
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    return f"HTTP {code}, {type(e).__name__}" if isinstance(code, int) else type(e).__name__


class Gemini:
    """The only place that calls Gemini."""

    def __init__(self, settings: Settings, ledger: SpendLedger | None = None) -> None:
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set (webapp/.env).")
        from google import genai
        self._s = settings
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self.ledger = ledger or SpendLedger(settings)

    def json(self, prompt: str, *, purpose: str, max_tokens: int = 4096,
             temperature: float = 0.9, model: str | None = None, retries: int = 3) -> Any:
        from google.genai import types
        model = model or self._s.model_text
        cfg = types.GenerateContentConfig(response_mime_type="application/json",
                                          temperature=temperature, max_output_tokens=max_tokens)
        last: Exception | None = None
        for attempt in range(retries):
            reserved = self.ledger.check(model, len(prompt) // 3, max_tokens)   # every attempt costs
            t0 = time.perf_counter()
            try:
                resp = self._client.models.generate_content(model=model, contents=prompt, config=cfg)
            except Exception as e:  # provider errors vary by SDK version
                last = e
                log.warning("Gemini %s failed (%s), attempt %d", purpose, _status_hint(e), attempt + 1)
                time.sleep(2 ** attempt)
                continue
            finally:
                self.ledger.release(reserved)
            um = resp.usage_metadata
            tin = um.prompt_token_count or 0
            tout = (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
            self.ledger.record(model, purpose, tin, tout, time.perf_counter() - t0)
            try:
                return parse_json(resp.text or "")
            except LLMError as e:
                last = e
                log.warning("Gemini %s returned bad JSON (%d tokens out), attempt %d",
                            purpose, tout, attempt + 1)
        raise LLMError(f"Gemini call for {purpose} failed ({_status_hint(last) if last else '?'}).")

    def embed(self, texts: list[str], *, purpose: str, task: str = "RETRIEVAL_DOCUMENT",
              retries: int = 4) -> list[list[float]]:
        """Embed up to 100 texts in one request; vectors are returned as-is (caller normalises)."""
        from google.genai import types
        model = self._s.model_embed
        est = sum(len(t) for t in texts) // 3
        cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=self._s.embed_dim)
        for attempt in range(retries):
            reserved = self.ledger.check(model, est, 0)
            t0 = time.perf_counter()
            try:
                resp = self._client.models.embed_content(model=model, contents=texts, config=cfg)
            except Exception as e:
                log.warning("Embedding failed (%s), attempt %d", _status_hint(e), attempt + 1)
                time.sleep(2 ** attempt * 2)
                continue
            finally:
                self.ledger.release(reserved)
            # The embed API does not report tokens for every model; fall back to an estimate.
            meta = getattr(resp, "metadata", None)
            tin = getattr(meta, "billable_character_count", None) or est
            self.ledger.record(model, purpose, int(tin), 0, time.perf_counter() - t0)
            return [e.values for e in resp.embeddings]
        raise LLMError("Embedding request failed after retries.")
