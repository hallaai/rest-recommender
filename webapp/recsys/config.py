"""Settings from environment variables (and an optional webapp/.env file).

Secrets never live in code: DATABASE_URL and GEMINI_API_KEY come from webapp/.env.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WEBAPP_DIR / "data"            # generated world (parquet), gitignored
ARTIFACT_DIR = WEBAPP_DIR / "artifacts"   # fitted models and experiment results, gitignored

# USD per million tokens: (input, output). Check ai.google.dev/gemini-api/docs/pricing;
# override with PRICING_JSON.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-embedding-001": (0.15, 0.0),
    "gemini-embedding-2": (0.20, 0.0),
}


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    database_url: str
    gemini_api_key: str
    db_schema: str
    model_text: str          # menu generation, LLM reranker
    model_embed: str
    embed_dim: int
    budget_total_usd: float
    budget_daily_usd: float
    vcpu_hour_usd: float     # reference compute price for cost estimates
    max_concurrent: int      # CPU-bound recommendation requests at once
    app_username: str
    app_password: str
    pricing: dict[str, tuple[float, float]]


def validate_schema(schema: str) -> None:
    """The loader runs DROP SCHEMA on this name, so allow only a lab-owned lowercase identifier."""
    if (not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", schema) or schema == "public"
            or schema.startswith(("pg_", "information_schema"))):
        raise RuntimeError("DB_SCHEMA must be a lowercase identifier other than public/pg_*/information_schema.")


def use_os_trust_store() -> None:
    """Trust the OS certificate store (needed when antivirus/proxy intercepts HTTPS)."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


def quiet_http_logs() -> None:
    import logging
    for name in ("httpx", "google_genai", "google_genai.models"):
        logging.getLogger(name).setLevel(logging.WARNING)


def load_settings() -> Settings:
    use_os_trust_store()
    _load_dotenv(WEBAPP_DIR / ".env")
    pricing = dict(DEFAULT_PRICING)
    if os.environ.get("PRICING_JSON"):
        pricing.update({k: tuple(v) for k, v in json.loads(os.environ["PRICING_JSON"]).items()})
    schema = os.environ.get("DB_SCHEMA", "recsys")
    validate_schema(schema)
    return Settings(
        database_url=os.environ.get("DATABASE_URL", ""),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        db_schema=schema,
        model_text=os.environ.get("MODEL_TEXT", "gemini-3.1-flash-lite"),
        model_embed=os.environ.get("MODEL_EMBED", "gemini-embedding-001"),
        embed_dim=int(os.environ.get("EMBED_DIM", 768)),
        budget_total_usd=float(os.environ.get("BUDGET_TOTAL_USD", 10)),
        budget_daily_usd=float(os.environ.get("BUDGET_DAILY_USD", 5)),
        vcpu_hour_usd=float(os.environ.get("VCPU_HOUR_USD", 0.04)),
        max_concurrent=int(os.environ.get("MAX_CONCURRENT", 4)),
        app_username=os.environ.get("APP_USERNAME", "admin"),
        app_password=os.environ.get("APP_PASSWORD", ""),
        pricing=pricing,
    )
