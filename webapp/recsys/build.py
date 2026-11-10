"""One command to build what is missing or stale and start the app.

    python -m recsys                 # world -> embeddings -> database -> server on :8000
    python -m recsys --rebuild       # regenerate everything from scratch
    python -m recsys --no-serve      # build only
    python -m recsys --no-gemini     # never call Gemini (template menus, no Gemini embeddings)

Each step is skipped when its output is up to date:
- world: data/meta.json exists. If some menus are templates and Gemini is reachable, the
  world is regenerated so they become Gemini menus (menus already generated are cached).
- embeddings: data/embeddings/<backend>.json was built from the current items.parquet.
- database: the fingerprint stored in <schema>.build_info matches the data on disk; only the
  parts that changed are reloaded (tables, or just some vector spaces).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

from .config import DATA_DIR, Settings, load_settings, quiet_http_logs

log = logging.getLogger("recsys.build")
ALL_BACKENDS = ("tfidf", "bge", "gemini")
DEFAULT_BACKENDS = ("tfidf", "gemini")   # bge is opt-in: 15-30 min on CPU with Gemini-length texts


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def data_fingerprint(data_dir: Path) -> dict:
    """What a database load contains: the world version and each vector space's source."""
    from .embed import file_hash
    meta = _read_json(data_dir / "meta.json") or {}
    spaces = {}
    for p in sorted((data_dir / "embeddings").glob("*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        spaces[p.stem] = f"{m.get('items_hash')}:{m.get('dim')}"
    return {"world": meta.get("generated_at"), "items": file_hash(data_dir / "items.parquet"), "spaces": spaces}


def gemini_ok(settings: Settings) -> bool:
    """One tiny call (well under a cent) to see whether the key works and has credit."""
    if not settings.gemini_api_key:
        return False
    from .llm import Gemini, LLMError
    try:
        Gemini(settings).json('Return JSON {"ok": true}', purpose="probe", max_tokens=20, retries=1)
        return True
    except LLMError as e:
        log.warning("Gemini unavailable (%s): using template menus and local embeddings.", e)
        return False


def step_world(data_dir: Path, rebuild: bool, use_gemini: bool, settings: Settings) -> bool:
    from .synth.generate import build_world, write
    from .synth.world import WorldConfig
    meta = _read_json(data_dir / "meta.json")
    templates = sum(v == "template" for v in (meta or {}).get("menu_sources", {}).values())
    if meta and not rebuild and not (templates and use_gemini):
        log.info("World: up to date (%s, %d template menus)", meta["generated_at"], templates)
        return False
    reason = "rebuild requested" if rebuild else ("none yet" if not meta else f"upgrading {templates} template menus to Gemini")
    log.info("World: generating (%s) ...", reason)
    from .llm import Gemini
    cfg = replace(WorldConfig())
    visible, truth, new_meta = build_world(cfg, Gemini(settings) if use_gemini else None, data_dir)
    write(data_dir, visible, truth, new_meta)
    left = sum(v == "template" for v in new_meta["menu_sources"].values())
    log.info("World: %d orders, %d sessions in %.0fs; %d template menus left",
             len(visible["orders"]), len(visible["sessions"]), new_meta["seconds"], left)
    return True


def step_embeddings(data_dir: Path, backends: list[str]) -> None:
    from .embed import build, file_hash
    items_hash = file_hash(data_dir / "items.parquet")
    for b in backends:
        meta = _read_json(data_dir / "embeddings" / f"{b}.json")
        if meta and meta.get("items_hash") == items_hash:
            log.info("Embeddings %-6s: up to date", b)
            continue
        log.info("Embeddings %-6s: building%s ...", b, " (15-30 min on CPU)" if b == "bge" else "")
        try:
            m = build(b, data_dir)
            log.info("Embeddings %-6s: %d x %d in %.0fs, $%.4f", b, m["items"], m["dim"], m["seconds"], m["usd"])
        except Exception as e:      # an optional backend must not stop the app
            log.warning("Embeddings %-6s: skipped (%s)", b, type(e).__name__)


def step_database(data_dir: Path, settings: Settings, rebuild: bool) -> None:
    from .store.loader import load, read_fingerprint
    want = data_fingerprint(data_dir)
    try:
        have = None if rebuild else read_fingerprint(settings.database_url, settings.db_schema)
    except Exception as e:
        log.warning("Database: unreachable (%s); the app will start and retry.", type(e).__name__)
        return
    if have == want:
        log.info("Database: up to date")
        return
    tables = not have or have.get("world") != want["world"] or have.get("items") != want["items"]
    spaces = tuple(want["spaces"]) if tables else tuple(
        s for s, v in want["spaces"].items() if (have or {}).get("spaces", {}).get(s) != v)
    log.info("Database: loading %s ...", "tables and all vectors" if tables else f"vectors {', '.join(spaces)}")
    t0 = time.perf_counter()
    load(settings.database_url, settings.db_schema, data_dir, want, tables=tables, spaces=spaces)
    log.info("Database: loaded in %.0fs", time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m recsys", description=__doc__.split("\n")[0])
    ap.add_argument("--rebuild", action="store_true", help="regenerate everything")
    ap.add_argument("--no-gemini", action="store_true", help="never call Gemini")
    ap.add_argument("--no-serve", action="store_true", help="build only, do not start the server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backends", nargs="+", choices=ALL_BACKENDS, default=list(DEFAULT_BACKENDS),
                    help="embedding backends (default: tfidf gemini; add bge for the local model)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    quiet_http_logs()
    for noisy in ("sentence_transformers", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = load_settings()
    data_dir = DATA_DIR
    use_gemini = not args.no_gemini and gemini_ok(settings)
    step_world(data_dir, args.rebuild, use_gemini, settings)
    step_embeddings(data_dir, [b for b in args.backends if b != "gemini" or use_gemini])
    step_database(data_dir, settings, args.rebuild)
    if args.no_serve:
        return
    import uvicorn
    log.info("Serving on http://%s:%d", args.host, args.port)
    uvicorn.run("recsys.api.main:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
