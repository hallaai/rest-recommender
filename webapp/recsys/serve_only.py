"""Cloud serve-only entrypoint: start the web app against data already on disk.

    python -m recsys.serve_only            # serve recsys.api.main:app, no build

Unlike `python -m recsys` (recsys/build.py), this NEVER generates the synthetic world,
rebuilds embeddings or loads Postgres. It assumes `data/` is already present (baked into the
container image) and that Postgres is already populated. That makes it the right entrypoint for
a cloud container: fast boot, no Gemini spend, no writes to the database on startup.

Host and port come from the environment so a container platform can place the listener:
    HOST (default 0.0.0.0)   PORT (default 8000)

The app still tolerates the database being down: Explore pages degrade, but the Recommend and
Compare pages (which read data/*.parquet) keep working, and the DB reconnects on its own.
"""
from __future__ import annotations

import logging
import os

from .config import DATA_DIR, load_settings, quiet_http_logs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    quiet_http_logs()
    load_settings()  # reads webapp/.env in dev; in the container the values come from the env
    host = os.environ.get("HOST", "0.0.0.0")  # noqa: S104 - a container must bind all interfaces
    port = int(os.environ.get("PORT", "8000"))
    log = logging.getLogger("recsys.serve_only")
    if not (DATA_DIR / "meta.json").exists():
        log.error("No world found at %s (meta.json missing). This entrypoint does not build; "
                  "bake data/ into the image or run `python -m recsys` locally first.", DATA_DIR)
        raise SystemExit(1)
    import uvicorn
    log.info("Serving (serve-only, no build) on http://%s:%d", host, port)
    uvicorn.run("recsys.api.main:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
