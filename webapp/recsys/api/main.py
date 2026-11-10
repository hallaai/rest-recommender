"""FastAPI app: API under /api, the UI as static files at /.

    python -m uvicorn recsys.api.main:app      # http://127.0.0.1:8000

With APP_PASSWORD set, nothing is served until login: the gate withholds every page, script,
stylesheet and API response from unauthenticated callers and returns only the login page, so a
scraper that has not logged in sees no data and no application code. Login is username + password
(APP_USERNAME, default "admin"), throttled per client address.
"""
from __future__ import annotations

import hmac
import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psycopg
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, Field

from ..config import DATA_DIR, WEBAPP_DIR, load_settings
from .explore import router as explore_router
from .results import router as results_router
from .recommend import router as recommend_router
from .truth_view import TruthView

log = logging.getLogger(__name__)
FRONTEND = WEBAPP_DIR / "frontend"
LOGIN_PAGE = FRONTEND / "login.html"
COOKIE = "recs_lab"
LOGIN_WINDOW_S, LOGIN_MAX_FAILS = 300, 10
# The only paths an unauthenticated caller may fetch: the login page's own assets and favicon.
ALLOW_UNAUTH = frozenset({"/login.html", "/login.css", "/favicon.ico"})


class Login(BaseModel):
    username: str = Field(max_length=200)
    password: str = Field(max_length=200)


def _eq(a: str, b: str) -> bool:
    """Constant-time comparison that accepts any unicode (compare_digest rejects non-ASCII str)."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


class LoginThrottle:
    """At most LOGIN_MAX_FAILS failed logins per client address per window."""

    def __init__(self) -> None:
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def blocked(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._fails.get(key, []) if now - t < LOGIN_WINDOW_S]
            self._fails[key] = recent
            return len(recent) >= LOGIN_MAX_FAILS

    def fail(self, key: str) -> None:
        with self._lock:
            self._fails.setdefault(key, []).append(time.monotonic())


def _connector():
    """Opens the repository; retried at most every 30 s while the database is down."""
    lock, state = threading.Lock(), {"last": 0.0}

    def connect(app: FastAPI) -> None:
        from ..store.repo import Repo
        with lock:
            if app.state.repo is not None or time.monotonic() - state["last"] < 30:
                return
            state["last"] = time.monotonic()
            s = load_settings()
            try:
                app.state.repo = Repo(s.database_url, s.db_schema)
                log.info("Database connected")
            except Exception as e:
                log.error("Database unavailable: %s", type(e).__name__)
    return connect


def create_app(repo: Any = None, data_dir: Path = DATA_DIR, password: str | None = None,
               connect: bool = True) -> FastAPI:
    """repo=None with connect=True opens Postgres from settings at startup (tests pass a fake)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.repo is None and connect:
            app.state.connect_db = _connector()
            app.state.connect_db(app)       # may fail; requests retry (see explore._repo)
        yield
        if connect and app.state.repo is not None:
            app.state.repo.close()

    # Docs live under /api so the password gate covers them too.
    app = FastAPI(title="Food Recs Lab", lifespan=lifespan, docs_url="/api/docs",
                  openapi_url="/api/openapi.json", redoc_url=None)
    app.state.repo = repo
    app.state.connect_db = None
    app.state.truth = TruthView(data_dir)
    settings = None if not connect else load_settings()
    pw = password if password is not None else (settings.app_password if settings else "")
    user = settings.app_username if settings else "admin"
    if not pw:
        log.warning("APP_PASSWORD is empty: the whole app, including the simulation's hidden truth, is open.")
    # Session token: random per process, unrelated to the password (no offline guessing from a
    # captured cookie); restarting the server logs everyone out.
    session_token = secrets.token_urlsafe(32)
    throttle = LoginThrottle()

    def authed(request: Request) -> bool:
        return _eq(request.cookies.get(COOKIE, ""), session_token)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        """Protect everything. Unauthenticated callers get only the login page and its assets;
        every other path (pages, JS, CSS, API, data) is withheld, so a scraper sees nothing."""
        path = request.url.path
        if not pw or path in ALLOW_UNAUTH or path == "/api/login" or authed(request):
            return await call_next(request)
        if path.startswith("/api"):
            return JSONResponse({"detail": "Login required."}, status_code=401)
        # Any other GET from a browser: hand back the login page (200) instead of the real app.
        if LOGIN_PAGE.exists():
            return FileResponse(LOGIN_PAGE, headers={"Cache-Control": "no-store"})
        return JSONResponse({"detail": "Login required."}, status_code=401)

    @app.post("/api/login")
    def login(body: Login, request: Request) -> JSONResponse:
        if not pw:
            return JSONResponse({"ok": True})
        client = request.client.host if request.client else "?"
        if throttle.blocked(client):
            return JSONResponse({"detail": "Too many attempts; try again in a few minutes."},
                                status_code=429, headers={"Retry-After": str(LOGIN_WINDOW_S)})
        ok = _eq(body.username, user) & _eq(body.password, pw)   # & not and: no early-out timing leak
        if not ok:
            throttle.fail(client)
            return JSONResponse({"detail": "Wrong username or password."}, status_code=401)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE, session_token, httponly=True, samesite="strict", max_age=7 * 86400,
                        secure=request.url.scheme == "https")
        return resp

    @app.post("/api/logout")
    def logout() -> JSONResponse:
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE)
        return resp

    @app.exception_handler(psycopg.OperationalError)
    @app.exception_handler(PoolTimeout)
    async def db_down(request: Request, exc: Exception) -> JSONResponse:
        log.error("Database error on %s: %s", request.url.path, type(exc).__name__)
        return JSONResponse({"detail": "The database is temporarily unavailable. Try again shortly."},
                            status_code=503, headers={"Retry-After": "10"})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "Internal error."}, status_code=500)

    app.include_router(explore_router)
    app.include_router(results_router)
    app.include_router(recommend_router)
    if FRONTEND.exists():
        app.mount("/", UTF8StaticFiles(directory=FRONTEND, html=True), name="ui")
    return app


class UTF8StaticFiles(StaticFiles):
    """StaticFiles that declares charset=utf-8 on JS/CSS, so multibyte characters in the
    source (euro, plus-minus, accents) parse correctly in the browser."""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        ctype = resp.headers.get("content-type", "")
        if ("javascript" in ctype or ctype.startswith("text/css")) and "charset" not in ctype:
            resp.headers["content-type"] = f"{ctype}; charset=utf-8"
        return resp


app = create_app()
