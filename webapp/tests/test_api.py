"""API through the real HTTP pipeline with a fake repository (no database)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from recsys.api.main import create_app


class FakeRepo:
    def overview(self):
        return {"counts": {"restaurants": 1}}

    def map_restaurants(self):
        return [{"id": 1, "lat": 41.39, "lon": 2.16}]

    def map_users(self):
        return []

    def restaurants_near(self, lat, lon):
        return [{"id": 1, "km": 0.5}]

    def restaurants(self, city, cuisine, q, limit, offset):
        return {"total": 1, "rows": [{"id": 1, "city": city, "cuisine": cuisine, "q": q, "limit": limit}]}

    def restaurant(self, rid):
        return {"id": rid, "menu": []} if rid == 1 else None

    def items(self, *a):
        return {"total": 0, "rows": []}

    def users(self, city, activity, limit, offset):
        return {"total": 0, "rows": []}

    def user(self, uid):
        return {"id": uid, "orders": []} if uid == 7 else None


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="", connect=False))


def test_meta_lists_cities_and_cuisines(client):
    m = client.get("/api/meta").json()
    assert [c["code"] for c in m["cities"]] == ["BCN", "MAD"]
    assert len(m["families"]) <= 8                 # chart colour budget
    assert all(c["family"] in m["families"] for c in m["cuisines"])


def test_filters_pass_through_validated(client):
    r = client.get("/api/restaurants", params={"city": "MAD", "cuisine": "ramen", "q": "ko", "limit": 10}).json()
    assert r["rows"][0] == {"id": 1, "city": "MAD", "cuisine": "ramen", "q": "ko", "limit": 10}


@pytest.mark.parametrize("path", [
    "/api/restaurants?city=PAR", "/api/restaurants?cuisine=martian", "/api/items?limit=0",
    "/api/items?limit=201", "/api/users?activity=vip", "/api/map/near?lat=10&lon=2",
    "/api/restaurants?q=" + "x" * 61, "/api/restaurants/abc",
])
def test_bad_input_is_rejected(client, path):
    assert client.get(path).status_code == 422


def test_missing_resources_are_404(client):
    assert client.get("/api/restaurants/2").status_code == 404
    assert client.get("/api/users/8").status_code == 404
    assert client.get("/api/users/7").json()["truth"] is None   # no truth files in tmp dir


def test_password_gate(tmp_path):
    c = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="s3cret", connect=False))
    assert c.get("/api/overview").status_code == 401
    assert c.post("/api/login", json={"username":"admin","password": "wrong"}).status_code == 401
    assert c.post("/api/login", json={"username":"admin","password": "s3cret"}).status_code == 200
    assert c.get("/api/overview").status_code == 200


def test_no_database_gives_503(tmp_path):
    c = TestClient(create_app(repo=None, data_dir=tmp_path, password="", connect=False))
    assert c.get("/api/overview").status_code == 503


def test_errors_do_not_leak_internals(tmp_path):
    class Broken(FakeRepo):
        def overview(self):
            raise RuntimeError("password=hunter2 at db.internal:5432")
    c = TestClient(create_app(repo=Broken(), data_dir=tmp_path, password="", connect=False),
                   raise_server_exceptions=False)
    r = c.get("/api/overview")
    assert r.status_code == 500 and "hunter2" not in r.text


def test_non_ascii_password_and_cookie_do_not_crash(tmp_path):
    c = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="contraseña", connect=False))
    assert c.post("/api/login", json={"username":"admin","password": "señor"}).status_code == 401
    raw = "recs_lab=ñ".encode("utf-8")          # what a browser may send; httpx won't encode str
    assert c.get("/api/overview", headers={"cookie": raw}).status_code == 401
    assert c.post("/api/login", json={"username":"admin","password": "contraseña"}).status_code == 200
    assert c.get("/api/overview").status_code == 200


def test_cookie_is_not_derived_from_password(tmp_path):
    a = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="pw", connect=False))
    b = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="pw", connect=False))
    a.post("/api/login", json={"username":"admin","password": "pw"})
    b.post("/api/login", json={"username":"admin","password": "pw"})
    assert a.cookies.get("recs_lab") != b.cookies.get("recs_lab")


def test_login_is_throttled(tmp_path):
    c = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="pw", connect=False))
    codes = [c.post("/api/login", json={"username":"admin","password": "x"}).status_code for _ in range(11)]
    assert codes[:10] == [401] * 10 and codes[10] == 429
    assert c.post("/api/login", json={"username":"admin","password": "pw"}).status_code == 429   # still blocked


def test_api_docs_are_behind_the_gate(tmp_path):
    c = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="pw", connect=False))
    assert c.get("/api/openapi.json").status_code == 401
    assert c.get("/docs").status_code in (404, 405) or "swagger" not in c.get("/docs").text.lower()


def test_dropped_database_connection_gives_503_not_500(tmp_path):
    import psycopg

    class Dropped(FakeRepo):
        def overview(self):
            raise psycopg.OperationalError("server closed the connection unexpectedly")
    c = TestClient(create_app(repo=Dropped(), data_dir=tmp_path, password="", connect=False),
                   raise_server_exceptions=False)
    r = c.get("/api/overview")
    assert r.status_code == 503 and r.headers["retry-after"] == "10"
    assert "closed" not in r.text


def test_leaderboard_404_without_results(tmp_path, monkeypatch):
    import recsys.api.results as results
    monkeypatch.setattr(results, "ARTIFACT_DIR", tmp_path)
    c = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="", connect=False))
    r = c.get("/api/leaderboard")
    assert r.status_code == 404 and "runner" in r.json()["detail"]


def test_leaderboard_served_when_present(tmp_path, monkeypatch):
    import json
    import recsys.api.results as results
    monkeypatch.setattr(results, "ARTIFACT_DIR", tmp_path)
    (tmp_path / "leaderboard.json").write_text(json.dumps({"algorithms": [], "n_requests": 5}))
    c = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="", connect=False))
    assert c.get("/api/leaderboard").json()["n_requests"] == 5


def test_recommend_warming_up_returns_503(tmp_path):
    import recsys.api.recommend as rec
    rec._state.update({"service": None, "status": "cold", "error": None})
    c = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="", connect=False))
    r = c.get("/api/recommend?user=1")
    assert r.status_code == 503 and r.headers.get("retry-after")
    assert c.get("/api/recommend/status").json()["status"] in ("fitting", "cold")


def test_gate_withholds_pages_and_code_until_login(tmp_path):
    c = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="pw", connect=False))
    # unauthenticated: the app page and its JS are NOT served; the login page is returned instead
    root = c.get("/")
    assert root.status_code == 200 and "Sign in" in root.text and 'id="view"' not in root.text
    js = c.get("/js/app.js")
    assert "Sign in" in js.text and "export async function api" not in js.text   # real code withheld
    assert c.get("/api/overview").status_code == 401
    # login page assets are reachable so the login screen renders
    assert c.get("/login.html").status_code == 200
    # after login, the real app and code are served
    assert c.post("/api/login", json={"username": "admin", "password": "pw"}).status_code == 200
    assert "export async function api" in c.get("/js/app.js").text


def test_wrong_username_is_rejected(tmp_path):
    c = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="pw", connect=False))
    assert c.post("/api/login", json={"username": "root", "password": "pw"}).status_code == 401
    assert c.get("/api/overview").status_code == 401


def test_logout_clears_the_session(tmp_path):
    c = TestClient(create_app(repo=FakeRepo(), data_dir=tmp_path, password="pw", connect=False))
    c.post("/api/login", json={"username": "admin", "password": "pw"})
    assert c.get("/api/overview").status_code == 200
    c.post("/api/logout")
    assert c.get("/api/overview").status_code == 401
