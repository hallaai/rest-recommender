"""Explore endpoints: overview, map layers, restaurants, products, users."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request

from ..geo import CITIES, DISTRICTS
from ..synth.vocab import CATEGORIES, CUISINES, DAYPARTS, FAMILIES, FAMILY_OF

router = APIRouter(prefix="/api")
City = Literal["BCN", "MAD"]
CuisineKey = Literal[tuple(c.key for c in CUISINES)]  # type: ignore[valid-type]
Category = Literal[CATEGORIES]  # type: ignore[valid-type]
Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0, le=100_000)]


def _repo(request: Request):
    app = request.app
    if app.state.repo is None and app.state.connect_db is not None:
        app.state.connect_db(app)            # reconnect if the database came back
    if app.state.repo is None:
        raise HTTPException(503, "The database is unavailable. Try again shortly.", headers={"Retry-After": "30"})
    return app.state.repo


@router.get("/meta")
def meta() -> dict[str, Any]:
    return {
        "cities": [{"code": c.code, "name": c.name, "lat": c.lat, "lon": c.lon, "zoom": c.zoom} for c in CITIES],
        "districts": [{"code": d.code, "city": d.city, "name": d.name, "lat": d.lat, "lon": d.lon} for d in DISTRICTS],
        "cuisines": [{"key": c.key, "label": c.label, "family": FAMILY_OF[c.key]} for c in CUISINES],
        "families": list(FAMILIES),
        "categories": list(CATEGORIES),
        "dayparts": list(DAYPARTS),
    }


@router.get("/overview")
def overview(request: Request) -> dict[str, Any]:
    return _repo(request).overview()


@router.get("/map/restaurants")
def map_restaurants(request: Request) -> list[dict[str, Any]]:
    return _repo(request).map_restaurants()


@router.get("/map/users")
def map_users(request: Request) -> list[dict[str, Any]]:
    return _repo(request).map_users()


@router.get("/map/near")
def near(request: Request, lat: Annotated[float, Query(ge=36.0, le=44.0)],
         lon: Annotated[float, Query(ge=-9.5, le=3.5)]) -> list[dict[str, Any]]:
    return _repo(request).restaurants_near(lat, lon)


@router.get("/restaurants")
def restaurants(request: Request, city: City | None = None, cuisine: CuisineKey | None = None,
                q: Annotated[str | None, Query(max_length=60)] = None,
                limit: Limit = 50, offset: Offset = 0) -> dict[str, Any]:
    return _repo(request).restaurants(city, cuisine, q, limit, offset)


@router.get("/restaurants/{rid}")
def restaurant(request: Request, rid: int) -> dict[str, Any]:
    r = _repo(request).restaurant(rid)
    if not r:
        raise HTTPException(404, "Restaurant not found.")
    return r


@router.get("/items")
def items(request: Request, city: City | None = None, cuisine: CuisineKey | None = None,
          category: Category | None = None, vegetarian: bool | None = None,
          q: Annotated[str | None, Query(max_length=60)] = None,
          limit: Limit = 50, offset: Offset = 0) -> dict[str, Any]:
    return _repo(request).items(city, cuisine, category, q, vegetarian, limit, offset)


@router.get("/users")
def users(request: Request, city: City | None = None,
          activity: Literal["none", "light", "regular", "heavy", "traveller"] | None = None,
          limit: Limit = 50, offset: Offset = 0) -> dict[str, Any]:
    return _repo(request).users(city, activity, limit, offset)


@router.get("/users/{uid}")
def user(request: Request, uid: int) -> dict[str, Any]:
    u = _repo(request).user(uid)
    if not u:
        raise HTTPException(404, "User not found.")
    u["truth"] = request.app.state.truth.user(uid)
    return u
