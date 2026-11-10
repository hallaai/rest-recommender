"""Geography of the two cities: districts, point sampling, distances, delivery zones.

Coordinates are real district centroids (WGS84). Density and income are rough relative
indices used only by the synthetic generator: density sets where people and restaurants
are, income shifts price sensitivity (a cold-start signal models can learn from geo).
Distances are great-circle kilometres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_KM = 6371.0088


@dataclass(frozen=True)
class District:
    code: str
    city: str
    name: str
    lat: float
    lon: float
    spread_km: float   # std-dev of the gaussian used to sample points around the centroid
    density: float     # relative share of residents / restaurants
    income: float      # 1.0 = city average


@dataclass(frozen=True)
class City:
    code: str
    name: str
    lat: float
    lon: float
    zoom: int


CITIES: tuple[City, ...] = (
    City("BCN", "Barcelona", 41.3990, 2.1650, 13),
    City("MAD", "Madrid", 40.4230, -3.6950, 12),
)

DISTRICTS: tuple[District, ...] = (
    # Barcelona
    District("BCN01", "BCN", "Ciutat Vella", 41.3809, 2.1734, 0.55, 1.4, 0.80),
    District("BCN02", "BCN", "Eixample", 41.3888, 2.1620, 0.90, 2.2, 1.20),
    District("BCN03", "BCN", "Sants-Montjuïc", 41.3727, 2.1405, 0.80, 1.2, 0.85),
    District("BCN04", "BCN", "Les Corts", 41.3858, 2.1300, 0.60, 0.6, 1.35),
    District("BCN05", "BCN", "Sarrià-Sant Gervasi", 41.4011, 2.1323, 0.80, 1.0, 1.70),
    District("BCN06", "BCN", "Gràcia", 41.4036, 2.1574, 0.60, 1.2, 1.10),
    District("BCN07", "BCN", "Horta-Guinardó", 41.4180, 2.1660, 0.80, 0.9, 0.85),
    District("BCN08", "BCN", "Nou Barris", 41.4400, 2.1770, 0.70, 0.8, 0.65),
    District("BCN09", "BCN", "Sant Andreu", 41.4350, 2.1900, 0.70, 0.7, 0.80),
    District("BCN10", "BCN", "Sant Martí", 41.4080, 2.1960, 0.90, 1.3, 1.00),
    # Madrid
    District("MAD01", "MAD", "Centro", 40.4155, -3.7074, 0.70, 1.6, 0.95),
    District("MAD02", "MAD", "Arganzuela", 40.3980, -3.6990, 0.70, 0.9, 1.00),
    District("MAD03", "MAD", "Retiro", 40.4080, -3.6720, 0.60, 0.7, 1.30),
    District("MAD04", "MAD", "Salamanca", 40.4300, -3.6770, 0.70, 1.2, 1.75),
    District("MAD05", "MAD", "Chamartín", 40.4590, -3.6760, 0.80, 0.9, 1.60),
    District("MAD06", "MAD", "Tetuán", 40.4600, -3.6980, 0.65, 0.9, 0.90),
    District("MAD07", "MAD", "Chamberí", 40.4340, -3.7040, 0.60, 1.2, 1.45),
    District("MAD08", "MAD", "Fuencarral-El Pardo", 40.4800, -3.7050, 1.20, 0.8, 1.05),
    District("MAD09", "MAD", "Moncloa-Aravaca", 40.4400, -3.7250, 0.90, 0.6, 1.40),
    District("MAD10", "MAD", "Latina", 40.4030, -3.7400, 0.90, 0.9, 0.80),
    District("MAD11", "MAD", "Carabanchel", 40.3830, -3.7270, 0.90, 0.9, 0.70),
    District("MAD12", "MAD", "Usera", 40.3830, -3.7060, 0.65, 0.6, 0.65),
    District("MAD13", "MAD", "Puente de Vallecas", 40.3910, -3.6600, 0.90, 0.9, 0.60),
    District("MAD14", "MAD", "Moratalaz", 40.4070, -3.6450, 0.60, 0.4, 0.85),
    District("MAD15", "MAD", "Ciudad Lineal", 40.4380, -3.6530, 0.90, 0.9, 0.95),
    District("MAD16", "MAD", "Hortaleza", 40.4700, -3.6400, 1.00, 0.7, 1.15),
)

DISTRICT_INDEX = {d.code: i for i, d in enumerate(DISTRICTS)}
CITY_CODES = tuple(c.code for c in CITIES)

# Areas with no homes or restaurants: (lat, lon, radius_km).
_PARKS = (
    (41.3640, 2.1580, 0.9),    # Montjuïc
    (41.4140, 2.1400, 1.2),    # Collserola edge
    (40.4180, -3.7550, 2.0),   # Casa de Campo
    (40.4150, -3.6840, 0.55),  # El Retiro
    (40.5100, -3.7600, 3.0),   # Monte de El Pardo
)
_KM_PER_DEG_LAT = 111.32


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; accepts scalars or numpy arrays (broadcasting)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _in_sea(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Barcelona coastline approximated by a line from Barceloneta to the Fòrum."""
    return (lon > 2.175) & (lat < 41.378 + 0.97 * (lon - 2.192))


def _in_park(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    bad = np.zeros(lat.shape, dtype=bool)
    for plat, plon, r in _PARKS:
        bad |= haversine_km(lat, lon, plat, plon) < r
    return bad


def is_land(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return ~(_in_sea(lat, lon) | _in_park(lat, lon))


def sample_points(rng: np.random.Generator, district: District, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n points around a district centroid, rejecting sea and parks."""
    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    need = n
    km_per_deg_lon = _KM_PER_DEG_LAT * np.cos(np.radians(district.lat))
    while need > 0:
        m = max(need * 2, 8)
        la = district.lat + rng.normal(0, district.spread_km, m) / _KM_PER_DEG_LAT
        lo = district.lon + rng.normal(0, district.spread_km, m) / km_per_deg_lon
        ok = is_land(la, lo)
        lats.append(la[ok][:need])
        lons.append(lo[ok][:need])
        need -= int(min(ok.sum(), need))
    return np.concatenate(lats), np.concatenate(lons)


def nearest_district(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Index into DISTRICTS of the closest centroid (a cheap stand-in for district polygons)."""
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)
    clat = np.array([d.lat for d in DISTRICTS])
    clon = np.array([d.lon for d in DISTRICTS])
    dist = haversine_km(lat[:, None], lon[:, None], clat[None, :], clon[None, :])
    return dist.argmin(axis=1)


def city_of(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """City code for each point: the cities are 500 km apart, so the closer centre wins."""
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)
    d = np.stack([haversine_km(lat, lon, c.lat, c.lon) for c in CITIES], axis=1)
    return np.array(CITY_CODES)[d.argmin(axis=1)]
