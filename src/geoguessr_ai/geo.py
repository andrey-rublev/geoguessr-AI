"""Geodesy helpers: great-circle distance, Web Mercator projection, and GeoGuessr scoring."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088
# Web Mercator clips latitudes here so that the projected world is a square.
MAX_MERCATOR_LAT = 85.05112878
# GeoGuessr-style scoring: 5000 points at 0 km, decaying exponentially with distance.
MAX_SCORE = 5000.0
SCORE_SCALE_KM = 1492.7


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if np.ndim(value) == 0 else value


def haversine_km(lat1, lon1, lat2, lon2) -> float | np.ndarray:
    """Great-circle distance in kilometres. Accepts scalars or broadcastable arrays (degrees)."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return _scalar_or_array(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def to_unit_vectors(lat, lon) -> np.ndarray:
    """Convert degrees to 3D unit vectors with shape (..., 3)."""
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def from_unit_vectors(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (..., 3) vectors (not necessarily normalised) back to (lat, lon) degrees."""
    xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def mercator_xy(lat, lon) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Project degrees to normalised Web Mercator coordinates, both in [0, 1] (origin top-left)."""
    lat = np.clip(np.asarray(lat, dtype=np.float64), -MAX_MERCATOR_LAT, MAX_MERCATOR_LAT)
    lon = np.asarray(lon, dtype=np.float64)
    x = (lon + 180.0) / 360.0
    y = 0.5 - np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) / (2 * np.pi)
    return _scalar_or_array(x), _scalar_or_array(y)


def inverse_mercator(x, y) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Inverse of :func:`mercator_xy`."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lon = x * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y))))
    return _scalar_or_array(lat), _scalar_or_array(lon)


def geoguessr_score(distance_km) -> float | np.ndarray:
    """Points a guess this far from the answer would earn (0-5000)."""
    return _scalar_or_array(
        MAX_SCORE * np.exp(-np.asarray(distance_km, dtype=np.float64) / SCORE_SCALE_KM)
    )
