"""Which country a place is in, from a 0.05 degree grid of Natural Earth borders.

The grid is built by ``scripts/build_countries.py``. Sea belongs to the nearest country, so
every location has one, including a pin dropped just offshore.
"""

from __future__ import annotations

import functools
from importlib import resources

import numpy as np
from PIL import Image

DEGREES_PER_PIXEL = 0.05


@functools.lru_cache(maxsize=1)
def _grid() -> tuple[np.ndarray, tuple[str, ...]]:
    folder = resources.files(__package__)
    with (folder / "countries.png").open("rb") as f:
        grid = np.asarray(Image.open(f))
    codes = ("",) + tuple((folder / "countries.txt").read_text(encoding="utf-8").split())
    return grid, codes


def country_codes() -> tuple[str, ...]:
    """ISO 3166-1 alpha-2 codes, indexed like :func:`country_index` (index 0 is unused)."""
    return _grid()[1]


def country_index(lat, lon) -> np.ndarray:
    """Index into :func:`country_codes` of the country at each location (degrees)."""
    grid, _ = _grid()
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    rows = np.clip(np.floor((90.0 - lat) / DEGREES_PER_PIXEL), 0, grid.shape[0] - 1)
    cols = np.floor((lon + 180.0) / DEGREES_PER_PIXEL) % grid.shape[1]
    return grid[rows.astype(np.intp), cols.astype(np.intp)]


def country_at(lat: float, lon: float) -> str:
    return country_codes()[int(country_index(lat, lon))]
