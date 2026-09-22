"""Weigh the model's geocells by what the bot knows beyond the photos.

The model only sees pixels: it spreads belief over places the game never goes, and it can't
read. Here that knowledge becomes a weight for every geocell, from

* where Google Street View exists at all (see :mod:`.facts`),
* clues about the country, like the writing on signs (see :mod:`.text`),
* clues about the latitude, like where the sun stands in the sky.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..geo import EARTH_RADIUS_KM, to_unit_vectors
from .countries import country_codes, country_index
from .facts import COVERAGE_WEIGHTS, per_country

DEFAULT_COVERAGE_STRENGTH = 0.25
"""How hard to favour countries with more Street View. Low, because where the game has really
sent you (the prior from played rounds) says the same thing and says it from experience; see
``scripts/check_strengths.py``."""
KM_PER_DEGREE = 111.2


@dataclass
class Evidence:
    """What the bot noticed besides the model's impression of the views."""

    countries: np.ndarray | None = None
    """Likelihood per country, indexed like :func:`.countries.country_codes`."""
    latitude: Callable[[np.ndarray], np.ndarray] | None = None
    """Likelihood of being at each latitude (degrees)."""
    notes: list[str] = field(default_factory=list)


def cell_countries(centroids: np.ndarray) -> np.ndarray:
    """(cells, countries) share of each geocell in each country.

    Points around each cell's centre, out to half the distance to its nearest neighbour, are
    looked up on the border map, land counting ten times as much as sea. A cell on a border
    is shared between the countries.
    """
    centroids = np.ascontiguousarray(centroids, dtype=np.float64)
    return _cell_countries(centroids.tobytes(), len(centroids))


@functools.lru_cache(maxsize=4)
def _cell_countries(raw: bytes, n: int) -> np.ndarray:
    from global_land_mask import globe
    from scipy.spatial import cKDTree

    lat, lon = np.frombuffer(raw).reshape(n, 2).T
    xyz = to_unit_vectors(lat, lon)
    chord = cKDTree(xyz).query(xyz, k=2)[0][:, 1] if n > 1 else np.full(n, 2.0)
    reach = np.clip(EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1)), 5.0, 300.0)

    rings = np.repeat([0.0, 1 / 3, 2 / 3, 1.0], [1, 8, 8, 8])
    spokes = np.arange(8) * 45.0
    bearings = np.radians(np.concatenate([[0.0], spokes, spokes + 22.5, spokes]))
    dist = reach[:, None] * rings
    lats = np.clip(lat[:, None] + dist * np.cos(bearings) / KM_PER_DEGREE, -89.0, 89.0)
    stretch = KM_PER_DEGREE * np.maximum(np.cos(np.radians(lat)), 0.05)[:, None]
    lons = (lon[:, None] + dist * np.sin(bearings) / stretch + 180.0) % 360.0 - 180.0

    land = globe.is_land(lats.ravel(), lons.ravel())
    cells = np.repeat(np.arange(n), rings.size)
    shares = np.zeros((n, len(country_codes())))
    np.add.at(shares, (cells, country_index(lats, lons).ravel()), np.where(land, 1.0, 0.1))
    return shares / shares.sum(axis=1, keepdims=True)


@functools.lru_cache(maxsize=1)
def coverage() -> np.ndarray:
    """How likely the game is to use each country, going by its Street View coverage."""
    return per_country(lambda country: COVERAGE_WEIGHTS[country.coverage], default=1.0)


def cell_weights(
    centroids: np.ndarray,
    evidence: Evidence | None = None,
    coverage_strength: float = DEFAULT_COVERAGE_STRENGTH,
) -> np.ndarray:
    """How well each geocell fits Street View coverage and the evidence, up to a constant."""
    per_country_weight = coverage() ** coverage_strength
    if evidence is not None and evidence.countries is not None:
        per_country_weight = per_country_weight * evidence.countries
    weights = cell_countries(centroids) @ per_country_weight
    if evidence is not None and evidence.latitude is not None:
        weights = weights * evidence.latitude(np.asarray(centroids)[:, 0])
    return weights
