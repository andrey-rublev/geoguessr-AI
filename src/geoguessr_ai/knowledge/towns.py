"""Towns and where they are, for telling the scale of a map by the towns it names.

The names are OpenStreetView-5M's, without accents and in lowercase, each at the middle of its
photos (see ``scripts/build_towns.py``). Several towns can share a name.
"""

from __future__ import annotations

import functools
from importlib import resources


@functools.lru_cache(maxsize=1)
def towns() -> dict[str, tuple[tuple[float, float], ...]]:
    """Where each town of each name is, as (latitude, longitude)."""
    found: dict[str, list[tuple[float, float]]] = {}
    text = resources.files(__package__).joinpath("towns.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        name, lat, lon = line.split("\t")
        found.setdefault(name, []).append((float(lat), float(lon)))
    return {name: tuple(places) for name, places in found.items()}
