"""Build the land borders the app's chart draws, which ship in ``geoguessr_ai/knowledge``.

Traces them from the country grid (see ``scripts/build_countries.py``), so the chart shows the very
borders a guess is judged by. Only rerun it after rebuilding that grid:

    python scripts/build_borders.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from global_land_mask import globe
from shapely.geometry import LineString

from geoguessr_ai.knowledge.countries import DEGREES_PER_PIXEL, _grid

OUT = Path(__file__).resolve().parents[1] / "src" / "geoguessr_ai" / "knowledge" / "borders.json"
FINE, COARSE = 0.02, 0.15
"""How far a line may stray from the grid's border, in degrees: one for looking closely, one for
the whole world, which is many fewer points to draw."""
SHORTEST = 3
"""Fewer cell sides than this is a sliver at a coast, not a border."""


def _sides() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Each side between two cells of different countries with land at it, between grid corners."""
    grid, _ = _grid()
    sides = []
    for down in (False, True):
        a, b = (grid[:-1], grid[1:]) if down else (grid[:, :-1], grid[:, 1:])
        rows, columns = np.nonzero(a != b)
        if down:
            ends = (rows + 1, columns), (rows + 1, columns + 1)
        else:
            ends = (rows, columns + 1), (rows + 1, columns + 1)
        lat = 90.0 - (ends[0][0] + ends[1][0]) / 2 * DEGREES_PER_PIXEL
        lon = (ends[0][1] + ends[1][1]) / 2 * DEGREES_PER_PIXEL - 180.0
        land = globe.is_land(np.clip(lat, -89.9, 89.9), np.clip(lon, -179.9, 179.9))
        sides += zip(
            zip(ends[0][0][land].tolist(), ends[0][1][land].tolist(), strict=True),
            zip(ends[1][0][land].tolist(), ends[1][1][land].tolist(), strict=True),
            strict=True,
        )
    return sides


def _chains(sides: list) -> list[list[tuple[int, int]]]:
    """The sides joined into lines, which end where three borders meet or at a coast."""
    meets = defaultdict(list)
    for a, b in sides:
        meets[a].append(b)
        meets[b].append(a)
    walked = set()

    def walk(start, step) -> list:
        chain = [start, step]
        walked.add(frozenset((start, step)))
        while len(meets[chain[-1]]) == 2:
            ahead = (p for p in meets[chain[-1]] if frozenset((chain[-1], p)) not in walked)
            onward = next(ahead, None)
            if onward is None:
                break
            walked.add(frozenset((chain[-1], onward)))
            chain.append(onward)
        return chain

    chains = []
    for start in [p for p, near in meets.items() if len(near) != 2]:
        for step in meets[start]:
            if frozenset((start, step)) not in walked:
                chains.append(walk(start, step))
    for start in meets:  # what's left is rings, round a country with nothing else touching it
        for step in meets[start]:
            if frozenset((start, step)) not in walked:
                chains.append(walk(start, step))
    return chains


def _degrees(row: float, column: float) -> tuple[float, float]:
    return column * DEGREES_PER_PIXEL - 180.0, 90.0 - row * DEGREES_PER_PIXEL


def _smooth(chain: list) -> list[tuple[float, float]]:
    """Through the middle of each side, not round its corners, which takes out the grid's steps."""
    sides = zip(chain, chain[1:], strict=False)
    middles = [_degrees((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in sides]
    return [_degrees(*chain[0]), *middles, _degrees(*chain[-1])]


def _lines(chains: list, tolerance: float) -> list[list[float]]:
    lines = []
    for chain in chains:
        points = LineString(_smooth(chain)).simplify(tolerance, preserve_topology=False).coords
        lines.append([round(value, 2) for point in points for value in point])
    return lines


def main() -> None:
    chains = [chain for chain in _chains(_sides()) if len(chain) > SHORTEST]
    borders = {"fine": _lines(chains, FINE), "coarse": _lines(chains, COARSE)}
    OUT.write_text(json.dumps(borders, separators=(",", ":")) + "\n", encoding="utf-8")
    for name, lines in borders.items():
        print(name, len(lines), "lines,", sum(len(line) for line in lines) // 2, "points")
    print(f"{OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
