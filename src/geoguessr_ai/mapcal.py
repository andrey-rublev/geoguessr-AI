"""Work out where the world is on a screenshot of the OpenGuessr map.

The map is Web Mercator, so the on-screen position of any (lat, lon) is fully
described by three numbers: the width of the whole world in screen pixels and the
screen position of the world's top-left corner. We recover them by matching the
screenshot's water/land pattern against a global land mask: a coarse search over
scales and all translations, then two finer passes around the best candidate.

This needs no knowledge of the page internals, window size, browser zoom, or
display scaling -- only pixels.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geo import inverse_mercator, mercator_xy

_BASE_RES = 4096
_PYRAMID = (512, 1024, 2048, _BASE_RES)
_CACHE_DIR = Path.home() / ".cache" / "geoguessr_ai"
OFF_WORLD = -1.0  # Outside the world the map is grey, i.e. "not water".


class MapNotFound(RuntimeError):
    """The screenshot could not be matched to the world map."""


@dataclass(frozen=True)
class MapProjection:
    world_px: float
    """Width (and height) of the whole Web Mercator world, in screenshot pixels."""
    origin_x: float
    """Screenshot x of the world's left edge (any copy, if the map wraps)."""
    origin_y: float
    """Screenshot y of the world's top edge."""
    wraps: bool = False
    """Whether the map repeats horizontally."""
    score: float = 0.0
    """Match quality: mean water/land agreement in [-1, 1]."""

    def to_pixel(self, lat: float, lon: float, view_width: float | None = None):
        mx, my = mercator_xy(lat, lon)
        x = self.origin_x + self.world_px * mx
        y = self.origin_y + self.world_px * my
        if self.wraps and view_width is not None:
            # Use the copy of the repeated world closest to the middle of the view.
            x -= self.world_px * round((x - view_width / 2) / self.world_px)
        return x, y

    def to_latlon(self, x: float, y: float):
        mx = (x - self.origin_x) / self.world_px
        if self.wraps:
            mx %= 1.0
        return inverse_mercator(mx, (y - self.origin_y) / self.world_px)


def water_mask(rgb: np.ndarray) -> np.ndarray:
    """+1 where a map screenshot shows water, -1 elsewhere (land, borders, labels, off-world)."""
    px = rgb[..., :3].astype(np.int16)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    water = (b > 140) & (b - r > 35) & (b - g > -25)
    return np.where(water, 1.0, -1.0).astype(np.float32)


@functools.lru_cache(maxsize=1)
def _ocean_bits() -> np.ndarray:
    """Boolean ocean mask of the Web Mercator world at the base resolution (cached on disk)."""
    cache = _CACHE_DIR / f"mercator_ocean_{_BASE_RES}.npy"
    if cache.exists():
        flat = np.unpackbits(np.load(cache))[: _BASE_RES * _BASE_RES]
        return flat.reshape(_BASE_RES, _BASE_RES).astype(bool)

    from global_land_mask import globe

    centers = (np.arange(_BASE_RES) + 0.5) / _BASE_RES
    lats, _ = inverse_mercator(np.zeros_like(centers), centers)
    _, lons = inverse_mercator(centers, np.zeros_like(centers))
    ocean = np.empty((_BASE_RES, _BASE_RES), dtype=bool)
    for row, lat in enumerate(lats):
        ocean[row] = globe.is_ocean(np.full(_BASE_RES, lat), lons.copy())
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, np.packbits(ocean))
    return ocean


@functools.lru_cache(maxsize=len(_PYRAMID))
def _pyramid_level(res: int) -> np.ndarray:
    world = np.where(_ocean_bits(), 1.0, -1.0).astype(np.float32)
    if res == _BASE_RES:
        return world
    return cv2.resize(world, (res, res), interpolation=cv2.INTER_AREA)


def world_mask(width: int, height: int) -> np.ndarray:
    """The world's water mask (+1 water, -1 land) resized to ``width`` x ``height`` pixels."""
    source_res = next((r for r in _PYRAMID if r >= max(width, height)), _BASE_RES)
    source = _pyramid_level(source_res)
    interp = cv2.INTER_AREA if max(width, height) <= source_res else cv2.INTER_LINEAR
    return cv2.resize(source, (width, height), interpolation=interp)


def _paste(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
    ch, cw = canvas.shape
    h, w = image.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, cw), min(y + h, ch)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = image[y0 - y : y1 - y, x0 - x : x1 - x]


def _parabola_peak(left: float, mid: float, right: float) -> float:
    denom = left - 2 * mid + right
    return 0.0 if denom >= 0 else float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def _best_placement(
    template: np.ndarray,
    world_w: int,
    world_h: int,
    origin: tuple[float, float],
    radius: tuple[int, int],
    wraps: bool,
) -> tuple[float, float, float]:
    """Slide the world around ``origin`` +/- ``radius`` (template pixels) under the template.

    Returns ``(score, origin_x, origin_y)`` for the best placement, with sub-pixel refinement.
    """
    th, tw = template.shape
    rx, ry = radius
    ox, oy = round(origin[0]), round(origin[1])
    canvas = np.full((th + 2 * ry, tw + 2 * rx), OFF_WORLD, dtype=np.float32)
    world = world_mask(world_w, world_h)
    x, y = rx + ox, ry + oy
    if wraps:
        first = -((x + world_w) // world_w)
        last = (canvas.shape[1] - x) // world_w
        for k in range(first, last + 1):
            _paste(canvas, world, x + k * world_w, y)
    else:
        _paste(canvas, world, x, y)

    result = cv2.matchTemplate(canvas, template, cv2.TM_CCORR) / template.size
    _, score, _, (u, v) = cv2.minMaxLoc(result)
    du = (
        _parabola_peak(result[v, u - 1], score, result[v, u + 1])
        if 0 < u < result.shape[1] - 1
        else 0.0
    )
    dv = (
        _parabola_peak(result[v - 1, u], score, result[v + 1, u])
        if 0 < v < result.shape[0] - 1
        else 0.0
    )
    return float(score), ox + rx - (u + du), oy + ry - (v + dv)


def _downsample(mask: np.ndarray, max_width: int) -> tuple[np.ndarray, float, float]:
    h, w = mask.shape
    if w <= max_width:
        return mask, 1.0, 1.0
    cw, ch = max_width, max(8, round(h * max_width / w))
    return cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_AREA), w / cw, h / ch


def locate_world(
    rgb: np.ndarray,
    *,
    min_world: float = 0.5,
    max_world: float = 8.0,
    wrap_modes: tuple[bool, ...] = (False, True),
) -> MapProjection:
    """Find the Web Mercator projection of a map screenshot.

    ``min_world``/``max_world`` bound the world width as a multiple of the screenshot width.
    Raises :class:`MapNotFound` when there is not enough water/land contrast to match.
    """
    mask = water_mask(rgb)
    h, w = mask.shape
    water = float((mask > 0).mean())
    if not 0.03 < water < 0.97:
        raise MapNotFound(f"only {water:.0%} of the map looks like water; zoom the map out")

    # Coarse: every scale on a ~4% grid, every translation.
    small, sx, sy = _downsample(mask, 160)
    th, tw = small.shape
    best = None
    for world_full in np.geomspace(min_world * w, max_world * w, 72):
        ww, wh = max(8, round(world_full / sx)), max(8, round(world_full / sy))
        for wraps in wrap_modes:
            if wraps:
                origin, radius = (0.0, (th - wh) / 2), (ww // 2 + 1, (th + wh) // 2 + 1)
            else:
                origin = ((tw - ww) / 2, (th - wh) / 2)
                radius = ((tw + ww) // 2 + 1, (th + wh) // 2 + 1)
            score, ox, oy = _best_placement(small, ww, wh, origin, radius, wraps)
            if best is None or score > best.score:
                best = MapProjection(ww * sx, ox * sx, oy * sy, wraps, score)

    # Fine: search scale around the candidate, keeping the view centre's location fixed.
    prev_px = max(sx, sy)
    for span, step, max_width in ((0.06, 0.005, 400), (0.008, 0.0008, 1000)):
        template, sx, sy = _downsample(mask, max_width)
        th, tw = template.shape
        radius_px = max(3, round(2.5 * prev_px / min(sx, sy)))
        cx, cy = (w / 2 - best.origin_x) / best.world_px, (h / 2 - best.origin_y) / best.world_px
        refined = best
        for rel in np.arange(-span, span + step / 2, step):
            world_full = best.world_px * (1 + rel)
            ww, wh = round(world_full / sx), round(world_full / sy)
            origin = ((w / 2 - cx * world_full) / sx, (h / 2 - cy * world_full) / sy)
            score, ox, oy = _best_placement(
                template, ww, wh, origin, (radius_px, radius_px), best.wraps
            )
            if score > refined.score or refined is best:
                refined = MapProjection(ww * sx, ox * sx, oy * sy, best.wraps, score)
        best, prev_px = refined, max(sx, sy)

    return best
