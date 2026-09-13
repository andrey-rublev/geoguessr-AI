"""Read the real location off OpenGuessr's result screen, using only pixels.

After a guess, the game shows a map with our red pin, a black flag on the answer, and a
line between them. The map is Web Mercator, so once its projection is known the flag's
pixel gives the answer. The view is usually zoomed in too far to recognise coastlines (see
:mod:`.mapcal`), so the reader zooms out one wheel notch at a time around a fixed point.
Every notch halves the map's scale, which the markers confirm as they draw together, so
the flag can still be measured at the most zoomed-in level where it stands clear of the pin.

An answer is only returned after checks: the recognised map must put our pin where we
clicked, and measurements from different zoom levels must agree. When in doubt, the reader
returns no answer rather than a wrong one.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .config import Layout, Region
from .geo import EARTH_RADIUS_KM, haversine_km, inverse_mercator, mercator_xy
from .mapcal import MapNotFound, MapProjection, locate_world

# The game's two markers at 100% browser scale, traced from real result screens: how red each
# pixel of the pin is, and how dark each pixel of the flag is, from "." (not at all) to "#".
_SHADES = ".-+*#"
PIN_ART = """
.........-+###*+.........
.......+#########*-......
.....-#############*.....
....+###############*....
...+##################...
...######*+++++*######*..
..*#####*-......+######-.
.-#####*.........+######.
.*#####-..........*#####-
.#####*...........-#####*
.#####-............######
.#####.............######
-#####.............######
.#####-............######
.#####*...........-######
.######...........*######
.######*.........+######+
.+######*-......+#######.
..########*+-++########+.
..-####################..
...*##################-..
....##################...
....+################....
.....###############+....
......##############.....
......*############......
.......###########*......
.......-##########.......
........#########-.......
.........#######*........
.........-######.........
..........#####+.........
...........###*..........
............*+...........
"""
FLAG_ART = """
......................
...+**+..............*
..*******..*.#..-***#+
.***********-*******#*
-*********************
.*#**.-*************+*
.###+...-***********+*
.###+......---...***++
.###*............***-.
.###*............***-.
.###*............***+.
.###*............##*+.
.###*............###+.
.####............###+.
.####............###+.
.####++***+-.....###+.
.#############+*####-.
.###################..
.##################-..
.#####*+-*########-...
.####*................
.####*................
.*####-...............
.*####+...............
.*#####*..............
.+#####-..............
.+####+...............
.-####................
..###+................
...++.................
"""


def _template(art: str) -> np.ndarray:
    rows = art.strip().splitlines()
    return np.array([[_SHADES.index(c) / 4 for c in row] for row in rows], dtype=np.float32)


MARKERS = {
    "pin": (_template(PIN_ART), (13.5, 33.5)),  # anchored at the pin's tip
    "flag": (_template(FLAG_ART), (4.5, 29.5)),  # anchored at the foot of the flagpole
}
MIN_SCORES = {"pin": 0.65, "flag": 0.72}
"""Template-match scores that count as seeing a marker. Real markers score 0.8-1.0; the
closest look-alikes on real result screens (labels, borders, buttons) scored up to 0.6."""
BLUR = 0.7
"""Screenshot and templates are both blurred this much (in 100%-scale pixels), which makes
matching forgiving of how the browser resamples the icons."""
MIN_MAP_SCORE = 0.6
PLACED_ERROR_KM = 20.0
"""How far our own record of the pin may be off (minimap matching, whole-pixel clicks)."""
APART = 10.0
"""Markers closer than this (in 100%-scale pixels) overlap and can't be measured apart."""
MATCH_WIDTH = 960
"""Result screenshots are shrunk to this width to recognise the world map quickly."""


@dataclass(frozen=True)
class Marker:
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class Answer:
    lat: float
    lon: float
    zoom_outs: int
    """Wheel notches the map had been zoomed out by when the flag was measured."""


@dataclass
class Reading:
    answer: Answer | None
    problem: str = ""
    """Why there is no answer, when there isn't one."""
    screenshot: Image.Image | None = None
    """The result screen as it first appeared."""


def result_region(layout: Layout) -> Region:
    """The part of the result screen to read: everything the view and the expanded map cover."""
    v, m = layout.view, layout.map_region
    left, top = min(v.left, m.left), min(v.top, m.top)
    right = max(v.left + v.width, m.left + m.width)
    bottom = max(v.top + v.height, m.top + m.height)
    return Region(left, top, right - left, bottom - top)


def marker_shades(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(redness, darkness)`` of every pixel, in [0, 1]: what the pin and flag are made of."""
    px = rgb[..., :3].astype(np.float32)
    red = np.clip((px[..., 0] - np.maximum(px[..., 1], px[..., 2]) - 40) / 120, 0, 1)
    dark = np.clip((160 - px.max(axis=2)) / 110, 0, 1)
    return red, dark


@functools.lru_cache(maxsize=32)
def _scaled_template(name: str, scale: float) -> np.ndarray:
    template, _ = MARKERS[name]
    h, w = template.shape
    size = (max(3, round(w * scale)), max(3, round(h * scale)))
    resized = cv2.resize(template, size, interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(resized, (0, 0), BLUR * scale)


def _window_std(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Standard deviation of every ``h`` x ``w`` window, laid out like ``cv2.matchTemplate``."""
    sums = cv2.integral(image, sdepth=cv2.CV_64F)
    squares = cv2.integral(image * image, sdepth=cv2.CV_64F)

    def window(table: np.ndarray) -> np.ndarray:
        return table[h:, w:] - table[:-h, w:] - table[h:, :-w] + table[:-h, :-w]

    mean = window(sums) / (h * w)
    return np.sqrt(np.clip(window(squares) / (h * w) - mean * mean, 0, None))


def find_marker(shade: np.ndarray, name: str, scale: float) -> Marker:
    """Best match for the ``"pin"`` (in redness) or ``"flag"`` (in darkness) drawn at ``scale``."""
    template = _scaled_template(name, scale)
    th, tw = template.shape
    if th > shade.shape[0] or tw > shade.shape[1]:
        return Marker(0.0, 0.0, -1.0)
    blurred = cv2.GaussianBlur(shade, (0, 0), BLUR * scale)
    result = cv2.matchTemplate(blurred, template, cv2.TM_CCOEFF_NORMED)
    # Featureless windows would otherwise score a meaningless perfect match.
    result[_window_std(blurred, th, tw) < 0.05] = -1.0
    _, score, _, (x, y) = cv2.minMaxLoc(result)
    base, (ax, ay) = MARKERS[name]
    sx, sy = tw / base.shape[1], th / base.shape[0]
    return Marker(x + (ax + 0.5) * sx - 0.5, y + (ay + 0.5) * sy - 0.5, float(score))


def find_scale(red: np.ndarray) -> float | None:
    """How big the markers are drawn relative to 100% browser scale, judged by the pin."""

    def best_of(scales) -> tuple[float, float]:
        scores = [find_marker(red, "pin", float(s)).score for s in scales]
        i = int(np.argmax(scores))
        return scores[i], float(scales[i])

    _, coarse = best_of(np.geomspace(0.75, 3.0, 21))
    score, scale = best_of(np.round(coarse * np.linspace(0.96, 1.04, 5), 3))
    return scale if score >= MIN_SCORES["pin"] else None


def _km_per_px(world_px: float, lat: float) -> float:
    return 2 * math.pi * EARTH_RADIUS_KM * math.cos(math.radians(lat)) / world_px


def _wrap_lon(lon: float) -> float:
    return (lon + 180.0) % 360.0 - 180.0


@dataclass
class _Level:
    rgb: np.ndarray
    pin: Marker | None
    flag: Marker | None
    halvings: int
    """How many times the map's scale had halved since the first level."""


class ResultReader:
    """Reads answers off result screens, remembering the markers' size between rounds."""

    def __init__(
        self, screen, controls, region: Region, *, max_zoom_outs: int = 15, settle_checks: int = 12
    ) -> None:
        self.screen = screen
        self.controls = controls
        self.region = region
        self.max_zoom_outs = max_zoom_outs
        self.settle_checks = settle_checks
        self.scale: float | None = None

    def read(self, placed_lat: float, placed_lon: float) -> Reading:
        """Find the answer; ``placed_lat``/``placed_lon`` is where our pin was dropped."""
        centre = self.region.center
        self.controls.move(centre)
        reading = Reading(None, "the result map never lined up with the world map")
        levels: list[_Level] = []
        step = 1  # scale halvings per notch, re-measured whenever the markers allow
        for zoom_outs in range(self.max_zoom_outs + 1):
            if zoom_outs:
                self.controls.scroll(centre, -1)
            rgb = self._settled_grab()
            if not levels:
                reading.screenshot = Image.fromarray(rgb)
            elif np.abs(rgb.astype(np.int16) - levels[-1].rgb).mean() < 0.5:
                break  # the map won't zoom out any further
            pin, flag = self._markers(rgb)
            halvings = 0
            if levels:
                step = self._halvings_between(levels[-1], pin, flag, step)
                halvings = levels[-1].halvings + step
            levels.append(_Level(rgb, pin, flag, halvings))
            projection = self._locate(rgb)
            if projection and self._shows_our_pin(projection, levels[-1], placed_lat, placed_lon):
                reading.answer, reading.problem = self._answer(
                    levels, projection, placed_lat, placed_lon
                )
                break
        return reading

    def _settled_grab(self) -> np.ndarray:
        """Grab the result map once tiles have loaded and animations have finished."""
        self.controls.sleep(0.5)
        previous = np.asarray(self.screen.grab(self.region).convert("RGB"))
        for _ in range(self.settle_checks):
            self.controls.sleep(0.25)
            current = np.asarray(self.screen.grab(self.region).convert("RGB"))
            if np.abs(current.astype(np.int16) - previous).mean() < 1.0:
                return current
            previous = current
        return previous

    def _markers(self, rgb: np.ndarray) -> tuple[Marker | None, Marker | None]:
        red, dark = marker_shades(rgb)
        if self.scale is None:
            self.scale = find_scale(red)
            if self.scale is None:
                return None, None
        found = {
            name: find_marker(shade, name, self.scale)
            for name, shade in (("pin", red), ("flag", dark))
        }
        pin, flag = (m if m.score >= MIN_SCORES[name] else None for name, m in found.items())
        return pin, flag

    def _halvings_between(
        self, before: _Level, pin: Marker | None, flag: Marker | None, step: int
    ) -> int:
        """Scale halvings since the previous level, measured from how the markers drew closer."""
        if not (before.pin and before.flag and pin and flag):
            return step
        after = math.dist((pin.x, pin.y), (flag.x, flag.y))
        if after < APART * self.scale:
            return step
        return max(
            0,
            round(
                math.log2(
                    math.dist((before.pin.x, before.pin.y), (before.flag.x, before.flag.y)) / after
                )
            ),
        )

    def _locate(self, rgb: np.ndarray) -> MapProjection | None:
        h, w = rgb.shape[:2]
        small = rgb
        if w > MATCH_WIDTH:
            size = (MATCH_WIDTH, round(h * MATCH_WIDTH / w))
            small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        try:
            p = locate_world(small, max_world=3.0)
        except MapNotFound:
            return None
        if p.score < MIN_MAP_SCORE:
            return None
        fx, fy = small.shape[1] / w, small.shape[0] / h
        return MapProjection(
            p.world_px / fx,
            (p.origin_x + 0.5) / fx - 0.5,
            (p.origin_y + 0.5) / fy - 0.5,
            p.wraps,
            p.score,
        )

    def _shows_our_pin(
        self, projection: MapProjection, level: _Level, lat: float, lon: float
    ) -> bool:
        """Whether the recognised map puts our pin where it really is."""
        if level.pin:
            target, slack = level.pin, 0.0
        elif level.flag:  # after a close guess the flag is drawn over the pin
            target, slack = level.flag, APART * self.scale
        else:
            return False
        x, y = projection.to_pixel(lat, lon, level.rgb.shape[1])
        dx = x - target.x
        if projection.wraps:
            dx = (dx + projection.world_px / 2) % projection.world_px - projection.world_px / 2
        tolerance = 4 * self.scale + slack + PLACED_ERROR_KM / _km_per_px(projection.world_px, lat)
        return math.hypot(dx, y - target.y) <= tolerance

    def _answer(
        self, levels: list[_Level], projection: MapProjection, lat: float, lon: float
    ) -> tuple[Answer | None, str]:
        last = levels[-1]
        h, w = last.rgb.shape[:2]
        mx, my = mercator_xy(lat, lon)
        estimates = []  # (zoom_outs, lat, lon, km per pixel), most zoomed-in first
        for zoom_outs, level in enumerate(levels):
            pin, flag = level.pin, level.flag
            if (
                not (pin and flag)
                or math.dist((pin.x, pin.y), (flag.x, flag.y)) < APART * self.scale
            ):
                continue
            if not all(2 <= m.x < w - 2 and 2 <= m.y < h - 2 for m in (pin, flag)):
                continue
            # The flag's offset from our pin, at this level's scale, starting from where we clicked.
            world = projection.world_px * 2.0 ** (last.halvings - level.halvings)
            y = min(max(my + (flag.y - pin.y) / world, 0.0), 1.0)
            a_lat, a_lon = inverse_mercator(mx + (flag.x - pin.x) / world, y)
            estimates.append((zoom_outs, a_lat, _wrap_lon(a_lon), _km_per_px(world, a_lat)))
        if last.flag:
            a_lat, a_lon = projection.to_latlon(last.flag.x, last.flag.y)
            kmpp = _km_per_px(projection.world_px, a_lat)
            estimates.append((len(levels) - 1, a_lat, _wrap_lon(a_lon), kmpp))
        if not estimates:
            return None, "couldn't see the flag on the result map"
        if len(estimates) == 1:
            zoom_outs, a_lat, a_lon, _ = estimates[0]
            return Answer(a_lat, a_lon, zoom_outs), ""
        for e in estimates:
            for o in estimates:
                allowed = 3 * self.scale * max(e[3], o[3]) + PLACED_ERROR_KM
                if o is not e and haversine_km(e[1], e[2], o[1], o[2]) <= allowed:
                    return Answer(e[1], e[2], e[0]), ""
        return None, "the zoom levels disagree about where the flag is"
