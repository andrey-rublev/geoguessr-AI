"""Read the real location off OpenGuessr's result screen, using only pixels.

After a guess, the game shows a map with our red pin, a black flag on the answer, and a
line between them. The map is Web Mercator, so once its projection is known the flag's
pixel gives the answer. The view is usually zoomed in too far to recognise coastlines (see
:mod:`.mapcal`), so the reader zooms out one wheel notch at a time around a fixed point.
Every notch halves the map's scale, which the markers confirm as they draw together, so
the flag can still be measured at the most zoomed-in level where it stands clear of the pin.
The notch that reaches the map's widest view goes only part of the way, which the markers
measure too, and the first view is always at a whole zoom level, which puts right a step
counted wrongly.

In a multiplayer room the result map shows every player's pin, all alike, so there the reader
takes ours to be the one the recognised map puts where we clicked, and measures the flag on
that map alone. And it shows for only a few seconds, before the standings, whatever the host
does: so there the reader first takes all the views it can, as fast as the map zooms, and works
them out afterwards.

An answer is only returned after checks: the recognised map must put our pin where we
clicked, at a whole zoom level if it is the first view, and measurements from more zoomed-in
levels must agree with the flag's offset from our pin on it. When in doubt, the reader
returns no answer rather than a wrong one.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field

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
MAX_PINS = 12
"""The most pins looked for on a multiplayer result map, one per player."""
MATCH_WIDTH = 960
"""Result screenshots are shrunk to this width to recognise the world map quickly."""
WHOLE_STEP = 0.25
"""A zoom step the markers measure within this many halvings of a whole number is taken as
whole. Further off, it is the notch that reached the map's widest view, which went 0.62 of a
halving: counted as one, it put 25 answers 1.3 times too far from our pin, one near
Bucharest in Turkey."""
TILE = 256
"""The world's width in pixels at zoom level 0 and 100% scale. The first result view is at a
whole zoom level, so its world is this, at the markers' scale, times a power of two: 96% of
the 2,474 answers read by 24 September put it within 0.005 of a halving of one."""
RESULT_SHOWS_S = 8.0
"""How long a multiplayer result map may take to show once the round has ended."""
LOOK_S = 0.1
"""Seconds between looks while waiting for a multiplayer result map to show, or to stop moving."""
ZOOM_SETTLE_S = 1.0
"""How long a notch's zoom may take on a multiplayer result map, whose tiles aren't waited for."""
PIN_RED, PIN_PIXELS = 0.8, 150
"""A pin on the map is at least this many pixels (at 100% scale) at least this red: a quick look
for whether the map still shows, which finding the markers takes too long for between notches."""
RED_STILL = 0.002
"""How little the map's redness may change between looks for its pins to have stopped moving."""
WHOLE_ZOOM_SLACK = 0.1
"""Halvings a first view may be off a whole zoom level before it is put on one. The markers'
scale that the levels are reckoned from is measured to about 1% (1.243 for 1.25), or 0.01.
A world recognised on the first view this far off one is a wrong fit: two in 900 rounds by
30 September were, both 0.38 off, which put one answer near Tehran instead of in Syria."""


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
    zoomed_out: list[Image.Image] = field(default_factory=list)
    """When there is no answer, the result screen after each notch it was zoomed out, to see why."""


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
    found = find_markers(shade, name, scale, most=1)
    return found[0] if found else Marker(0.0, 0.0, -1.0)


def find_markers(
    shade: np.ndarray, name: str, scale: float, at_least: float = -1.0, most: int = MAX_PINS
) -> list[Marker]:
    """The best matches for a marker (see :func:`find_marker`) scoring ``at_least``, best first,
    one to a place: up to ``most`` of them, like every player's pin on a result map."""
    template = _scaled_template(name, scale)
    th, tw = template.shape
    if th > shade.shape[0] or tw > shade.shape[1]:
        return []
    blurred = cv2.GaussianBlur(shade, (0, 0), BLUR * scale)
    result = cv2.matchTemplate(blurred, template, cv2.TM_CCOEFF_NORMED)
    # Featureless windows would otherwise score a meaningless perfect match.
    result[_window_std(blurred, th, tw) < 0.05] = -1.0
    base, (ax, ay) = MARKERS[name]
    sx, sy = tw / base.shape[1], th / base.shape[0]
    found: list[Marker] = []
    while len(found) < most:
        _, score, _, (x, y) = cv2.minMaxLoc(result)
        if score < at_least or (found and score <= -1.0):
            break
        found.append(Marker(x + (ax + 0.5) * sx - 0.5, y + (ay + 0.5) * sy - 0.5, float(score)))
        # The same marker matches again a pixel or two off: look elsewhere for the next.
        result[max(y - th // 2, 0) : y + th // 2 + 1, max(x - tw // 2, 0) : x + tw // 2 + 1] = -1.0
    return found


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


def _zoom_off(world_px: float, scale: float) -> float:
    """Halvings ``world_px`` is off the nearest whole zoom level, at the markers' ``scale``."""
    levels = math.log2(world_px / (TILE * scale))
    return levels - round(levels)


def _whole_zoom(world_px: float, scale: float) -> float:
    """The width of the world at the whole zoom level nearest ``world_px``, unless it is within
    how well the markers' scale is measured of one already."""
    off = _zoom_off(world_px, scale)
    return world_px if abs(off) <= WHOLE_ZOOM_SLACK else world_px / 2.0**off


@dataclass
class _Level:
    rgb: np.ndarray
    pin: Marker | None
    flag: Marker | None
    halvings: float
    """How many times the map's scale had halved since the first level."""
    pins: list[Marker] = field(default_factory=list)
    """On a multiplayer result map, every pin, which ours is among."""


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

    def read(
        self,
        placed_lat: float,
        placed_lon: float,
        *,
        crowded: bool = False,
        abort: Callable[[], bool] | None = None,
    ) -> Reading:
        """Find the answer; ``placed_lat``/``placed_lon`` is where our pin was dropped. With
        ``crowded``, other players' pins show too, and the map is about to go. ``abort`` says
        when to stop zooming out, as when it has gone and the next round begun."""
        self.controls.move(self.region.center)
        reading = Reading(None, "the result map never lined up with the world map")
        levels: list[_Level] = []
        step = 1.0  # scale halvings per whole notch, re-measured whenever the markers allow
        views = self._hurried_views if crowded else self._settled_views
        for rgb in views(reading, abort):
            if not levels:
                reading.screenshot = Image.fromarray(rgb)
            elif np.abs(rgb.astype(np.int16) - levels[-1].rgb).mean() < 0.5:
                break  # the map won't zoom out any further
            pin, flag = self._markers(rgb)
            pins = []
            if crowded:  # which pin is ours only the recognised map can say
                pins, pin = self._pins(rgb), None
            halvings = 0.0
            if levels:
                measured = self._halvings_between(levels[-1], pin, flag)
                if measured is not None and measured.is_integer():
                    step = measured
                halvings = levels[-1].halvings + (step if measured is None else measured)
            levels.append(_Level(rgb, pin, flag, halvings, pins))
            projection = self._locate(rgb)
            # The first view is at a whole zoom level, so a world recognised on it at another
            # size is a wrong fit that only happened to put our pin where it is: look further out.
            if projection and len(levels) == 1 and self.scale:
                if abs(_zoom_off(projection.world_px, self.scale)) > WHOLE_ZOOM_SLACK:
                    projection = None
            if projection and crowded and self.scale:  # no pin seen yet, none to tell ours by
                levels[-1].pin = self._our_pin(projection, levels[-1], placed_lat, placed_lon)
                self._trace_back(levels, projection.to_pixel(placed_lat, placed_lon, rgb.shape[1]))
            if projection and self._shows_our_pin(projection, levels[-1], placed_lat, placed_lon):
                reading.answer, reading.problem = self._answer(
                    levels, projection, placed_lat, placed_lon
                )
                break
        if reading.answer is None:
            reading.zoomed_out = [Image.fromarray(level.rgb) for level in levels[1:]]
        return reading

    def _settled_views(self, reading: Reading, abort: Callable[[], bool] | None):
        """Each view of the map, zooming out a notch at a time as each is worked out."""
        for zoom_outs in range(self.max_zoom_outs + 1):
            if abort is not None and abort():
                reading.problem = "the result map went before the answer was read"
                return
            if zoom_outs:
                self.controls.scroll(self.region.center, -1)
            yield self._settled_grab()

    def _hurried_views(
        self, reading: Reading, abort: Callable[[], bool] | None
    ) -> list[np.ndarray]:
        """All the views of the map there is time for before it goes, with no working out on
        the way: as soon as it shows, then a notch further out each, without waiting for tiles
        until the last, at the map's widest, which is the one to recognise."""
        first = self._first_view(abort)
        if first is None:
            reading.problem = "the result map never showed"
            return []
        views = [first]
        for _ in range(self.max_zoom_outs):
            if abort is not None and abort():
                reading.problem = "the result map went before the answer was read"
                return views
            self.controls.scroll(self.region.center, -1)
            rgb, red = self._zoomed_grab()
            quick = (red > PIN_RED).sum() >= PIN_PIXELS * (self.scale or 1.0) ** 2
            if not quick and not self._shows_markers(rgb):  # only then is it worth the time
                reading.problem = "the result map went before the answer was read"
                return views
            if np.abs(rgb.astype(np.int16) - views[-1]).mean() < 0.5:
                break  # the map won't zoom out any further
            views.append(rgb)
        rgb = self._settled_grab()
        if self._shows_markers(rgb):
            views[-1] = rgb
        return views

    def _first_view(self, abort: Callable[[], bool] | None) -> np.ndarray | None:
        """The result map as soon as it shows, once its tiles are in or about to go; None if it
        doesn't show, as when the round goes straight to the standings."""
        start = self.controls.now()
        while self.controls.now() - start < RESULT_SHOWS_S:
            if abort is not None and abort():
                return None
            rgb = np.asarray(self.screen.grab(self.region).convert("RGB"))
            if self._shows_markers(rgb):
                return self._tiles_in(rgb)
            self.controls.sleep(LOOK_S)
        return None

    def _tiles_in(self, rgb: np.ndarray) -> np.ndarray:
        """The map once it stops changing, for at most a few looks: the markers won't move."""
        for _ in range(self.settle_checks // 2):
            self.controls.sleep(0.25)
            current = np.asarray(self.screen.grab(self.region).convert("RGB"))
            if np.abs(current.astype(np.int16) - rgb).mean() < 1.0:
                return current
            rgb = current
        return rgb

    def _zoomed_grab(self) -> tuple[np.ndarray, np.ndarray]:
        """The map once its pins stop moving after a notch of the wheel, and its redness."""
        rgb = np.asarray(self.screen.grab(self.region).convert("RGB"))
        red = marker_shades(rgb)[0]
        start = self.controls.now()
        while self.controls.now() - start < ZOOM_SETTLE_S:
            self.controls.sleep(LOOK_S)
            rgb = np.asarray(self.screen.grab(self.region).convert("RGB"))
            before, red = red, marker_shades(rgb)[0]
            if np.abs(red - before).mean() < RED_STILL:
                break
        return rgb, red

    def _shows_markers(self, rgb: np.ndarray) -> bool:
        """Whether a pin or the flag shows, as they do while the result map does."""
        pin, flag = self._markers(rgb)
        return pin is not None or flag is not None

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

    def _pins(self, rgb: np.ndarray) -> list[Marker]:
        """Every pin on a result map, as on a multiplayer one, where all players' show."""
        if self.scale is None:
            return []
        red, _ = marker_shades(rgb)
        return find_markers(red, "pin", self.scale, MIN_SCORES["pin"])

    def _halvings_between(
        self, before: _Level, pin: Marker | None, flag: Marker | None
    ) -> float | None:
        """Scale halvings since the previous level, measured from how the markers drew closer:
        a whole number unless the map reached its widest view. None if they can't say."""
        if not (before.pin and before.flag and pin and flag):
            return None
        after = math.dist((pin.x, pin.y), (flag.x, flag.y))
        if after < APART * self.scale:
            return None
        drawn_in = math.dist((before.pin.x, before.pin.y), (before.flag.x, before.flag.y)) / after
        halvings = max(0.0, math.log2(drawn_in))
        whole = float(round(halvings))
        return whole if abs(halvings - whole) < WHOLE_STEP else halvings

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
        flag = level.flag
        beside = flag and [
            pin
            for pin in level.pins
            if math.dist((pin.x, pin.y), (flag.x, flag.y)) < APART * self.scale
        ]
        if level.pin:
            target, slack = level.pin, 0.0
        elif beside:  # a pin by the flag, among other players': judged as ours is, by itself
            target, slack = (
                min(beside, key=lambda pin: math.dist((pin.x, pin.y), (flag.x, flag.y))),
                0.0,
            )
        elif flag:  # after a close guess the flag is drawn over the pin
            target, slack = flag, APART * self.scale
        else:
            return False
        off = self._off(projection, target, lat, lon, level.rgb.shape[1])
        return off <= self._reach(projection, lat) + slack

    def _our_pin(
        self, projection: MapProjection, level: _Level, lat: float, lon: float
    ) -> Marker | None:
        """Of every pin on the level, the nearest to where the recognised map puts ours, if
        any is within reach of it."""
        width, reach = level.rgb.shape[1], self._reach(projection, lat)
        offs = [(self._off(projection, pin, lat, lon, width), pin) for pin in level.pins]
        near = [(off, pin) for off, pin in offs if off <= reach]
        return min(near, key=lambda found: found[0])[1] if near else None

    def _trace_back(self, levels: list[_Level], placed: tuple[float, float]) -> None:
        """Find our pin on the levels before the last, from where the last puts it: our pin
        there if it shows, or else ``placed``, where the recognised map puts where we clicked,
        as when the flag is drawn over it after a close guess. Each notch out drew the map in
        towards the middle, where the wheel turned, so on each level before, our pin was as
        many times further out: the pin nearest there is ours. Zoomed in, those levels can
        tell the flag from our pin, which the recognised map can't after a close guess."""
        last = levels[-1]
        h, w = last.rgb.shape[:2]
        cx, cy = w / 2, h / 2
        px, py = (last.pin.x, last.pin.y) if last.pin else placed
        for level in levels[:-1]:
            if level.pin is not None or not level.pins:
                continue
            grow = 2.0 ** (last.halvings - level.halvings)
            x, y = cx + (px - cx) * grow, cy + (py - cy) * grow
            near = min(level.pins, key=lambda pin: math.dist((pin.x, pin.y), (x, y)))
            if math.dist((near.x, near.y), (x, y)) <= (4 + APART) * self.scale * grow:
                level.pin = near

    @staticmethod
    def _off(
        projection: MapProjection, marker: Marker, lat: float, lon: float, width: int
    ) -> float:
        """Pixels between ``marker`` and where the recognised map puts (``lat``, ``lon``)."""
        x, y = projection.to_pixel(lat, lon, width)
        dx = x - marker.x
        if projection.wraps:
            dx = (dx + projection.world_px / 2) % projection.world_px - projection.world_px / 2
        return math.hypot(dx, y - marker.y)

    def _reach(self, projection: MapProjection, lat: float) -> float:
        """How far from where we clicked our pin may be found: a marker's misplacement, plus
        how far off our own record of it may be (see :data:`PLACED_ERROR_KM`)."""
        return 4 * self.scale + PLACED_ERROR_KM / _km_per_px(projection.world_px, lat)

    def _answer(
        self, levels: list[_Level], projection: MapProjection, lat: float, lon: float
    ) -> tuple[Answer | None, str]:
        last = levels[-1]
        h, w = last.rgb.shape[:2]
        mx, my = mercator_xy(lat, lon)
        # Levels before the last are at whole zoom levels, whatever the steps counted to it.
        first_world = _whole_zoom(projection.world_px * 2.0**last.halvings, self.scale)
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
            world = (
                projection.world_px if level is last else first_world / 2.0 ** round(level.halvings)
            )
            y = min(max(my + (flag.y - pin.y) / world, 0.0), 1.0)
            a_lat, a_lon = inverse_mercator(mx + (flag.x - pin.x) / world, y)
            estimates.append((zoom_outs, a_lat, _wrap_lon(a_lon), _km_per_px(world, a_lat)))
        # Measurements further in rest on the zoom steps counted since, so a miscounted step moves
        # them together. What counts none is the flag's offset from our pin on the recognised map,
        # or, where the two overlap there, the flag's place on it, which is only as good as the
        # map's fit: the pin had to be within as much of where we clicked.
        offset = estimates[-1] if estimates and estimates[-1][0] == len(levels) - 1 else None
        place = None
        if last.flag:
            a_lat, a_lon = projection.to_latlon(last.flag.x, last.flag.y)
            kmpp = _km_per_px(projection.world_px, a_lat)
            place = (len(levels) - 1, a_lat, _wrap_lon(a_lon), kmpp)
            estimates.append(place)
        if not estimates:
            return None, "couldn't see the flag on the result map"
        if len(estimates) == 1:
            zoom_outs, a_lat, a_lon, _ = estimates[0]
            return Answer(a_lat, a_lon, zoom_outs), ""

        def agree(e, o) -> bool:
            allowed = 3 * self.scale * max(e[3], o[3]) + PLACED_ERROR_KM
            if place is e or place is o:  # as far off as the pin was let be, beside the flag
                allowed += (4 + APART) * self.scale * place[3]
            return haversine_km(e[1], e[2], o[1], o[2]) <= allowed

        for e in estimates:
            judges = [place] if e is offset else [offset or place]
            if not any(judges):  # the flag is gone from the recognised map: any two will do
                judges = estimates
            if any(o is not None and o is not e and agree(e, o) for o in judges):
                return Answer(e[1], e[2], e[0]), ""
        return None, "the zoom levels disagree about where the flag is"
