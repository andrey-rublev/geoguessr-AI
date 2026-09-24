import math

import cv2
import numpy as np
import pytest
from PIL import Image

from geoguessr_ai.config import Layout, Point, Region
from geoguessr_ai.geo import haversine_km, mercator_xy
from geoguessr_ai.mapcal import MapProjection, world_mask
from geoguessr_ai.result import (
    MARKERS,
    ResultReader,
    find_marker,
    find_scale,
    marker_shades,
    result_region,
)

WATER, LAND, OFF_WORLD = (90, 200, 235), (245, 240, 229), (229, 227, 223)
OCEAN = world_mask(4096, 4096) > 0
MADRID, LISBON = (40.42, -3.70), (38.72, -9.14)
REGION = Region(100, 200, 1200, 500)
PIN_RED, FLAG_BLACK = (235, 35, 35), (15, 15, 15)


def paint(rgb, name, x, y, colour, scale=1.5):
    """Draw a marker from its template, anchored at (x, y), the way the detector expects it."""
    template, (ax, ay) = MARKERS[name]
    th, tw = template.shape
    alpha = cv2.resize(template, (round(tw * scale), round(th * scale)))
    left, top = round(x - (ax + 0.5) * scale + 0.5), round(y - (ay + 0.5) * scale + 0.5)
    y0, x0 = max(top, 0), max(left, 0)
    y1 = min(top + alpha.shape[0], rgb.shape[0])
    x1 = min(left + alpha.shape[1], rgb.shape[1])
    if y1 > y0 and x1 > x0:
        a = alpha[y0 - top : y1 - top, x0 - left : x1 - left, None]
        rgb[y0:y1, x0:x1] = rgb[y0:y1, x0:x1] * (1 - a) + np.array(colour) * a


def fitted(world, a=MADRID, b=LISBON):
    """The view the game opens with: both places centred at the given zoom."""
    (ax, ay), (bx, by) = mercator_xy(*a), mercator_xy(*b)
    cx, cy = (ax + bx) / 2, (ay + by) / 2
    return MapProjection(world, REGION.width / 2 - cx * world, REGION.height / 2 - cy * world, True)


class FakeResultScreen:
    """A result map at any zoom, drawn from the land mask, with our pin and the answer's flag."""

    def __init__(self, projection, pin=MADRID, flag=LISBON, show_flag=True):
        self.projection, self.pin, self.flag, self.show_flag = projection, pin, flag, show_flag

    def grab(self, region):
        assert region == REGION
        p = self.projection
        ys, xs = np.mgrid[0 : region.height, 0 : region.width]
        mx, my = (xs - p.origin_x) / p.world_px, (ys - p.origin_y) / p.world_px
        rows = np.clip(np.floor(my * 4096), 0, 4095).astype(int)
        cols = np.floor(mx * 4096).astype(np.int64) % 4096
        rgb = np.where(OCEAN[rows, cols][..., None], WATER, LAND).astype(np.float32)
        rgb[(my < 0) | (my >= 1)] = OFF_WORLD
        paint(rgb, "pin", *p.to_pixel(*self.pin, region.width), PIN_RED)
        if self.show_flag:  # the game draws the flag over the pin
            paint(rgb, "flag", *p.to_pixel(*self.flag, region.width), FLAG_BLACK)
        return Image.fromarray(rgb.round().astype(np.uint8))


class FakeControls:
    """Each wheel notch halves the map's scale around the cursor, like Leaflet does."""

    def __init__(self, screen):
        self.screen, self.notches = screen, 0

    def sleep(self, seconds):
        pass

    def move(self, p):
        pass

    def scroll(self, p, clicks):
        assert clicks < 0, "the reader only ever zooms out"
        for _ in range(-clicks):
            cx, cy = p.x - REGION.left, p.y - REGION.top
            q = self.screen.projection
            self.screen.projection = MapProjection(
                q.world_px / 2, cx - (cx - q.origin_x) / 2, cy - (cy - q.origin_y) / 2, True
            )
            self.notches += 1


def test_markers_are_found_at_any_display_scale():
    rgb = np.full((300, 400, 3), LAND, np.float32)
    paint(rgb, "pin", 150.0, 200.0, PIN_RED, scale=2.0)
    paint(rgb, "flag", 300.0, 120.0, FLAG_BLACK, scale=2.0)
    red, dark = marker_shades(rgb.astype(np.uint8))

    scale = find_scale(red)

    assert scale == pytest.approx(2.0, rel=0.05)
    pin, flag = find_marker(red, "pin", scale), find_marker(dark, "flag", scale)
    assert pin.score > 0.9 and math.dist((pin.x, pin.y), (150, 200)) < 2.5
    assert flag.score > 0.9 and math.dist((flag.x, flag.y), (300, 120)) < 2.5


def test_reads_the_answer_after_zooming_out():
    screen = FakeResultScreen(fitted(24_576))
    controls = FakeControls(screen)

    reading = ResultReader(screen, controls, REGION).read(*MADRID)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 10
    assert controls.notches >= 2 and reading.answer.zoom_outs == 0
    assert reading.screenshot.size == (REGION.width, REGION.height)


def test_close_guess_is_measured_before_the_markers_overlap():
    near = (40.52, -3.52)  # about 19 km from Madrid
    screen = FakeResultScreen(fitted(786_432, MADRID, near), pin=near, flag=MADRID)

    reading = ResultReader(screen, FakeControls(screen), REGION).read(*near)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *MADRID) < 3


def test_no_answer_when_the_flag_never_shows():
    screen = FakeResultScreen(fitted(6_144), show_flag=False)
    reading = ResultReader(screen, FakeControls(screen), REGION, max_zoom_outs=2).read(*MADRID)
    assert reading.answer is None and "flag" in reading.problem


def test_no_answer_when_the_map_disagrees_with_our_pin():
    screen = FakeResultScreen(fitted(6_144))
    paris = (48.86, 2.35)  # we think we clicked here, but the pin is in Madrid
    reading = ResultReader(screen, FakeControls(screen), REGION, max_zoom_outs=2).read(*paris)
    assert reading.answer is None and "lined up" in reading.problem


def test_result_region_covers_the_view_and_the_expanded_map():
    layout = Layout(
        view=Region(14, 262, 1405, 513),
        map_hover=Point(1626, 869),
        map_region=Region(768, 450, 1039, 522),
        guess_button=Point(1626, 1150),
        continue_button=Point(987, 1141),
    )
    assert result_region(layout) == Region(14, 262, 1793, 710)
