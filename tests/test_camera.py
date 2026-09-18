import math

import cv2
import numpy as np
import pytest
from PIL import Image

from geoguessr_ai.camera import (
    GROUND_FAR,
    GROUND_SCALE,
    GROUND_SIDE,
    Camera,
    default_camera,
    ground_view,
    measure,
)
from geoguessr_ai.config import Point, Region

VIEW = Region(20, 120, 900, 420)
TRUE = Camera(cx=640.0, cy=300.0, f=440.0)
"""Like OpenGuessr's: Street View's frame is centred right of and above the calibrated view."""


def test_directions_of_pixels():
    assert TRUE.direction(640, 300, heading=90) == pytest.approx((90, 0))
    assert TRUE.direction(640 + 440, 300, heading=90) == pytest.approx((135, 0))
    assert TRUE.direction(640, 300 - 440, heading=0)[1] == pytest.approx(45)
    assert TRUE.direction(640, 300, heading=200, pitch=45) == pytest.approx((200, 45))
    # Tilted up, a spot off to the side lies further round the compass than it looks.
    azimuth, height = TRUE.direction(640 + 440, 300, heading=0, pitch=45)
    assert azimuth == pytest.approx(math.degrees(math.atan2(1, math.cos(math.radians(45)))))
    assert height == pytest.approx(30)


def test_drag_rows_and_tilts_agree():
    for start, degrees in ((150, 45), (500, -30), (300, 10)):
        end = TRUE.row_for_tilt(start, degrees)
        assert TRUE.tilt(start, end) == pytest.approx(degrees)
    assert TRUE.tilt(150, 400) > 0  # dragging down looks up


def test_default_camera_centres_the_view():
    camera = default_camera(VIEW)
    assert (camera.cx, camera.cy) == (470, 330) and not camera.measured
    assert camera.direction(VIEW.left, 330, heading=0)[0] == pytest.approx(360 - 52.5)


def test_the_road_seen_from_above_comes_out_square():
    """A patch painted on the road, 1 to 2 camera heights right and 4 to 6 ahead, is drawn
    squashed in the view but comes out as a rectangle from above."""
    ys, xs = np.mgrid[VIEW.top : VIEW.top + VIEW.height, VIEW.left : VIEW.left + VIEW.width]
    below = np.maximum(ys - TRUE.cy, 1e-6)
    ahead, side = TRUE.f / below, (xs - TRUE.cx) / below
    painted = (ys > TRUE.cy) & (1 <= side) & (side <= 2) & (4 <= ahead) & (ahead <= 6)
    view = np.zeros((VIEW.height, VIEW.width, 3), np.uint8)
    view[painted] = 255

    above = np.asarray(ground_view(Image.fromarray(view), Point(VIEW.left, VIEW.top), TRUE))

    rows, cols = np.nonzero(above[..., 0] > 128)
    expected_cols = ((1 + GROUND_SIDE) * GROUND_SCALE, (2 + GROUND_SIDE) * GROUND_SCALE)
    expected_rows = ((GROUND_FAR - 6) * GROUND_SCALE, (GROUND_FAR - 4) * GROUND_SCALE)
    assert (cols.min(), cols.max()) == pytest.approx(expected_cols, abs=3)
    assert (rows.min(), rows.max()) == pytest.approx(expected_rows, abs=3)


def test_no_road_from_above_in_a_view_of_the_sky():
    sky = Image.new("RGB", (VIEW.width, 100))
    assert ground_view(sky, Point(VIEW.left, 100), TRUE) is None


def panorama(seed=0):
    """A made-up 360 degree panorama full of corners to match, as Street View's image."""
    rng = np.random.default_rng(seed)
    pano = np.full((1024, 2048), 128, np.uint8)
    for _ in range(3000):
        x, y = rng.integers(0, 2048), rng.integers(0, 1024)
        w, h = rng.integers(8, 60, size=2)
        cv2.rectangle(pano, (x, y), (x + w, y + h), int(rng.integers(0, 256)), -1)
    return pano


def render(pano, camera, heading, view=VIEW):
    """What ``view`` shows of the panorama with the camera level, facing ``heading``."""
    ys, xs = np.mgrid[view.top : view.top + view.height, view.left : view.left + view.width]
    right, up = (xs - camera.cx) / camera.f, (camera.cy - ys) / camera.f
    azimuth = np.degrees(np.arctan2(right, 1.0)) + heading
    height = np.degrees(np.arctan2(up, np.hypot(right, 1.0)))
    u = (azimuth % 360 / 360 * pano.shape[1]).astype(np.float32)
    v = ((90 - height) / 180 * pano.shape[0]).astype(np.float32)
    return cv2.remap(pano, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def test_measures_the_camera_from_a_sideways_drag():
    pano = panorama()
    before, after = render(pano, TRUE, 10.0), render(pano, TRUE, 42.0)

    found = measure(before, after, VIEW, default_camera(VIEW))

    assert found is not None
    camera, yaw = found
    assert camera.measured
    assert (camera.cx, camera.cy, camera.f) == pytest.approx((640, 300, 440), abs=4)
    assert yaw == pytest.approx(32, abs=0.3)


def test_cant_measure_a_featureless_view():
    grey = np.full((VIEW.height, VIEW.width), 128, np.uint8)
    assert measure(grey, grey, VIEW, default_camera(VIEW)) is None
