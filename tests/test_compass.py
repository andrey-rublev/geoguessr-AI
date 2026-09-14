import math

import cv2
import numpy as np
import pytest
from PIL import Image

from geoguessr_ai.compass import NEEDLE_TO_RADIUS, CompassReader, find_compass
from geoguessr_ai.config import Region

BACKGROUND = (86, 110, 64)
CENTER = (90, 55)


def draw_compass(heading, radius=34, size=(160, 120)):
    """Google's compass: a dark disc, a grey hub, two turn arrows, and a needle red to the north."""
    w, h = size
    img = np.full((h, w, 3), BACKGROUND, np.uint8)
    c = np.array(CENTER, float)
    cv2.circle(img, CENTER, radius, (34, 34, 34), -1, cv2.LINE_AA)
    cv2.circle(img, CENTER, round(radius * 0.45), (96, 96, 96), -1, cv2.LINE_AA)
    arc = (round(radius * 0.72), round(radius * 0.72))
    for start in (-40, 140):
        cv2.ellipse(img, CENTER, arc, 0, start, start + 80, (200, 200, 200), 2, cv2.LINE_AA)
    t = math.radians(-heading)  # north is as far anticlockwise from straight up as we face east
    tip = np.array([math.sin(t), -math.cos(t)])
    side = np.array([-tip[1], tip[0]])
    length = radius / NEEDLE_TO_RADIUS
    corners = [c + side * length * 0.27, c - side * length * 0.27]
    north = np.round([c + tip * length, *corners]).astype(np.int32)
    south = np.round([c - tip * length, *corners]).astype(np.int32)
    cv2.fillPoly(img, [north], (211, 40, 45), cv2.LINE_AA)
    cv2.fillPoly(img, [south], (222, 222, 222), cv2.LINE_AA)
    return img


@pytest.mark.parametrize("radius", [22, 34])
@pytest.mark.parametrize("heading", [0, 17, 90, 135, 200, 271, 349])
def test_reads_the_heading_off_the_needle(heading, radius):
    found = find_compass(draw_compass(heading, radius))

    assert found is not None
    x, y, r, h = found
    assert math.dist((x, y), CENTER) < 3 and abs(r - radius) < 0.25 * radius
    assert abs((h - heading + 180) % 360 - 180) < 6


def test_ignores_red_things_that_are_not_a_compass():
    img = np.full((120, 160, 3), BACKGROUND, np.uint8)
    cv2.rectangle(img, (40, 40), (70, 50), (220, 30, 40), -1)  # a red sign
    cv2.line(img, (100, 20), (130, 90), (220, 30, 40), 3)  # a red pole
    assert find_compass(img) is None


def test_reader_gives_screen_positions_and_the_clockwise_button():
    class Screen:
        def grab(self, region):
            return Image.fromarray(draw_compass(90))

    compass = CompassReader(Screen(), Region(1800, 300, 160, 120)).read()

    assert compass is not None and abs(compass.heading - 90) < 6
    assert math.dist((compass.center.x, compass.center.y), (1890, 355)) <= 2
    assert compass.clockwise.x > compass.center.x + 15 and compass.clockwise.y == compass.center.y
