from types import SimpleNamespace

import pytest

from geoguessr_ai.config import Region
from geoguessr_ai.screen import Screen


class X11:
    """A 1920x1080 desktop of grey that, like X11, refuses to grab anything off it."""

    monitors = [{"left": 0, "top": 0, "width": 1920, "height": 1080}] * 2

    def __init__(self):
        self.grabs = []

    def grab(self, box):
        if box["left"] < 0 or box["top"] < 0:
            raise RuntimeError("X Error of failed request: BadMatch")
        if box["left"] + box["width"] > 1920 or box["top"] + box["height"] > 1080:
            raise RuntimeError("X Error of failed request: BadMatch")
        self.grabs.append(box)
        size = (box["width"], box["height"])
        return SimpleNamespace(size=size, bgra=bytes([128, 128, 128, 255]) * (size[0] * size[1]))


@pytest.fixture
def screen():
    screen = Screen.__new__(Screen)  # no real screen to grab
    screen._sct = X11()
    return screen


def test_grabs_a_region_on_the_desktop_as_it_is(screen):
    image = screen.grab(Region(100, 200, 300, 40))

    assert image.size == (300, 40) and image.getpixel((0, 0)) == (128, 128, 128)
    assert screen._sct.grabs == [{"left": 100, "top": 200, "width": 300, "height": 40}]


def test_a_region_running_off_the_desktop_comes_out_black_there(screen):
    image = screen.grab(Region(1419, 982, 480, 120))  # round a Guess button near the bottom

    assert image.size == (480, 120)
    assert image.getpixel((10, 10)) == (128, 128, 128)  # on the desktop
    assert image.getpixel((10, 110)) == (0, 0, 0)  # below it
    assert screen._sct.grabs == [{"left": 1419, "top": 982, "width": 480, "height": 98}]


def test_a_region_wholly_off_the_desktop_is_black(screen):
    image = screen.grab(Region(-500, -100, 200, 50))

    assert image.size == (200, 50) and image.getextrema() == ((0, 0), (0, 0), (0, 0))
    assert screen._sct.grabs == []
