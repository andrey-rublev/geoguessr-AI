import math

from PIL import Image

from geoguessr_ai.clock import RoundClock, parse_timer, timer_region
from geoguessr_ai.config import Point


def test_reads_timers_as_the_recogniser_gives_them():
    assert parse_timer("00:42") == 42 and parse_timer("o 01:05") == 65
    assert parse_timer("00:07 1/3") == 7  # the round's number beside it
    assert parse_timer("1/3") is None and parse_timer("") is None and parse_timer("00:75") is None


class Timer:
    """The timer under a map that has to be open for it to show, ``left`` seconds from 0."""

    def __init__(self, left):
        self.t, self.left, self.open, self.opened = 0.0, left, False, 0

    def grab(self, region):
        return Image.new("RGB", (region.width, region.height))

    def read_text(self, image):
        return f"00:{max(self.left - int(self.t), 0):02d}" if self.open else "Multiplayer"

    def show(self):
        self.open, self.opened = True, self.opened + 1

    def hide(self):
        self.open = False


def clock_on(timer):
    region = timer_region(Point(500, 400))
    return RoundClock(
        timer, region, timer.read_text, lambda: timer.t, show=timer.show, hide=timer.hide
    )


def test_opens_the_map_to_read_the_timer_and_remembers_when_the_round_ends():
    timer = Timer(left=42)
    clock = clock_on(timer)
    assert clock.left() == math.inf  # never read

    assert clock.read() == 42 and timer.opened == 1 and not timer.open
    timer.t = 10.0
    assert clock.left() == 32.0


def test_a_timer_it_cant_read_forgets_nothing():
    timer = Timer(left=42)
    clock = clock_on(timer)
    clock.read()
    clock.read_text = lambda image: "Multipl"  # covered, say
    timer.t = 5.0
    assert clock.read() is None and clock.left() == 37.0
    assert not RoundClock(timer, None, None, lambda: 0.0).readable
