"""How long a multiplayer round has left, read off the timer the game shows under the map.

In a multiplayer room every round runs against a timer that everyone shares, and the game can
cut it short, as when another player guesses in a duel. So the bot reads it as it goes rather
than once, and spends only the time left on looking round, reading signs and walking on.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

from PIL import Image

from .buttons import button_region
from .config import Point, Region

_TIMER = re.compile(r"(\d{1,2})\s*[:.;]\s*(\d{2})(?!\d)")


def parse_timer(text: str) -> float | None:
    """Seconds left on a timer read as text, like ``00:42``, or None if it isn't one."""
    match = _TIMER.search(text)
    if match is None or int(match[2]) >= 60:
        return None
    return 60.0 * int(match[1]) + int(match[2])


def timer_region(point: Point) -> Region:
    """The patch read around the timer's calibrated point."""
    return button_region(point)


class RoundClock:
    """The round's timer, read with ``read_text`` from ``region`` on screen, if both are known.
    ``show`` and ``hide`` open and close whatever has to be open for it to show, like the map;
    ``now`` is the time in seconds by a clock that only goes forward."""

    def __init__(
        self,
        screen,
        region: Region | None,
        read_text: Callable[[Image.Image], str] | None,
        now: Callable[[], float],
        *,
        show: Callable[[], None] | None = None,
        hide: Callable[[], None] | None = None,
    ) -> None:
        self.screen = screen
        self.region = region
        self.read_text = read_text
        self.now = now
        self.show = show
        self.hide = hide
        self.deadline: float | None = None
        """When the round ends, by ``now``, as of the last reading."""

    @property
    def readable(self) -> bool:
        """Whether there is a timer to read, and a way to read it."""
        return self.region is not None and self.read_text is not None

    def read(self) -> float | None:
        """Seconds left, read as the timer shows or, failing that, with it opened to show, and
        remembered as when the round ends. None if it can't be read, which forgets nothing."""
        if self.region is None or self.read_text is None:
            return None
        left = parse_timer(self.read_text(self.screen.grab(self.region)))
        if left is None and self.show is not None:
            self.show()
            try:
                left = parse_timer(self.read_text(self.screen.grab(self.region)))
            finally:
                if self.hide is not None:
                    self.hide()
        if left is not None:
            self.deadline = self.now() + left
        return left

    def left(self) -> float:
        """Seconds left by the last reading, or infinitely many if it has never been read."""
        return math.inf if self.deadline is None else self.deadline - self.now()

    def forget(self) -> None:
        """Forget the last reading, as a new round begins."""
        self.deadline = None
