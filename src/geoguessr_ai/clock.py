"""How long a multiplayer round has left, read off the timer the game shows under the map, and
whether another player has guessed, which the game says over Street View.

In a multiplayer room every round runs against a timer that everyone shares, and the game can
cut it short: in a duel, once a player guesses, the others have at most 15 seconds left. So the
bot keeps track of the time all round, and spends only what is left on looking round, reading
signs and walking on.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager

from PIL import Image

from .buttons import button_region
from .config import Point, Region

_TIMER = re.compile(r"(\d{1,2})\s*[:.;]\s*(\d{2})(?!\d)")
_GUESSED = re.compile(r"player\s*has\s*gu|guessed", re.I)
_CUT_TO = re.compile(r"max\W*(\d{1,3})\s*s", re.I)
NOTICE_WIDTH = 480
"""The middle of Street View is shrunk to this width to read what the game writes over it. The
notice that a player has guessed is big, so it still reads, and a look takes about 30 ms when
nothing is written there, on one thread."""
NOTICE_BAND = (0.2, 0.3, 0.8, 0.68)
"""Where in the view, as fractions of its width and height, the game writes its notices."""


def parse_timer(text: str) -> float | None:
    """Seconds left on a timer read as text, like ``00:42``, or None if it isn't one."""
    match = _TIMER.search(text)
    if match is None or int(match[2]) >= 60:
        return None
    return 60.0 * int(match[1]) + int(match[2])


DUEL_CUT = 15.0
"""Seconds a duel leaves the others once a player guesses, assumed when the notice is half-read."""


def parse_notice(text: str) -> float | None:
    """Seconds the round was cut to, when ``text`` says another player has guessed, as in "A
    player has guessed. Time reduced to max. 15s", or None if it isn't that notice."""
    if not _GUESSED.search(text):
        return None
    cut = _CUT_TO.search(text)
    return DUEL_CUT if cut is None else float(cut[1])


def timer_region(point: Point) -> Region:
    """The patch read around the timer's calibrated point."""
    return button_region(point)


def notice_region(view: Region) -> Region:
    """The middle of the view, where the game writes that a player has guessed."""
    left, top, right, bottom = NOTICE_BAND
    return Region(
        view.left + round(view.width * left),
        view.top + round(view.height * top),
        round(view.width * (right - left)),
        round(view.height * (bottom - top)),
    )


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

    def cut(self, deadline: float) -> None:
        """The round now ends at ``deadline`` at the latest, by ``now``."""
        self.deadline = deadline if self.deadline is None else min(self.deadline, deadline)

    def forget(self) -> None:
        """Forget the last reading, as a new round begins."""
        self.deadline = None


class GuessWatch:
    """Watches ``region``, the middle of Street View, for the game saying another player has
    guessed, which it says for only a few seconds: so in the background, a look every
    ``interval`` seconds while a round is on, reading with ``read_text`` (see
    :func:`parse_notice`). ``now`` is the time by a clock that only goes forward."""

    def __init__(
        self,
        region: Region,
        read_text: Callable[[Image.Image], str],
        now: Callable[[], float],
        interval: float = 0.4,
    ) -> None:
        self.region = region
        self.read_text = read_text
        self.now = now
        self.interval = interval
        self.guessed_at: float | None = None
        """When the notice was first seen this round, by ``now``."""
        self.cut_to = DUEL_CUT
        """Seconds the notice said the round was cut to, from then."""
        self._on = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def watching(self) -> bool:
        """Whether it is looking in the background, rather than only when asked to."""
        return self._thread is not None and self._thread.is_alive()

    def look(self, screen) -> None:
        """Look once, with ``screen``."""
        image = screen.grab(self.region)
        size = (NOTICE_WIDTH, max(1, round(image.height * NOTICE_WIDTH / image.width)))
        cut = parse_notice(self.read_text(image.resize(size, Image.BILINEAR)))
        if cut is not None and self.guessed_at is None:
            self.cut_to = cut
            self.guessed_at = self.now()

    def begin_round(self) -> None:
        """Forget the last round's notice and watch for this one's."""
        self.guessed_at, self.cut_to = None, DUEL_CUT
        self._on.set()

    def end_round(self) -> None:
        """Stop looking until the next round begins."""
        self._on.clear()

    def watch(self, open_screen: Callable[[], AbstractContextManager]) -> None:
        """Look in the background from now on, with a screen of its own from ``open_screen``,
        since screen grabbers belong to the thread that opened them."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(open_screen,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._on.set()  # wake it to stop
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self, open_screen: Callable[[], AbstractContextManager]) -> None:
        with open_screen() as screen:
            while not self._stop.is_set():
                self._on.wait()
                if self._stop.wait(self.interval) or not self._on.is_set():
                    continue
                try:
                    self.look(screen)
                except Exception as exc:  # a missed look is no reason to stop the game
                    print(f"  couldn't look for other players' guesses: {exc}")
                    self._stop.wait(5.0)
