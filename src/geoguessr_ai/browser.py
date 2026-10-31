"""Play in a browser of the bot's own: a Chrome window it drives itself, so your own mouse,
keyboard and screen stay yours while it plays.

The bot sees and acts exactly as on your screen: it takes screenshots of the window and moves a
mouse and presses keys in it. It never reads the page's code. Playwright only carries the
screenshots and the input, into a window that needn't be in front, or even on screen.

Playwright's objects belong to the thread that made them, but the bot grabs the screen from more
than one (the watch for other players' guesses runs in its own), so one thread of its own runs
the browser, and every call is handed to it.
"""

from __future__ import annotations

import contextlib
import io
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from PIL import Image

from .config import Point, Region
from .controls import Controls, StopRequested

GAME_URL = "https://openguessr.com/"
PROFILE_DIR = Path(".browser-profile")
"""Where the bot's browser keeps its cookies and settings, so choices made in it (like a
consent banner's) and a sign-in, if you sign in there, last from one run to the next."""
SHOT_TIMEOUT_MS = 10_000
"""How long a screenshot may take before the frame counts as blank: in a live multiplayer
room, one took over 30 s as a round began, and the bot stopped on it."""
WHEEL_DELTA = 100
"""Pixels a wheel notch scrolls in Chrome, which Leaflet's maps take for one zoom level."""
KEYS = {
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "esc": "Escape",
    "enter": "Enter",
    "space": " ",
}
"""pyautogui's key names, as Playwright knows them."""


class BrowserSession:
    """A Chrome window of the bot's own, run by a thread of its own: :meth:`call` hands it work
    from any thread and waits for the result."""

    def __init__(
        self,
        url: str = GAME_URL,
        size: tuple[int, int] = (1600, 900),
        profile: Path = PROFILE_DIR,
        channel: str | None = "chrome",
        headless: bool = False,
    ) -> None:
        self.size = size
        self._work: queue.Queue[tuple[Callable[[Any], Any], Future] | None] = queue.Queue()
        started: Future = Future()
        args = (url, size, profile, channel, headless, started)
        self._thread = threading.Thread(target=self._run, args=args, daemon=True)
        self._thread.start()
        started.result()  # raises here if the browser couldn't start

    def _run(self, url, size, profile, channel, headless, started: Future) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            started.set_exception(SystemExit("--browser needs Playwright: pip install playwright"))
            return
        try:
            playwright = sync_playwright().start()
            context = playwright.chromium.launch_persistent_context(
                str(profile),
                channel=channel,
                headless=headless,
                viewport={"width": size[0], "height": size[1]},
                device_scale_factor=1,
                # Street View draws with the GPU, and stops drawing in a window it thinks hidden.
                args=[
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ],
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(url)
        except Exception as error:  # noqa: BLE001 - handed to whoever started the session
            started.set_exception(error)
            return
        self.page = page
        started.set_result(None)
        while (item := self._work.get()) is not None:
            work, done = item
            try:
                done.set_result(work(page))
            except Exception as error:  # noqa: BLE001 - raised again in the calling thread
                done.set_exception(error)
        with contextlib.suppress(Exception):  # closed already, if its window was
            context.close()
        playwright.stop()

    def call(self, work: Callable[[Any], Any]) -> Any:
        """Run ``work(page)`` on the browser's thread and return what it returns. Once its
        window has been closed, raises :class:`.controls.StopRequested`: closing it stops the
        bot, as the stop key does."""
        done: Future = Future()
        self._work.put((work, done))
        try:
            return done.result()
        except Exception as error:
            if type(error).__name__ == "TargetClosedError":  # Playwright's, raised on its thread
                raise StopRequested("the bot's browser was closed") from None
            raise

    def close(self) -> None:
        self._work.put(None)
        self._thread.join(timeout=10)


class BrowserScreen:
    """Screenshots of the bot's browser window, like :class:`.screen.Screen` takes of yours."""

    def __init__(self, session: BrowserSession) -> None:
        self.session = session

    def grab(self, region: Region) -> Image.Image:
        width, height = self.session.size
        left, top = max(region.left, 0), max(region.top, 0)
        right = min(region.left + region.width, width)
        bottom = min(region.top + region.height, height)
        shot = Image.new("RGB", (region.width, region.height))
        if right > left and bottom > top:
            clip = {"x": left, "y": top, "width": right - left, "height": bottom - top}
            try:
                png = self.session.call(
                    lambda page: page.screenshot(
                        clip=clip, animations="allow", timeout=SHOT_TIMEOUT_MS
                    )
                )
            except Exception as error:
                if type(error).__name__ != "TimeoutError":  # Playwright's, raised on its thread
                    raise
                print("  the bot's browser was too busy to take a screenshot; trying again")
                return shot  # blank: whatever waits for the game looks again
            shot.paste(
                Image.open(io.BytesIO(png)).convert("RGB"), (left - region.left, top - region.top)
            )
        return shot

    def open_another(self) -> BrowserScreen:
        return self  # any thread may grab: the session hands each grab to its own thread

    def virtual_desktop(self) -> Region:
        return Region(0, 0, *self.session.size)

    def monitors(self) -> list[Region]:
        return [self.virtual_desktop()]

    def close(self) -> None:
        pass

    def __enter__(self) -> BrowserScreen:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _NeverRaised(Exception):
    """In the bot's browser there are no screen corners to slam the mouse into."""


class BrowserControls(Controls):
    """A mouse and keyboard in the bot's browser window, like :class:`.controls.Controls` on
    your screen. The stop key still works from anywhere."""

    stop_hint = "F8"

    def __init__(self, session: BrowserSession, *, dry_run: bool = False, stop_key: str = "f8"):
        self._listen(dry_run, stop_key)  # no screen corners, and no mouse of yours to move
        self.session = session
        self._at = Point(session.size[0] // 2, session.size[1] // 2)

    @property
    def failsafe(self) -> type[BaseException]:
        return _NeverRaised

    def position(self) -> Point:
        return self._at

    def move(self, p: Point, duration: float = 0.15) -> None:
        """Straight to ``p``, not gliding there: on the way, the mouse would open the map, which
        here only closes again once the mouse leaves the window (see :meth:`leave`)."""
        self.check()
        self.session.call(lambda page: page.mouse.move(p.x, p.y))
        self._at = p

    def leave(self, p: Point) -> None:
        """Out of the window first: moved over Street View, Playwright's mouse never leaves the
        map as far as the game can tell, so it stays open over the view; out of the window, it
        closes, as when your own mouse leaves it."""
        self.check()
        self.session.call(lambda page: page.mouse.move(-10, -10))
        self.move(p)

    def click(self, p: Point) -> None:
        self.move(p)
        if self.dry_run:
            print(f"    [dry-run] would click at ({p.x}, {p.y})")
            return
        self.session.call(lambda page: page.mouse.click(p.x, p.y))

    def press(self, key: str) -> None:
        self.check()
        if self.dry_run:
            print(f"    [dry-run] would press {key}")
            return
        self.session.call(lambda page: page.keyboard.press(KEYS.get(key, key)))

    def drag(self, start: Point, dx: int, dy: int = 0, duration: float = 0.5) -> None:
        self.move(start)
        end = Point(start.x + dx, start.y + dy)
        steps = max(2, round(duration / 0.02))

        def drag(page) -> None:
            page.mouse.down()
            page.mouse.move(end.x, end.y, steps=steps)

        self.session.call(drag)
        try:
            self.sleep(0.25)
        finally:
            self.session.call(lambda page: page.mouse.up())
        self._at = end

    def scroll(self, p: Point, clicks: int) -> None:
        self.move(p)
        for _ in range(abs(clicks)):
            self.check()
            delta = -WHEEL_DELTA if clicks > 0 else WHEEL_DELTA
            self.session.call(lambda page, delta=delta: page.mouse.wheel(0, delta))
            time.sleep(0.08)
