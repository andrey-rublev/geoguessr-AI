"""Mouse control with two independent emergency stops.

* Press the stop key (F8 by default) at any time: the bot finishes its current
  micro-action and exits.
* Slam the mouse into any screen corner: pyautogui's fail-safe aborts immediately.
"""

from __future__ import annotations

import sys
import threading
import time

from .config import Point
from .screen import enable_dpi_awareness

# pyautogui passes the scroll amount straight to the OS: raw wheel units on Windows
# (120 per notch), whole notches on macOS and Linux.
WHEEL_NOTCH = 120 if sys.platform == "win32" else 1


class StopRequested(Exception):
    """Raised when the user asks the bot to stop."""


class Controls:
    def __init__(self, *, dry_run: bool = False, stop_key: str = "f8") -> None:
        enable_dpi_awareness()
        import pyautogui  # imported lazily: it touches the display on import
        from pynput import keyboard

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
        self._gui = pyautogui
        self.dry_run = dry_run
        self._stop = threading.Event()

        target = getattr(keyboard.Key, stop_key.lower(), None) or keyboard.KeyCode.from_char(
            stop_key
        )

        def on_press(key) -> None:
            if key == target:
                self._stop.set()

        self._listener = keyboard.Listener(on_press=on_press, daemon=True)
        self._listener.start()

    def check(self) -> None:
        if self._stop.is_set():
            raise StopRequested("Stop key pressed")

    def sleep(self, seconds: float) -> None:
        """Sleep in small slices so the stop key stays responsive."""
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            self.check()
            time.sleep(min(remaining, 0.05))

    def position(self) -> Point:
        x, y = self._gui.position()
        return Point(x, y)

    def move(self, p: Point, duration: float = 0.15) -> None:
        self.check()
        self._gui.moveTo(p.x, p.y, duration=duration)

    def click(self, p: Point) -> None:
        """Click at ``p``. In dry-run mode the mouse only moves there."""
        self.move(p)
        if self.dry_run:
            print(f"    [dry-run] would click at ({p.x}, {p.y})")
            return
        self._gui.click()

    def press(self, key: str) -> None:
        """Press a key, like ``"up"``. In dry-run mode nothing is pressed."""
        self.check()
        if self.dry_run:
            print(f"    [dry-run] would press {key}")
            return
        self._gui.press(key)

    def drag(self, start: Point, dx: int, dy: int = 0, duration: float = 0.5) -> None:
        """Drag by (``dx``, ``dy``) from ``start``, pausing before release so maps don't glide."""
        self.move(start)
        self._gui.mouseDown()
        try:
            self._gui.moveTo(start.x + dx, start.y + dy, duration=duration)
            self.sleep(0.25)
        finally:
            self._gui.mouseUp()

    def scroll(self, p: Point, clicks: int) -> None:
        """Scroll the wheel at ``p``; positive zooms a map in, negative zooms out."""
        self.move(p)
        for _ in range(abs(clicks)):
            self.check()
            self._gui.scroll(WHEEL_NOTCH if clicks > 0 else -WHEEL_NOTCH)
            time.sleep(0.08)

    def close(self) -> None:
        self._listener.stop()

    def __enter__(self) -> Controls:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
