"""Screen capture with mss, plus Windows DPI awareness so pixels match mouse coordinates."""

from __future__ import annotations

import ctypes
import sys

from PIL import Image

from .config import Region

_dpi_aware = False


def enable_dpi_awareness() -> None:
    """Make Windows report physical pixels to this process.

    Without this, a 150% display scale makes screenshots and mouse coordinates
    disagree and every click lands in the wrong place.
    """
    global _dpi_aware
    if _dpi_aware or sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()
    _dpi_aware = True


class Screen:
    def __init__(self) -> None:
        enable_dpi_awareness()
        import mss

        self._sct = mss.MSS()

    def grab(self, region: Region) -> Image.Image:
        """What ``region`` of the screen shows, black where it runs off the desktop: Windows
        gives black there, but X11 (as on WSL's screen of the bot's own) refuses the grab."""
        desk = self.virtual_desktop()
        left, top = max(region.left, desk.left), max(region.top, desk.top)
        right, bottom = min(region.right, desk.right), min(region.bottom, desk.bottom)
        if (left, top, right, bottom) == (region.left, region.top, region.right, region.bottom):
            return self._grab(region)
        canvas = Image.new("RGB", (region.width, region.height))
        if left < right and top < bottom:
            shown = self._grab(Region(left, top, right - left, bottom - top))
            canvas.paste(shown, (left - region.left, top - region.top))
        return canvas

    def _grab(self, region: Region) -> Image.Image:
        shot = self._sct.grab(
            {"left": region.left, "top": region.top, "width": region.width, "height": region.height}
        )
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def open_another(self) -> Screen:
        """Another screen grabber, for another thread: each belongs to the one that opened it."""
        return Screen()

    def virtual_desktop(self) -> Region:
        """The bounding box of all monitors."""
        mon = self._sct.monitors[0]
        return Region(mon["left"], mon["top"], mon["width"], mon["height"])

    def monitors(self) -> list[Region]:
        """Each monitor's part of the desktop."""
        return [
            Region(mon["left"], mon["top"], mon["width"], mon["height"])
            for mon in self._sct.monitors[1:]
        ]

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> Screen:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
