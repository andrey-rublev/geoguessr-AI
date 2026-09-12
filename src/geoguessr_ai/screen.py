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
        shot = self._sct.grab(
            {"left": region.left, "top": region.top, "width": region.width, "height": region.height}
        )
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def virtual_desktop(self) -> Region:
        """The bounding box of all monitors."""
        mon = self._sct.monitors[0]
        return Region(mon["left"], mon["top"], mon["width"], mon["height"])

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> Screen:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
