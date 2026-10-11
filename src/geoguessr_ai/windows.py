"""Bring OpenGuessr's browser window to the front, on Windows, for when none of it shows on
screen to be found: hidden behind other windows, or minimised. Only the window's title is
looked at, as the taskbar shows it; the game itself is still only ever seen in pixels."""

from __future__ import annotations

import sys

TITLE = "OpenGuessr - Free"
"""The start of the game's tab title, which a browser window shows while that tab is open."""
SW_RESTORE = 9


def bring_forward(title: str = TITLE) -> bool:
    """Bring forward the first window whose title contains ``title``. Whether there was one."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        length = user32.GetWindowTextLengthW(hwnd)
        if length and user32.IsWindowVisible(hwnd):
            text = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, text, length + 1)
            if title.lower() in text.value.lower():
                found.append(hwnd)
                return False
        return True

    user32.EnumWindows(visit, 0)
    if not found:
        return False
    hwnd = found[0]
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    # Windows lets only the window in front hand the front on: borrow its thread's say-so.
    ours = kernel32.GetCurrentThreadId()
    theirs = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    user32.AttachThreadInput(ours, theirs, True)
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        user32.AttachThreadInput(ours, theirs, False)
    return True
