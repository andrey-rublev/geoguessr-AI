"""How hot the CPU runs, on Windows, so long jobs can pause before a laptop overheats: one
went down with fatal hardware errors when training ran beside other work. Read from the
"Thermal Zone Information" performance counter, which needs no administrator rights."""

from __future__ import annotations

import contextlib
import ctypes
import sys
from collections.abc import Iterator

COUNTER = r"\Thermal Zone Information(*)\Temperature"
"""Each thermal zone's temperature, in kelvin."""
PDH_FMT_DOUBLE = 0x00000200
PDH_MORE_DATA = 0x800007D2
PROCESS_POWER_THROTTLING = 4
"""What ``SetProcessInformation`` is to set, of all it can."""
THROTTLE_EXECUTION_SPEED = 1
"""What Windows calls efficiency mode."""


class _Item(ctypes.Structure):
    """``PDH_FMT_COUNTERVALUE_ITEM_W``: a zone's name, its reading's status and the reading."""

    _fields_ = [("name", ctypes.c_wchar_p), ("status", ctypes.c_ulong), ("value", ctypes.c_double)]


_query = None
"""The open query and its counter, once made: ``False`` where there's none to read."""


def cpu_temperature() -> float | None:
    """The hottest thermal zone's temperature in °C, or ``None`` where it can't be read."""
    global _query
    if _query is None:
        _query = _open() or False
    if not _query:
        return None
    pdh, query, counter = _query
    if pdh.PdhCollectQueryData(query):
        return None
    size, count = ctypes.c_ulong(0), ctypes.c_ulong(0)
    args = (counter, PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count))
    if pdh.PdhGetFormattedCounterArrayW(*args, None) != PDH_MORE_DATA:
        return None
    buffer = (ctypes.c_byte * size.value)()
    if pdh.PdhGetFormattedCounterArrayW(*args, buffer):
        return None
    items = ctypes.cast(buffer, ctypes.POINTER(_Item))
    kelvins = [items[i].value for i in range(count.value) if items[i].status == 0]
    return max(kelvins) - 273.15 if kelvins else None


class _Throttling(ctypes.Structure):
    """``PROCESS_POWER_THROTTLING_STATE``."""

    _fields_ = [("version", ctypes.c_ulong), ("control", ctypes.c_ulong), ("state", ctypes.c_ulong)]


def _efficiency_mode(on: bool) -> None:
    if sys.platform != "win32":
        return
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessInformation.argtypes = (
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong,
    )  # fmt: skip
    asked = THROTTLE_EXECUTION_SPEED if on else 0
    state = _Throttling(1, asked, asked)
    kernel32.SetProcessInformation(
        kernel32.GetCurrentProcess(),
        PROCESS_POWER_THROTTLING,
        ctypes.byref(state),
        ctypes.sizeof(state),
    )


@contextlib.contextmanager
def running_cool() -> Iterator[None]:
    """What's inside runs in Windows' efficiency mode, which holds the CPU's boost down, where fewer
    threads only let the ones left boost harder. Getting the image model ready for the GPU took one
    laptop's CPU to 90 C within seconds on a quarter of its threads, and to 57 C at most like this.
    """
    _efficiency_mode(True)
    try:
        yield
    finally:
        _efficiency_mode(False)


def _open():
    if sys.platform != "win32":
        return None
    try:
        pdh = ctypes.WinDLL("pdh")
    except OSError:
        return None
    for name in (
        "PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData",
        "PdhGetFormattedCounterArrayW",
    ):  # fmt: skip
        getattr(pdh, name).restype = ctypes.c_ulong  # its status codes are unsigned
    query, counter = ctypes.c_void_p(), ctypes.c_void_p()
    if pdh.PdhOpenQueryW(None, None, ctypes.byref(query)):
        return None
    if pdh.PdhAddEnglishCounterW(query, COUNTER, None, ctypes.byref(counter)):
        return None
    return pdh, query, counter
