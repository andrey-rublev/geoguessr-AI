"""How hot the CPU runs, on Windows, so long jobs can pause before a laptop overheats: one
went down with fatal hardware errors when training ran beside other work. Read from the
"Thermal Zone Information" performance counter, which needs no administrator rights."""

from __future__ import annotations

import ctypes
import sys

COUNTER = r"\Thermal Zone Information(*)\Temperature"
"""Each thermal zone's temperature, in kelvin."""
PDH_FMT_DOUBLE = 0x00000200
PDH_MORE_DATA = 0x800007D2


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
