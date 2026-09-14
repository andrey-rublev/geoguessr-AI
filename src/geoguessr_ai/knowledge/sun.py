"""Which hemisphere the sun says we're in.

Street View is filmed in daylight. North of the tropics the sun then stands in the southern
half of the sky, south of them in the northern half, and near the equator it can be either.
With views that face north, east, south and west (see :mod:`..compass`), a sun low enough to
show in one gives its compass direction, which shifts belief between latitudes.
"""

from __future__ import annotations

import functools

import cv2
import numpy as np

VIEW_FOV = 105.0
"""Degrees across the calibrated view (Street View's whole picture spans about 130)."""
FLOOR = 0.25
"""The least likely latitude keeps this share of the likeliest one's weight, in case the sun
was misjudged."""
BIN_DEGREES = 5


def find_sun(rgb: np.ndarray) -> tuple[float, float] | None:
    """Where the sun is in a view, if it plainly shows: one small blown-out disc with a glow.

    Big blown-out patches (overcast skies, white walls) or several bright spots mean no sun.
    """
    h, w = rgb.shape[:2]
    sky = rgb[: int(h * 0.6), :, :3].astype(np.int16)
    blown = (sky.min(axis=2) >= 245).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(blown)
    least, most = 0.0002 * h * w, 0.02 * h * w
    patches = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= least]
    if len(patches) != 1:
        return None
    i = patches[0]
    x, y, bw, bh, area = stats[i]
    if area > most or area < 0.45 * bw * bh or max(bw, bh) > 2.5 * min(bw, bh):
        return None
    cx, cy = centroids[i]
    radius = np.sqrt(area / np.pi)
    brightness = sky.mean(axis=2)
    ys, xs = np.mgrid[0 : sky.shape[0], 0:w]
    distance = np.hypot(xs - cx, ys - cy)
    near = brightness[(distance >= 1.2 * radius) & (distance <= 2 * radius)]
    far = brightness[(distance >= 2.5 * radius) & (distance <= 4 * radius)]
    if near.size == 0 or far.size == 0:
        return None
    # The sun lights up the sky around it; a white sign or wall doesn't.
    if near.mean() < np.median(brightness) + 30 or near.mean() < far.mean() + 10:
        return None
    return float(cx), float(cy)


def sun_azimuth(x: float, width: int, heading: float) -> float:
    """Compass direction of a point ``x`` pixels across a view facing ``heading``."""
    return (heading + (x / width - 0.5) * VIEW_FOV) % 360.0


@functools.lru_cache(maxsize=1)
def _azimuth_table() -> tuple[np.ndarray, np.ndarray]:
    """How often, at each latitude, a sun low enough to see stands in each direction."""
    lats = np.arange(-60.0, 76.0)
    days = np.arange(0, 365, 3)
    hours = np.arange(7.0, 18.01, 0.25)
    phi = np.radians(lats)[:, None, None]
    tilt = np.radians(23.44) * np.sin(2 * np.pi * (284 + days) / 365)[None, :, None]
    hour = np.radians(15.0 * (hours - 12.0))[None, None, :]
    height = np.degrees(
        np.arcsin(np.sin(phi) * np.sin(tilt) + np.cos(phi) * np.cos(tilt) * np.cos(hour))
    )
    azimuth = np.degrees(
        np.arctan2(
            -np.sin(hour) * np.cos(tilt),
            np.sin(tilt) * np.cos(phi) - np.cos(tilt) * np.sin(phi) * np.cos(hour),
        )
    )
    azimuth = np.broadcast_to(azimuth % 360.0, height.shape)
    bins = np.arange(0, 360 + BIN_DEGREES, BIN_DEGREES)
    visible = (height > 2) & (height < 40)  # in a level view, not overhead
    table = np.stack([np.histogram(a[v], bins)[0] for a, v in zip(azimuth, visible, strict=True)])
    # Blur around the compass, since the sun's direction is only judged roughly.
    offsets = np.arange(-6, 7)
    kernel = np.exp(-0.5 * (offsets * BIN_DEGREES / 20.0) ** 2)
    table = sum(k * np.roll(table, o, axis=1) for o, k in zip(offsets, kernel, strict=True))
    return lats, table / table.sum(axis=1, keepdims=True)


def latitude_likelihood(azimuth: float):
    """A function giving how well each latitude fits a sun seen towards ``azimuth``."""
    lats, table = _azimuth_table()
    fit = table[:, int(azimuth % 360 // BIN_DEGREES)]
    weights = FLOOR + (1 - FLOOR) * fit / fit.max()
    return lambda lat: np.interp(lat, lats, weights)
