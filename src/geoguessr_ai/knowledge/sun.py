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
FLOOR = 0.2
"""Latitudes where such a sun would be rare keep this much weight, in case it was misjudged."""
BIN_DEGREES = 5


def find_sun(rgb: np.ndarray) -> tuple[float, float] | None:
    """Where the sun is in a view, if it plainly shows: one small blown-out disc glowing into
    open sky all round.

    Blown-out patches that are big (overcast skies, white walls), several, or not ringed by
    glowing sky (lamps, reflections, the edges of clouds and roofs) don't count.
    """
    h, w = rgb.shape[:2]
    sky = rgb[: int(h * 0.6), :, :3].astype(np.int16)
    blown = (sky.min(axis=2) >= 245).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(blown)
    least, most = 0.0002 * h * w, 0.02 * h * w
    patches = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= least]
    if len(patches) != 1:
        return None
    i = patches[0]
    _, _, bw, bh, area = stats[i]
    if area > most or area < 0.45 * bw * bh or max(bw, bh) > 2.5 * min(bw, bh):
        return None
    cx, cy = centroids[i]
    radius = np.sqrt(area / np.pi)
    brightness = sky.mean(axis=2)
    ys, xs = np.mgrid[0 : sky.shape[0], 0:w]
    distance = np.hypot(xs - cx, ys - cy)
    near = (distance >= 1.2 * radius) & (distance <= 2 * radius)
    far = (distance >= 3 * radius) & (distance <= 5 * radius)
    sector = ((np.arctan2(ys - cy, xs - cx) + np.pi) // (np.pi / 4)).astype(int) % 8
    glow = [brightness[near & (sector == s)] for s in range(8)]
    if any(g.size < 5 for g in glow) or far.sum() < 100:
        return None  # too near the edge of the view to judge
    around = brightness[far]
    # The sun lights the sky evenly all round, brightest close in...
    if min(g.mean() for g in glow) < np.median(brightness) + 30:
        return None
    if around.mean() > 235 or brightness[near].mean() < around.mean() + 25:
        return None
    # ...and hangs in open sky, not against a wall or roof.
    if (around < 120).mean() > 0.2:
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
    """A function giving how well each latitude fits a sun seen towards ``azimuth``.

    Latitudes where a low sun stands that way at least as often as on average get 1; rarer
    ones get less, down to :data:`FLOOR`.
    """
    lats, table = _azimuth_table()
    fit = table[:, int(azimuth % 360 // BIN_DEGREES)]
    weights = np.clip(fit / fit.mean(), FLOOR, 1.0)
    return lambda lat: np.interp(lat, lats, weights)
