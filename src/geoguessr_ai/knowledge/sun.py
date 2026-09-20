"""Which hemisphere the sun says we're in.

Street View is filmed in daylight. North of the tropics the sun then stands in the southern
half of the sky, south of them in the northern half, and near the equator it can be either;
how high it climbs says more. A sun seen in a view whose heading and tilt are known gives its
compass direction and height (see :mod:`..camera`), which shift belief between latitudes.

In a level view the sun seldom shows: on 150 saved rounds it never did, since from the height
it usually stands at it is above the picture. Looking up at a clear sky usually finds it.
"""

from __future__ import annotations

import functools

import cv2
import numpy as np

FLOOR = 0.05
"""Latitudes where such a sun would be rare keep this much weight, in case it was misjudged.

A sun in the wrong half of the sky isn't unlikely there but impossible, so this only covers a
misread sun or compass. On the 16 saved rounds where the sun was found it fitted the real
latitude in 15; twice the model's guess was in the wrong hemisphere and only the sun said so."""
BIN_DEGREES = 5
AZIMUTH_BLUR, HEIGHT_BLUR = 15.0, 8.0
"""Degrees: how roughly the sun's direction and height are judged."""


def find_sun(rgb: np.ndarray, sky_share: float = 1.0) -> tuple[float, float] | None:
    """Where the sun is in a view, if it plainly shows: one round blown-out disc, whole and
    in the top ``sky_share`` of the view, glowing into the sky all round and fading outwards.

    White clouds end sharply or run into more cloud, overcast and hazy skies are white all
    over, and lamps and reflections don't glow into open sky, so none of them count.
    """
    h, w = rgb.shape[:2]
    sky = rgb[: max(1, int(h * sky_share)), :, :3]
    darkest = sky.min(axis=2).astype(np.float32)
    blown = (darkest >= 250).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(blown, connectivity=8)
    found = [
        centroids[i]
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= 0.0002 * h * w and _glows(darkest, blown, stats[i])
    ]
    if len(found) != 1:
        return None
    return float(found[0][0]), float(found[0][1])


def clear_sky(rgb: np.ndarray) -> float:
    """Share of the top third of a view that is clear blue sky, where the sun may be out."""
    top = rgb[: max(1, rgb.shape[0] // 3), :, :3].astype(np.int16)
    red, green, blue = top[..., 0], top[..., 1], top[..., 2]
    return float(((blue > 120) & (blue > red + 25) & (blue >= green)).mean())


def _glows(darkest: np.ndarray, blown: np.ndarray, stats: np.ndarray) -> bool:
    """Whether a blown-out patch looks like the sun: round, and ringed by a glow that fades
    outwards in every direction not hidden by something dark, like a tree or a roof."""
    h, w = darkest.shape
    x, y, bw, bh, area = stats
    if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
        return False  # cut off, so its middle is unknown
    if area < 0.6 * bw * bh or max(bw, bh) > 1.35 * min(bw, bh):
        return False
    cx, cy, radius = x + bw / 2, y + bh / 2, np.sqrt(area / np.pi)
    ys, xs = np.ogrid[:h, :w]
    distance = np.hypot(xs - cx, ys - cy)
    sector = ((np.arctan2(ys - cy, xs - cx) + np.pi) // (np.pi / 4)).astype(int) % 8
    near = (distance >= 1.3 * radius) & (distance <= 1.8 * radius)
    far = (distance >= 2.5 * radius) & (distance <= 3.5 * radius)
    seen, glows = 0, []
    for s in range(8):
        close, away = near & (sector == s), far & (sector == s)
        if close.sum() < 10 or away.sum() < 10:
            continue
        seen += 1
        if darkest[close].mean() >= 120:  # not hidden
            glows.append((darkest[close].mean(), darkest[away].mean(), blown[close].mean()))
    if seen < 5 or len(glows) < 4:
        return False  # too near the edge, or mostly hidden
    halo = np.array(glows)
    return bool(
        (halo[:, 2] <= 0.5).all()  # ends, rather than running into more white
        and (halo[:, 0] >= halo[:, 1] + 8).all()  # fades outwards
        and (halo[:, 0] >= 150).all()  # glows
        and np.median(halo[:, 0]) < 245  # in a sky that isn't white all over
    )


@functools.lru_cache(maxsize=1)
def _sky_table() -> tuple[np.ndarray, np.ndarray]:
    """``(latitudes, table)``: how often, at each latitude, the daytime sun stands in each
    direction (bins of :data:`BIN_DEGREES` round the compass) at each height above the horizon."""
    from scipy.ndimage import gaussian_filter1d

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
    bins = [np.arange(0, top + BIN_DEGREES, BIN_DEGREES) for top in (360, 90)]
    up = height > 2
    table = np.stack(
        [np.histogram2d(a[v], e[v], bins)[0] for a, e, v in zip(azimuth, height, up, strict=True)]
    )
    table = gaussian_filter1d(table, AZIMUTH_BLUR / BIN_DEGREES, axis=1, mode="wrap")
    table = gaussian_filter1d(table, HEIGHT_BLUR / BIN_DEGREES, axis=2, mode="nearest")
    return lats, table / table.sum(axis=(1, 2), keepdims=True)


def latitude_likelihood(azimuth: float, height: float):
    """A function giving how well each latitude fits a sun seen towards ``azimuth``, ``height``
    degrees above the horizon.

    Latitudes where the sun stands there at least as often as on average get 1; rarer ones get
    less, down to :data:`FLOOR`.
    """
    lats, table = _sky_table()
    column = min(int(np.clip(height, 0, 89.9) // BIN_DEGREES), table.shape[2] - 1)
    fit = table[:, int(azimuth % 360 // BIN_DEGREES), column]
    weights = np.clip(fit / fit.mean(), FLOOR, 1.0)
    return lambda lat: np.interp(lat, lats, weights)
