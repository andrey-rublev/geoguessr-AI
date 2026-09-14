"""Read Street View's compass from pixels, and find its buttons.

OpenGuessr shows Google's compass at the right edge of the panorama: a dark disc with a needle
whose red half points north. Pressing the disc turns the view to face north, and the arrows
on either side turn it by exactly 90 degrees, so the bot can look north, east, south and west.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Layout, Point, Region

NEEDLE_TO_RADIUS = 1.75
"""The disc's radius, measured in lengths of the needle's red half."""
CLOCKWISE_OFFSET = 0.74
"""How far right of the centre the clockwise arrow sits, as a fraction of the radius."""


@dataclass(frozen=True)
class Compass:
    center: Point
    radius: float
    heading: float
    """Where the view faces, in degrees clockwise from north."""

    @property
    def clockwise(self) -> Point:
        """The arrow that turns the view 90 degrees clockwise."""
        return self.center.offset(CLOCKWISE_OFFSET * self.radius, 0)


def compass_region(layout: Layout, desktop: Region) -> Region:
    """Where to look for the compass: right of the expanded minimap, down to the Guess button."""
    left = layout.map_region.left + layout.map_region.width - 20
    right = desktop.left + desktop.width
    bottom = max(layout.guess_button.y, layout.map_region.top + layout.map_region.height)
    return Region(left, layout.view.top, max(right - left, 1), bottom - layout.view.top)


class CompassReader:
    def __init__(self, screen, region: Region) -> None:
        self.screen = screen
        self.region = region

    def read(self) -> Compass | None:
        rgb = np.asarray(self.screen.grab(self.region).convert("RGB"))
        found = find_compass(rgb)
        if found is None:
            return None
        x, y, radius, heading = found
        return Compass(self.region.to_screen(x, y), radius, heading)


def find_compass(rgb: np.ndarray) -> tuple[float, float, float, float] | None:
    """The compass in a screenshot as ``(x, y, radius, heading)``, or ``None`` if none shows."""
    px = rgb[..., :3].astype(np.int16)
    red = (px[..., 0] > 150) & (px[..., 0] - np.maximum(px[..., 1], px[..., 2]) > 90)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8))
    best, best_area = None, 0
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if not 12 <= area <= 4000 or area <= best_area:
            continue
        ys, xs = np.nonzero(labels == i)
        found = _read_needle(np.stack([xs, ys], axis=1).astype(np.float64), px)
        if found is not None:
            best, best_area = found, area
    return best


def _read_needle(points: np.ndarray, px: np.ndarray) -> tuple[float, float, float, float] | None:
    """Treat a red blob as the needle's north half: a thin triangle from the centre to the tip."""
    mean = points.mean(axis=0)
    spread, axes = np.linalg.eigh(np.cov((points - mean).T))
    if spread[1] < 2.5 * max(spread[0], 1e-9):
        return None  # not long and thin
    axis = axes[:, 1]
    along = (points - mean) @ axis
    if along.max() < -along.min():  # point the axis at the tip, the end farther from the centroid
        axis, along = -axis, -along
    length = 1.5 * along.max()  # a triangle's centroid is a third of the way from its base
    center = mean - axis * length / 3
    radius = NEEDLE_TO_RADIUS * length
    south_half = _share(px, center - axis * length / 2, length / 4, _is_light)
    if south_half < 0.3 or not _disc_is_dark(px, center, radius):
        return None
    north = math.degrees(math.atan2(axis[0], -axis[1]))  # clockwise from straight up
    return float(center[0]), float(center[1]), float(radius), (-north) % 360.0


def _is_light(px: np.ndarray) -> np.ndarray:
    return (px.min(axis=-1) >= 170) & (px.max(axis=-1) - px.min(axis=-1) <= 40)


def _share(px: np.ndarray, point: np.ndarray, radius: float, test) -> float:
    """Fraction of the pixels within ``radius`` of ``point`` that pass ``test``."""
    h, w = px.shape[:2]
    x0, x1 = max(int(point[0] - radius), 0), min(int(point[0] + radius) + 1, w)
    y0, y1 = max(int(point[1] - radius), 0), min(int(point[1] + radius) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = (xs - point[0]) ** 2 + (ys - point[1]) ** 2 <= radius**2
    return float(test(px[y0:y1, x0:x1])[inside].mean()) if inside.any() else 0.0


def _disc_is_dark(px: np.ndarray, center: np.ndarray, radius: float) -> bool:
    """Whether the top and bottom of the disc's rim are dark (the arrows sit left and right)."""
    h, w = px.shape[:2]
    angles = np.radians(np.concatenate([np.linspace(-40, 40, 9), np.linspace(140, 220, 9)]))
    xs = np.round(center[0] + 0.85 * radius * np.sin(angles)).astype(int)
    ys = np.round(center[1] - 0.85 * radius * np.cos(angles)).astype(int)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if inside.mean() < 0.8:
        return False
    return float((px[ys[inside], xs[inside]].max(axis=-1) < 80).mean()) >= 0.7
