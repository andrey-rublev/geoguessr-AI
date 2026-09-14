"""Drop the guess pin precisely on OpenGuessr's minimap, using only pixels.

Even fully zoomed out, the expanded minimap shows only about two thirds of the world, so a
guess in Japan or California can lie beyond its edge. Placing a pin takes three steps:

1. Recognise the map from its coastlines (see :mod:`.mapcal`). If the guess is off screen,
   drag the map until it isn't.
2. Zoom in on the guess with the mouse wheel. Leaflet keeps the point under the cursor
   still, so the guess stays put while every notch doubles how precisely it can be clicked.
3. Click, check that the pin appeared under the cursor, and zoom back out for next round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from .config import Point, Region
from .mapcal import MapNotFound, MapProjection, locate_world
from .result import MIN_SCORES, find_marker, find_scale, marker_shades

MARGIN = 0.12
"""A guess closer than this to the map's edge (as a fraction of its size) is dragged inwards."""
GRIP = 0.1
"""Drags start and end at least this far inside the map, clear of its buttons."""
MAX_DRAGS = 3
MAX_ZOOM_OUTS = 6


@dataclass(frozen=True)
class Placement:
    lat: float
    lon: float
    """Where the pin went."""
    click: Point
    """The screen point that was clicked, on the zoomed-in map."""
    image: Image.Image
    """The map as recognised before zooming in."""
    projection: MapProjection
    map_xy: tuple[float, float]
    """Where the guess is in ``image``."""
    confirmed: bool | None
    """Whether the pin was seen where it was clicked (``None`` when nothing was clicked)."""


class GuessMap:
    """OpenGuessr's minimap: recognise it, then pan and zoom to place the pin precisely."""

    def __init__(
        self,
        screen,
        controls,
        region: Region,
        hover: Point,
        *,
        zoom_levels: int = 3,
        min_score: float = 0.6,
        expand_wait: float = 1.0,
        zoom_wait: float = 0.4,
        dry_run: bool = False,
    ) -> None:
        self.screen = screen
        self.controls = controls
        self.region = region
        self.hover = hover
        self.zoom_levels = zoom_levels
        self.min_score = min_score
        self.expand_wait = expand_wait
        self.zoom_wait = zoom_wait
        self.dry_run = dry_run
        self.scale: float | None = None
        """How big markers are drawn relative to 100% browser scale, once a pin has been seen."""

    def place(self, lat: float, lon: float) -> Placement:
        image, projection = self.recognise()
        image, projection = self._bring_into_view(image, projection, lat, lon)
        w, h = self.region.width, self.region.height
        x, y = projection.to_pixel(lat, lon, w)
        if not (0 <= x < w and 0 <= y < h):
            print("  guess is outside the map even after dragging; placing the pin at the edge")
        cursor = self.region.to_screen(float(np.clip(x, 3, w - 4)), float(np.clip(y, 3, h - 4)))
        local = (cursor.x - self.region.left, cursor.y - self.region.top)

        self._zoom(cursor, self.zoom_levels)
        zoomed = projection.zoomed(*local, 2**self.zoom_levels)
        zx, zy = zoomed.to_pixel(lat, lon, w)
        click = self.region.to_screen(float(np.clip(zx, 3, w - 4)), float(np.clip(zy, 3, h - 4)))
        self.controls.click(click)
        confirmed = None
        if not self.dry_run:
            confirmed = self._pin_at(click)
            if not confirmed:  # a click can be lost while the map is still settling
                self.controls.click(click)
                confirmed = self._pin_at(click)
        placed = zoomed.to_latlon(click.x - self.region.left, click.y - self.region.top)
        self._zoom(cursor, -self.zoom_levels)
        return Placement(*placed, click, image, projection, (x, y), confirmed)

    def recognise(self) -> tuple[Image.Image, MapProjection]:
        """Expand the map and recognise it, zooming out first if it was left zoomed in."""
        self.controls.move(self.hover)
        self.controls.move(self.region.center)  # stay over the expanded map so it doesn't collapse
        self.controls.sleep(self.expand_wait)
        image, projection, problem = self._locate()
        if projection is None:
            print(f"  couldn't read the map ({problem}); zooming out and retrying")
            self._zoom(self.region.center, -MAX_ZOOM_OUTS)
            image, projection, problem = self._locate()
        if projection is None:
            raise MapNotFound(f"Couldn't read the map: {problem}. Check map_region in layout.json.")
        return image, projection

    def _locate(self) -> tuple[Image.Image, MapProjection | None, str]:
        image = self.screen.grab(self.region)
        try:
            projection = locate_world(np.asarray(image.convert("RGB")))
        except MapNotFound as exc:
            return image, None, str(exc)
        if projection.score < self.min_score:
            return image, None, f"match score {projection.score:.2f} is too low"
        # The minimap repeats the world sideways, though too little of it shows to tell.
        return image, replace(projection, wraps=True), ""

    def _bring_into_view(
        self, image: Image.Image, projection: MapProjection, lat: float, lon: float
    ) -> tuple[Image.Image, MapProjection]:
        """Drag the map until the guess is comfortably inside it."""
        w, h = self.region.width, self.region.height
        for _ in range(MAX_DRAGS):
            x, y = projection.to_pixel(lat, lon, w)
            if MARGIN * w <= x <= (1 - MARGIN) * w and MARGIN * h <= y <= (1 - MARGIN) * h:
                break
            # Pull the guess towards the middle with one drag that stays inside the map.
            dx = round(float(np.clip(w / 2 - x, -(1 - 2 * GRIP) * w, (1 - 2 * GRIP) * w)))
            dy = round(float(np.clip(h / 2 - y, -(1 - 2 * GRIP) * h, (1 - 2 * GRIP) * h)))
            self.controls.drag(self.region.to_screen(w / 2 - dx / 2, h / 2 - dy / 2), dx, dy)
            self.controls.sleep(self.zoom_wait)
            image, found, _ = self._locate()
            projection = found or projection.panned(dx, dy)
        return image, projection

    def _zoom(self, at: Point, notches: int) -> None:
        """Turn the wheel a notch at a time: Leaflet ignores notches while it is still zooming."""
        for _ in range(abs(notches)):
            self.controls.scroll(at, 1 if notches > 0 else -1)
            self.controls.sleep(self.zoom_wait)

    def _pin_at(self, click: Point) -> bool:
        """Whether a pin now stands with its tip where we clicked."""
        self.controls.sleep(self.zoom_wait)
        rgb = np.asarray(self.screen.grab(self.region).convert("RGB"))
        cx, cy = click.x - self.region.left, click.y - self.region.top
        # The pin's tip is at its bottom middle; this window fits it at up to 3x browser scale.
        left, top = max(cx - 60, 0), max(cy - 120, 0)
        red, _ = marker_shades(rgb[top : cy + 20, left : cx + 60])
        if self.scale is None:
            self.scale = find_scale(red)
            if self.scale is None:
                return False
        pin = find_marker(red, "pin", self.scale)
        tip = (left + pin.x, top + pin.y)
        return pin.score >= MIN_SCORES["pin"] and math.dist(tip, (cx, cy)) <= 3 * self.scale
