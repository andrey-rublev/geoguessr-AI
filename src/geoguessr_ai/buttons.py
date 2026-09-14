"""Is a button still showing as calibration saw it? Adverts sometimes cover OpenGuessr's Continue
button, and clicking one can open a new tab mid-session."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import Point, Region

HALF_WIDTH, HALF_HEIGHT = 80, 24
"""The patch around a button's calibrated point that is compared, in screen pixels."""
MIN_SIMILARITY = 0.5
"""How alike the patch must look to calibration's snapshot to count as the button (1 = same)."""


def button_region(point: Point) -> Region:
    return Region(point.x - HALF_WIDTH, point.y - HALF_HEIGHT, 2 * HALF_WIDTH, 2 * HALF_HEIGHT)


def button_image_path(layout_path: Path) -> Path:
    """Where calibration keeps its snapshot of the Continue button, beside the layout."""
    layout_path = Path(layout_path)
    return layout_path.with_name(f"{layout_path.stem}-continue.png")


def similarity(patch: Image.Image, snapshot: Image.Image) -> float:
    """Correlation of the two images' light and dark patterns: 1 for the same button, about 0 for
    something else. Overall brightness and contrast don't count, so a see-through button tinted
    by the map behind still matches, and a plain white advert doesn't."""
    a = np.asarray(patch.convert("L").resize(snapshot.size), dtype=np.float64).ravel()
    b = np.asarray(snapshot.convert("L"), dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    spread = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / spread) if spread > 0 else 0.0
