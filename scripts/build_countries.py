"""Build the country lookup grid that ships in ``geoguessr_ai/knowledge``.

Rasterises Natural Earth's 1:10m admin-0 countries (public domain) onto a 0.05 degree
latitude/longitude grid. Sea pixels take the nearest country, so coastal Street View spots
and pins dropped just offshore still belong to a country. Only rerun it to update borders:

    python scripts/build_countries.py [ne_10m_admin_0_countries.geojson]
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

SOURCE = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_admin_0_countries.geojson"
)
DEGREES_PER_PIXEL = 0.05
OUT_DIR = Path(__file__).resolve().parents[1] / "src" / "geoguessr_ai" / "knowledge"
# Natural Earth gives these areas no ISO code; they join the country that surrounds them.
UNCODED = {
    "Akrotiri": "CY",
    "Bir Tawil": "EG",
    "Cyprus U.N. Buffer Zone": "CY",
    "Dhekelia": "CY",
    "N. Cyprus": "CY",
    "Siachen Glacier": "IN",
    "Somaliland": "SO",
    "Southern Patagonian Ice Field": "AR",
    "USNB Guantanamo Bay": "CU",
}


def _rings(geometry: dict) -> list[list[list[float]]]:
    """Outer rings of a (Multi)Polygon; holes are filled by whatever is drawn over them later."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    return [polygon[0] for polygon in geometry["coordinates"]]


def _area(ring: list[list[float]]) -> float:
    x, y = np.asarray(ring, dtype=np.float64).T
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def build(geojson: dict) -> tuple[np.ndarray, list[str]]:
    width, height = round(360 / DEGREES_PER_PIXEL), round(180 / DEGREES_PER_PIXEL)
    shapes = []
    for feature in geojson["features"]:
        props = feature["properties"]
        code = props["ISO_A2_EH"] if props["ISO_A2_EH"] != "-99" else UNCODED.get(props["NAME"])
        if code:
            shapes += [(code, ring) for ring in _rings(feature["geometry"])]
    codes = sorted({code for code, _ in shapes})
    if len(codes) > 255:
        raise ValueError(f"{len(codes)} countries don't fit in one byte")
    index = {code: i + 1 for i, code in enumerate(codes)}

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    # Largest first, so enclaves (Lesotho, San Marino) and islands are drawn on top.
    for code, ring in sorted(shapes, key=lambda shape: -_area(shape[1])):
        points = [
            ((lon + 180) / DEGREES_PER_PIXEL, (90 - lat) / DEGREES_PER_PIXEL) for lon, lat in ring
        ]
        draw.polygon(points, fill=index[code], outline=index[code])

    grid = np.asarray(canvas)
    nearest = ndimage.distance_transform_edt(grid == 0, return_distances=False, return_indices=True)
    return grid[nearest[0], nearest[1]], codes


def main() -> None:
    if len(sys.argv) > 1:
        geojson = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(SOURCE) as response:
            geojson = json.load(response)
    grid, codes = build(geojson)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(OUT_DIR / "countries.png", optimize=True)
    (OUT_DIR / "countries.txt").write_text("\n".join(codes) + "\n", encoding="utf-8")
    print(f"{len(codes)} countries, {grid.shape[1]}x{grid.shape[0]} grid saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
