"""One-time, interactive recording of where OpenGuessr's UI sits on your screen."""

from __future__ import annotations

import time
from pathlib import Path

from PIL import ImageDraw

from .config import DEFAULT_LAYOUT_PATH, Layout, Point, Region
from .controls import Controls
from .screen import Screen

COUNTDOWN_S = 3


def _record(controls: Controls, step: str, prompt: str) -> Point:
    print(f"\n[{step}] {prompt}")
    input(f"  Press Enter, then hover there within {COUNTDOWN_S}s (don't click)...")
    for remaining in range(COUNTDOWN_S, 0, -1):
        print(f"  {remaining}...", end="", flush=True)
        time.sleep(1)
    point = controls.position()
    print(f"  recorded ({point.x}, {point.y})")
    return point


def _save_preview(screen: Screen, layout: Layout, path: Path) -> None:
    desktop = screen.virtual_desktop()
    image = screen.grab(desktop)
    draw = ImageDraw.Draw(image)

    def rel(p: Point) -> tuple[int, int]:
        return p.x - desktop.left, p.y - desktop.top

    for region, colour in ((layout.view, "lime"), (layout.map_region, "cyan")):
        x, y = rel(Point(region.left, region.top))
        draw.rectangle([x, y, x + region.width, y + region.height], outline=colour, width=4)
    for point in (layout.map_hover, layout.guess_button, layout.continue_button):
        x, y = rel(point)
        draw.ellipse([x - 12, y - 12, x + 12, y + 12], outline="magenta", width=4)
    image.save(path)


def run_calibration(
    path: Path = DEFAULT_LAYOUT_PATH, preview_path: Path = Path("calibration_preview.png")
) -> Layout:
    print(
        "Calibration: open https://openguessr.com, start a Singleplayer game, and put the browser\n"
        "window where it will stay (re-run this if you move or resize it). This terminal may\n"
        "overlap the browser: after each Enter you get a short countdown to position the mouse."
    )
    with Controls() as controls, Screen() as screen:
        view_a = _record(
            controls, "1/7", "Street View area: TOP-LEFT corner (below the logo and Menu row)."
        )
        view_b = _record(
            controls,
            "2/7",
            "Street View area: BOTTOM-RIGHT corner (above the buttons, left of the minimap).",
        )
        hover = _record(controls, "3/7", "The collapsed minimap in the bottom-right corner.")
        map_a = _record(
            controls, "4/7", "Hover the minimap so it expands, then its TOP-LEFT corner."
        )
        map_b = _record(
            controls,
            "5/7",
            "Keep it expanded: its BOTTOM-RIGHT corner (map tiles only, above the bar below).",
        )
        guess = _record(controls, "6/7", "The Guess button.")
        print("\nNow drop any pin on the map and press Guess yourself, so the result screen shows.")
        cont = _record(controls, "7/7", "The Continue button on the result screen.")

        layout = Layout(
            view=Region.from_corners(view_a, view_b),
            map_hover=hover,
            map_region=Region.from_corners(map_a, map_b),
            guess_button=guess,
            continue_button=cont,
        )
        layout.save(path)
        _save_preview(screen, layout, preview_path)

    print(f"\nSaved {path}. Check {preview_path}: green = view, cyan = map, magenta = clicks.")
    return layout
