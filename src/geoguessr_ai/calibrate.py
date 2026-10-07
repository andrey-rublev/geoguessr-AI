"""One-time, interactive recording of where OpenGuessr's UI sits on your screen: in a
singleplayer game, or in a multiplayer room, which lays the game out differently."""

from __future__ import annotations

import time
from pathlib import Path

from PIL import ImageDraw

from .buttons import button_image_path, button_region
from .clock import parse_timer, timer_region
from .config import DEFAULT_LAYOUT_PATH, PARTY_LAYOUT_PATH, Layout, Point, Region
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

    boxes = [(layout.view, "lime"), (layout.map_region, "cyan")]
    boxes += [(region, "red") for region in layout.covered]
    if layout.timer is not None:
        boxes.append((timer_region(layout.timer), "yellow"))
    for region, colour in boxes:
        x, y = rel(Point(region.left, region.top))
        draw.rectangle([x, y, x + region.width, y + region.height], outline=colour, width=4)
    for point in (layout.map_hover, layout.guess_button, layout.continue_button):
        if point is None:
            continue
        x, y = rel(point)
        draw.ellipse([x - 12, y - 12, x + 12, y + 12], outline="magenta", width=4)
    image.save(path)


def run_calibration(
    path: Path = DEFAULT_LAYOUT_PATH, preview_path: Path = Path("calibration_preview.png")
) -> Layout:
    """Record a singleplayer game's layout to ``path``."""
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
        # Remember how the button looks, unhovered, so play can tell when an advert covers it.
        controls.move(view_a)
        time.sleep(0.5)
        screen.grab(button_region(cont)).save(button_image_path(path))

        layout = Layout(
            view=Region.from_corners(view_a, view_b),
            map_hover=hover,
            map_region=Region.from_corners(map_a, map_b),
            guess_button=guess,
            continue_button=cont,
        )
        layout.save(path)
        _save_preview(screen, layout, preview_path)

    print(
        f"\nSaved {path}. Check {preview_path}: green = view, cyan = map, magenta = clicks.\n"
        f"Check {button_image_path(path)} shows the Continue button, not an advert."
    )
    return layout


def run_party_calibration(
    path: Path = PARTY_LAYOUT_PATH, preview_path: Path = Path("calibration_preview_party.png")
) -> Layout:
    """Record a multiplayer room's layout to ``path``: no Continue button, since the host
    presses it, but the round's timer, and the chat box drawn over Street View."""
    print(
        "Multiplayer calibration: on https://openguessr.com host a room (Multiplayer > Host),\n"
        "have a friend or a second browser window join it, and Start, so that a round shows.\n"
        "Put the browser window where it will stay (re-run this if you move or resize it)."
    )
    with Controls() as controls, Screen() as screen:
        view_a = _record(
            controls, "1/9", "Street View area: TOP-LEFT corner (below the logo and Menu row)."
        )
        view_b = _record(
            controls,
            "2/9",
            "Street View area: BOTTOM-RIGHT corner (above the buttons, left of the map).",
        )
        chat_a = _record(controls, "3/9", "The chat box over Street View: its TOP-LEFT corner.")
        chat_b = _record(controls, "4/9", "The chat box: its BOTTOM-RIGHT corner.")
        hover = _record(controls, "5/9", "The collapsed map (or Map button), bottom right.")
        map_a = _record(controls, "6/9", "Hover the map so it opens, then its TOP-LEFT corner.")
        map_b = _record(
            controls,
            "7/9",
            "Keep it open: its BOTTOM-RIGHT corner (map tiles only, above the bar below).",
        )
        timer = _record(controls, "8/9", "Keep it open: the round's timer (like 00:42) in the bar.")
        guess = _record(controls, "9/9", "Keep it open: the Guess button.")
        _check_timer(screen, timer)

        try:
            covered: tuple[Region, ...] = (Region.from_corners(chat_a, chat_b),)
        except ValueError:
            print("  no chat box recorded; nothing will be blanked out")
            covered = ()
        layout = Layout(
            view=Region.from_corners(view_a, view_b),
            map_hover=hover,
            map_region=Region.from_corners(map_a, map_b),
            guess_button=guess,
            continue_button=None,
            timer=timer,
            covered=covered,
        )
        layout.save(path)
        _save_preview(screen, layout, preview_path)

    print(
        f"\nSaved {path}. Check {preview_path}: green = view, cyan = map, red = blanked out,\n"
        "yellow = timer, magenta = clicks. Then play with `geoguessr-ai party`."
    )
    return layout


def _check_timer(screen: Screen, timer: Point) -> None:
    """Read the timer just recorded, while the map is still open, to show it can be."""
    try:
        from .ocr import SignReader

        text = SignReader(models=["latin"]).read_line(screen.grab(timer_region(timer)))
    except ImportError:
        print("  can't read the timer without RapidOCR: install it with `pip install rapidocr`")
        return
    left = parse_timer(text)
    if left is None:
        print(f"  couldn't read a time there (read {text!r}); try step 8 again on the digits")
    else:
        print(f"  read the timer: {left:.0f} s left")
