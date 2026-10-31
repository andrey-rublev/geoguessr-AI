"""Where OpenGuessr's UI sits on your screen: found by itself (see :mod:`.locate`) whenever the
game isn't where it was last time, or recorded by hand, by hovering over each part in turn. A
multiplayer room lays the game out a little differently from a singleplayer game, so the two are
kept apart."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .buttons import button_image_path, button_region
from .clock import parse_timer, timer_region
from .config import DEFAULT_LAYOUT_PATH, PARTY_LAYOUT_PATH, Layout, Point, Region
from .controls import Controls
from .locate import (
    VERSION,
    Sighting,
    find_game,
    find_tab,
    measure_map,
    shows_guess,
    shows_narrow_game,
)
from .mapcal import MapNotFound, locate_world
from .ocr import Word
from .screen import Screen

COUNTDOWN_S = 3
LOOK_AGAIN_S = 3.0
"""How often to look again for a game that doesn't show yet."""
TAB_WAIT_S = 2.0
"""How long a browser takes to show a tab, or a window, once asked to."""
PREVIEW_PATH = Path("calibration_preview.png")

ReadWords = Callable[[Image.Image], list[Word]]


class WrongGame(Exception):
    """The game on screen is a multiplayer room where a singleplayer game was wanted, or the
    other way round."""


class TooNarrow(Exception):
    """The game shows, but laid out for a window too narrow for the bot to play."""


def ensure_layout(
    screen,
    controls,
    path: Path,
    party: bool,
    read_words: ReadWords,
    *,
    again: bool = False,
    bring_forward: Callable[[], bool] | None = None,
    expand_wait: float = 1.0,
    loading_s: float = 0.0,
) -> Layout:
    """The game's layout: the one saved at ``path``, while the game still shows there, or else
    found afresh, waiting until the game shows, and saved there. ``again`` finds it afresh
    regardless. Raises :class:`WrongGame` if the game found is the other kind. ``loading_s``: how
    long a game just started may take to show before anyone is asked to start one."""
    if not again and Path(path).exists():
        layout = Layout.load(path)
        outdated = layout.page is not None and layout.found_by < VERSION  # by an older finder
        if outdated:
            print("Looking for the game again: the bot finds it better now.")
        elif layout.party == party and still_there(screen, read_words, layout):
            return layout
        else:
            print("The game isn't where it was last time: looking for it.")
    kind = "start a Singleplayer game"
    if party:
        kind = "join a multiplayer room, and its host starts one"
    said, waited = None, 0.0
    while True:
        try:
            layout = find_layout(screen, controls, party, read_words, bring_forward, expand_wait)
            why = f"Can't see an OpenGuessr round on screen yet: {kind}."
        except TooNarrow as narrow:
            layout, why = None, str(narrow)
        if layout is not None:
            break
        if why != said and waited >= loading_s:  # each once
            print(f"{why} ({getattr(controls, 'stop_name', 'F8')} stops.)")
            said = why
        controls.sleep(LOOK_AGAIN_S)
        waited += LOOK_AGAIN_S
    layout.save(path)
    button_image_path(path).unlink(missing_ok=True)  # the bot snapshots Continue when it shows
    _save_preview(screen, layout, PREVIEW_PATH)
    print(f"Found the game, and saved where it is to {path} (see {PREVIEW_PATH}).")
    return layout


def still_there(screen, read_words: ReadWords, layout: Layout) -> bool:
    """Whether the Guess button (Locked in, in a multiplayer lobby) is where ``layout`` has it."""
    guess = layout.guess_button
    around = Region(guess.x - 240, guess.y - 60, 480, 120)
    return shows_guess(read_words(screen.grab(around)))


def find_layout(
    screen,
    controls,
    party: bool,
    read_words: ReadWords,
    bring_forward: Callable[[], bool] | None = None,
    expand_wait: float = 1.0,
) -> Layout | None:
    """Find the game on whichever screen shows it; if none does, click its browser tab, or have
    ``bring_forward`` bring its window to the front, and look again. Then hover over the map
    to see how far it opens. None if the game can't be found; raises :class:`TooNarrow` if it
    shows in a window too narrow to play it in."""
    sightings, tabs, narrow = _look(screen, read_words)
    if not sightings and tabs:
        print("Found OpenGuessr's tab: switching to it.")
        controls.click(tabs[0])
        controls.sleep(TAB_WAIT_S)
        sightings, tabs, narrow = _look(screen, read_words)
    if not sightings and not narrow and bring_forward is not None and bring_forward():
        controls.sleep(TAB_WAIT_S)
        sightings, tabs, narrow = _look(screen, read_words)
    if not sightings and narrow:
        raise TooNarrow(
            "OpenGuessr's window is too narrow: it hides the map behind a Map button, which the "
            "bot can't play. Widen the browser window until the map shows in its bottom right."
        )
    if not sightings:
        return None
    wanted = [s for s in sightings if s.party == party]
    if not wanted:
        found, command = ("a multiplayer room", "friends")
        if party:
            found, command = ("a singleplayer game", "play")
        raise WrongGame(f"OpenGuessr is showing {found}: play it with `geoguessr-ai {command}`.")
    if len(wanted) > 1:
        print(f"OpenGuessr shows {len(wanted)} times on screen: using the first. Hide the others.")
    sighting = wanted[0]
    return Layout(
        view=sighting.view,
        map_hover=sighting.map_hover,
        map_region=_open_map(screen, controls, sighting, expand_wait),
        guess_button=sighting.guess_button,
        continue_button=None,  # found on the first result screen
        timer=sighting.timer,
        covered=() if sighting.chat is None else (sighting.chat,),
        page=sighting.page,
        party=party,
        minimap=sighting.panel,
        found_by=VERSION,
    )


def _look(screen, read_words: ReadWords) -> tuple[list[Sighting], list[Point], bool]:
    """The game wherever it shows, on any screen, where OpenGuessr's browser tabs are, and
    whether it shows anywhere laid out for a window too narrow to play it in."""
    sightings, tabs, narrow = [], [], False
    for monitor in screen.monitors():
        image = screen.grab(monitor)
        words = read_words(image)
        corner = Point(monitor.left, monitor.top)
        sighting = find_game(words, np.asarray(image.convert("RGB")), corner)
        if sighting is not None:
            sightings.append(sighting)
        elif shows_narrow_game(words):
            narrow = True
        elif (tab := find_tab(words, corner)) is not None:
            tabs.append(tab)
    return sightings, tabs, narrow


MAP_LOOKS = 3
"""How many times at most to watch the map open, for two looks that agree."""


def _open_map(screen, controls, sighting: Sighting, expand_wait: float) -> Region:
    """Hover over the map and see how far it opens, from what changes on screen as it does.
    Anything else that changes meanwhile can only make that look bigger: in a live multiplayer
    room, a chat message and Street View still drawing took it 340 pixels too far left, onto
    Street View, where the map's own check still saw a world. So it is watched until two looks
    agree, or else the smallest is kept. If it can't be told at all, take it to open to twice
    its size, which it opens to at least: that much of it is enough to place a pin."""
    page, panel = sighting.page, sighting.panel
    area = Region(page.left, page.top, panel.right - page.left, sighting.bar_top - page.top)
    seen: list[tuple[Region, np.ndarray]] = []
    for _ in range(MAP_LOOKS):
        controls.move(sighting.view.center)
        controls.sleep(expand_wait)
        collapsed = np.asarray(screen.grab(area).convert("RGB"))
        controls.move(sighting.map_hover)
        controls.move(sighting.map_hover.offset(-panel.width / 4, 0))  # stay over it, so it stays
        controls.sleep(expand_wait)
        opened = np.asarray(screen.grab(area).convert("RGB"))
        controls.leave(sighting.view.center)
        found = measure_map(collapsed, opened, area, sighting)
        if found is None or found.width <= panel.width or found.height <= panel.height:
            continue
        if any(_agree(found, before) for before, _ in seen):
            seen = [(found, opened)]
            break
        seen.append((found, opened))
    if not seen:
        print("  couldn't see how far the map opens; if guesses go astray, find the game again")
        width, height = 2 * panel.width, 2 * panel.height
        return Region(panel.right - width, sighting.bar_top - height, width, height)
    found, opened = min(seen, key=lambda look: look[0].width * look[0].height)
    top, left = found.top - area.top, found.left - area.left
    try:
        locate_world(np.ascontiguousarray(opened[top:, left:][: found.height, : found.width]))
    except MapNotFound as exc:
        print(f"  the opened map doesn't look like the world ({exc}); check {PREVIEW_PATH}")
    return found


def _agree(a: Region, b: Region, slack: int = 4) -> bool:
    return abs(a.left - b.left) <= slack and abs(a.top - b.top) <= slack


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

    boxes = [(layout.picture, "lime"), (layout.map_region, "cyan")]
    boxes += [(region, "red") for region in layout.hidden]
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
        "Manual calibration: open https://openguessr.com, start a Singleplayer game, and put the\n"
        "browser window where it will stay (re-run this if you move or resize it). This terminal\n"
        "may overlap the browser: after each Enter you get a short countdown to position the mouse."
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
        "Manual multiplayer calibration: on https://openguessr.com host a room (Multiplayer >\n"
        "Host), have a friend or a second browser window join it, and Start, so that a round\n"
        "shows. Put the browser window where it will stay (re-run this if you move or resize it)."
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
            party=True,
        )
        layout.save(path)
        _save_preview(screen, layout, preview_path)

    print(
        f"\nSaved {path}. Check {preview_path}: green = view, cyan = map, red = blanked out,\n"
        "yellow = timer, magenta = clicks. Then play with `geoguessr-ai friends`."
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
