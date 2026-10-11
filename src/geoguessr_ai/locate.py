"""Find OpenGuessr on screen by itself, from pixels: the words the game writes, like Menu, Guess
and the bar under its map, and the edges of its panels. So calibration needs nobody to point
things out, and is done again whenever the game has moved, as when a laptop is docked.

The game lays itself out alike in a singleplayer round, a multiplayer round and a multiplayer
room's lobby: the map collapsed in the bottom right, above a bar that says Singleplayer or
Multiplayer, with the Guess button (Locked in, in a lobby) below; OpenGuessr's name at the top
left and Menu at the top right; Return at the bottom left, and in a multiplayer room the chat box
above it. The map's panel scales with the window, but its words don't, so sizes are measured in
heights of the bar's words.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .clock import parse_timer
from .config import Point, Region
from .ocr import Word

GUESS = re.compile(r"^\W*(guess|locked\s*in)\W*$", re.I)
BAR = re.compile(r"(single|multi)\s*player", re.I)
MENU = re.compile(r"menu", re.I)
LOGO = re.compile(r"open\s*guess", re.I)
RETURN = re.compile(r"^\W*return\W*$", re.I)
CHAT = re.compile(r"your\s*mess", re.I)
CONTINUE = re.compile(r"^\W*continue\W*$", re.I)
TAB = re.compile(r"open\s*guessr\s*\W\s*free", re.I)
"""The browser tab's title, "OpenGuessr - Free GeoGuessr Alternative", however cut short."""

PANEL_TOLERANCE = 18
"""How far a pixel's colour may be from a panel's to count as part of it: the panels are see-
through, so a little tinted by what's behind them, and the chat box shades by up to 15."""
PANEL_INSET = 0.3
"""How far inside its panel the bar's word starts, in word heights (9 pixels of 30)."""
EDGE_STEP, EDGE_SHARE = 25, 0.7
"""The collapsed map's top edge: a row at least this share of whose pixels differ from the row
above by at least this much (of 255). Of six screenshots it scored 0.88 to 0.99, and Street
View's own edges, like the horizon, at most 0.95 but above it."""
CHANGED = 40
"""How much a pixel must change as the map opens to count as part of the opened map."""
TIMER_FROM_RIGHT = 1.7
"""Where the timer sits in a multiplayer room's bar, in word heights from its right end, for
when no round is on to show it."""


@dataclass(frozen=True)
class Sighting:
    """The game as found in a screenshot, its map collapsed, in screen pixels."""

    party: bool
    """Whether the bar under the map says Multiplayer."""
    guess_button: Point
    panel: Region
    """The collapsed map, as wide as the panel the Guess button sits in."""
    bar_top: int
    """Where the bar under the map begins: the bottom of the map, collapsed or opened."""
    view: Region
    """Street View, from below the top row of buttons to the map, clear of both."""
    page: Region
    text_height: int
    timer: Point | None = None
    chat: Region | None = None

    @property
    def map_hover(self) -> Point:
        return self.panel.center


def find_game(
    words: Sequence[Word], rgb: np.ndarray, origin: Point = Point(0, 0)
) -> Sighting | None:
    """Where the game shows in ``rgb``, a screenshot of the screen from ``origin``, going by the
    ``words`` read in it (in its pixels), or None if it doesn't show there."""
    pair = _guess_and_bar(words)
    if pair is None:
        return None
    guess, bar = pair
    h = bar.box.height
    colour = _panel_colour(rgb, bar.box)
    if colour is None:
        return None
    left = _panel_left(rgb, bar.box, colour, h)
    right = 2 * guess.box.center.x - left
    if right <= bar.box.right:
        return None
    bar_top = _bar_top(rgb, bar.box, right, colour, h)
    map_top = _map_top(rgb, left, right, bar_top)

    above = [w for w in words if w.box.bottom < map_top]
    menus = [w for w in above if MENU.search(w.text) and w.box.right >= right - h]
    logos = [w for w in above if LOGO.search(w.text) and w.box.right < left]
    backs = [w for w in words if RETURN.search(w.text) and w.box.top > bar.box.top]
    menu = min(menus, key=lambda w: w.box.top, default=None)
    logo = min(logos, key=lambda w: w.box.top, default=None)
    back = min(backs, key=lambda w: w.box.left, default=None)
    tops = [w for w in (menu, logo) if w is not None]
    if not tops or (logo is None and back is None):
        return None
    pad = round(0.75 * h)
    page_left = logo.box.left - pad if logo is not None else back.box.left - h - pad
    page_left = max(page_left, 0)
    page_top = max(min(w.box.top for w in tops) - pad, 0)
    page_right = min(max(right, menu.box.right if menu else right) + h, rgb.shape[1])
    page_bottom = min(max(guess.box.bottom, back.box.bottom if back else 0) + pad, rgb.shape[0])

    view_top = max(w.box.bottom for w in tops) + round(0.6 * h)
    view_left, view_right = page_left + pad, left - round(0.3 * h)
    view_bottom = map_top - round(0.3 * h)
    if view_right - view_left < 8 * h or view_bottom - view_top < 4 * h:
        return None

    party = BAR.search(bar.text)[1].lower() == "multi"
    timer = chat = None
    if party:
        timers = [
            w
            for w in words
            if parse_timer(w.text) is not None
            and abs(w.box.center.y - bar.box.center.y) < h / 2
            and bar.box.left < w.box.center.x < right
        ]
        if timers:
            timer = timers[0].box.center
        else:  # in the lobby, before a round starts
            timer = Point(right - round(TIMER_FROM_RIGHT * h), bar.box.center.y)
        chats = [w for w in words if CHAT.search(w.text) and w.box.right < left]
        if chats:
            bottom = back.box.top if back is not None else page_bottom
            chat = _chat_box(rgb, chats[0].box, h, bottom)

    def screen(region: Region) -> Region:
        return Region(origin.x + region.left, origin.y + region.top, region.width, region.height)

    def point(p: Point) -> Point:
        return Point(origin.x + p.x, origin.y + p.y)

    return Sighting(
        party=party,
        guess_button=point(guess.box.center),
        panel=screen(Region(left, map_top, right - left, bar_top - map_top)),
        bar_top=origin.y + bar_top,
        view=screen(Region(view_left, view_top, view_right - view_left, view_bottom - view_top)),
        page=screen(Region(page_left, page_top, page_right - page_left, page_bottom - page_top)),
        text_height=h,
        timer=None if timer is None else point(timer),
        chat=None if chat is None else screen(chat),
    )


def measure_map(
    collapsed: np.ndarray, opened: np.ndarray, area: Region, sighting: Sighting
) -> Region | None:
    """The opened map, from screenshots of ``area`` before and after it opened: what changed,
    out from where the collapsed map's right side meets the bar. It opens up and to the left."""
    changed = np.abs(collapsed.astype(np.int16) - opened.astype(np.int16)).max(axis=2) > CHANGED
    panel = sighting.panel
    right, bottom = panel.right - area.left, sighting.bar_top - area.top
    near = max(round(0.04 * panel.width), 1)
    beside = changed[panel.top - area.top : bottom - near]  # level with the collapsed map
    left = _reach(beside.mean(axis=0), panel.left - area.left - 1)
    if left is None:
        return None
    across = changed[:, left + near : right - near]  # as wide as the opened map
    top = _reach(across.mean(axis=1), panel.top - area.top - 1)
    if top is None:
        return None
    return Region(area.left + left, area.top + top, right - left, bottom - top)


def find_tab(words: Sequence[Word], origin: Point = Point(0, 0)) -> Point | None:
    """Where OpenGuessr's browser tab is, by its title, to click it."""
    for word in words:
        if TAB.search(word.text):
            return Point(origin.x + word.box.center.x, origin.y + word.box.center.y)
    return None


def find_continue(words: Sequence[Word], origin: Point = Point(0, 0)) -> Point | None:
    """Where the result screen's Continue button is, by its name: the rightmost one."""
    found = [w for w in words if CONTINUE.search(w.text)]
    if not found:
        return None
    center = max(found, key=lambda w: w.box.right).box.center
    return Point(origin.x + center.x, origin.y + center.y)


def shows_guess(words: Sequence[Word]) -> bool:
    """Whether the Guess button (Locked in, in a multiplayer lobby) is among ``words``."""
    return any(GUESS.search(word.text) for word in words)


def _guess_and_bar(words: Sequence[Word]) -> tuple[Word, Word] | None:
    """The Guess button's word, and the bar's just above it, which starts at the panel's left."""
    best = None
    for guess in (w for w in words if GUESS.search(w.text)):
        for bar in (w for w in words if BAR.search(w.text)):
            h = bar.box.height
            gap = guess.box.center.y - bar.box.center.y
            if not 0 < gap < 6 * h or not bar.box.left < guess.box.center.x < bar.box.left + 12 * h:
                continue
            if best is None or gap < best[0]:
                best = (gap, guess, bar)
    return None if best is None else (best[1], best[2])


def _matches(pixels: np.ndarray, colour: np.ndarray) -> np.ndarray:
    return np.abs(pixels.astype(np.int16) - colour).max(axis=-1) <= PANEL_TOLERANCE


def _panel_colour(rgb: np.ndarray, box: Region) -> np.ndarray | None:
    """The colour of the dark panel round the light word in ``box``."""
    patch = rgb[box.top : box.bottom, box.left : box.right, :3].reshape(-1, 3)
    dark = patch[patch.max(axis=1) < 120]
    if len(dark) < 0.3 * max(len(patch), 1):
        return None
    return np.median(dark, axis=0).astype(np.int16)


def _panel_left(rgb: np.ndarray, box: Region, colour: np.ndarray, h: int) -> int:
    """The bar's left edge: as far left of its word as its colour goes, which is a little way.
    If it goes on, something as dark lies beyond, like a car: the usual way, then."""
    rows = rgb[box.center.y - h // 4 : box.center.y + h // 4 + 1]
    limit = max(box.left - round(2 * PANEL_INSET * h), 0)
    x = box.left - 1
    while x > limit and _matches(rows[:, x], colour).mean() >= 0.5:
        x -= 1
    return x + 1 if x > limit else box.left - round(PANEL_INSET * h)


def _bar_top(rgb: np.ndarray, box: Region, right: int, colour: np.ndarray, h: int) -> int:
    """The bar's top edge: as far above its word as its colour goes, column by column."""
    stops = []
    for x in range(box.left, min(box.right, right), max(h // 10, 1)):
        y = box.top - 1
        while y > max(box.top - h, 0) and _matches(rgb[y, x], colour):
            y -= 1
        stops.append(y + 1)
    return int(np.median(stops)) if stops else box.top


def _map_top(rgb: np.ndarray, left: int, right: int, bar_top: int) -> int:
    """The collapsed map's top edge: the lowest straight edge right across it, where Street
    View gives way to the map. Half as high as it is wide if none shows."""
    width = right - left
    strip = rgb[:, left + width // 10 : right - width // 10, :3].astype(np.int16)
    for y in range(bar_top - round(0.3 * width), max(bar_top - round(0.9 * width), 1), -1):
        if (np.abs(strip[y] - strip[y - 1]).max(axis=-1) > EDGE_STEP).mean() >= EDGE_SHARE:
            return y
    return bar_top - round(0.5 * width)


def _chat_box(rgb: np.ndarray, box: Region, h: int, bottom: int) -> Region:
    """The chat box round its message field, whose word is in ``box``: a panel as wide as the
    row of its colour just above the field, and as tall as its colour goes near its right side,
    where messages seldom reach, but not below ``bottom``, where Return is."""
    rgb = rgb[..., :3]
    y = box.top - h // 3
    colour = np.median(rgb[y - 1 : y + 2, box.left : box.left + h].reshape(-1, 3), axis=0)
    colour = colour.astype(np.int16)
    row = _matches(rgb[y], colour)
    left = right = box.left + h // 2
    while left > 0 and row[left - 1]:
        left -= 1
    while right < len(row) - 1 and row[right + 1]:
        right += 1
    columns = rgb[:, max(right - h // 3, left) : right + 1]
    top = bottom_edge = y
    while top > max(y - 10 * h, 0) and _matches(columns[top - 1], colour).mean() >= 0.5:
        top -= 1
    while bottom_edge < bottom - 1 and _matches(columns[bottom_edge + 1], colour).mean() >= 0.5:
        bottom_edge += 1
    if right - left < 4 * h:  # not the panel's colour after all: just round the field
        return Region(box.left - h // 2, box.top - h, box.width + h, 2 * h)
    return Region(left, top, right + 1 - left, bottom_edge + 1 - top)


def _reach(share: np.ndarray, start: int, gap: int = 4) -> int | None:
    """From ``start`` down to 0, the last place where ``share`` is at least a half, over gaps
    of up to ``gap`` where it isn't."""
    last, misses = None, 0
    for i in range(min(start, len(share) - 1), -1, -1):
        if share[i] >= 0.5:
            last, misses = i, 0
        else:
            misses += 1
            if misses > gap:
                break
    return last
