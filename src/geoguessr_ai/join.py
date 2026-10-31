"""Join a friend's multiplayer room by its code, or start a Singleplayer game, in a browser of the
bot's own, as a person would: by the words on screen, read with OCR, never by the page's code.

OpenGuessr's start page shows its maps under Singleplayer: World starts a game of the whole world.

OpenGuessr's multiplayer page (:data:`ROOMS_URL`) shows Join beside Host. Join opens six boxes
for the code, like EW2-78U, which take it typed once the first is clicked, over a Join button
that lights up once all six are filled. After it, the game may ask to confirm the name it picked
for a guest. If the code page still shows after that, the room didn't let the bot in. And
OpenGuessr's site may stop to check that a person is there, which only a person can answer: the
bot says so and waits.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
from PIL import Image

from .config import Point, Region
from .ocr import Word

ROOMS_URL = "https://openguessr.com/multiplayer"
CODE_LETTERS = 6
FIRST_BOX = (-5.0, -3.3)
"""Where the middle of the first of the code's boxes is from the middle of the Join button under
them, in heights of its word, as measured in the bot's browser."""
LIT_RED = 200
"""How red the Join button under the boxes is once all of the code is in: about 255, against
about 145 while it waits for some."""
LOOK_S = 1.0
"""Seconds between looks at the screen while waiting for the next step to show."""
WAIT_S = 20.0
"""How long each step may take to show."""
CONFIRM_WAIT_S = 8.0
"""How long the game may take to ask to confirm a guest's name, which it doesn't always do."""

# Underscores too, which OCR adds after a word now and then, as in Join_.
_JOIN = re.compile(r"^[\W_]*join[\W_]*$", re.I)
_HOST = re.compile(r"^[\W_]*host[\W_]*$", re.I)
_CONFIRM = re.compile(r"^[\W_]*confirm[\W_]*$", re.I)
_SINGLE = re.compile(r"single\s*player", re.I)
_WORLD = re.compile(r"^[\W_]*world[\W_]*$", re.I)
_SKIP = re.compile(r"\bskip\b", re.I)
"""The welcome's "No, skip to menu", which a browser new to the game may be shown first."""
_CHECK = re.compile(r"verify you are human|security verification", re.I)
"""What the site's check that a person is there says, in place of the page asked for."""
STILL_THERE = 1.0
"""How far, in heights of its word, the code page's Join may have moved and be the same one."""

ReadWords = Callable[[Image.Image], list[Word]]


def room_code(text: str) -> str:
    """``text`` as a room's code: six letters and digits, in capitals, like EW278U for ew2-78u."""
    code = re.sub(r"[^0-9a-z]", "", text, flags=re.I).upper()
    if len(code) != CODE_LETTERS:
        raise ValueError(
            f"{text!r} isn't a room's code: one is six letters and digits, like EW2-78U"
        )
    return code


def start_solo(screen, controls, read_words: ReadWords, wait: float = WAIT_S) -> bool:
    """Start a Singleplayer game of the whole world from OpenGuessr's start page, showing on
    ``screen``. Whether it found where to."""
    page = screen.virtual_desktop()

    def start_page(words: Sequence[Word]) -> bool:
        if (skip := _find(words, _SKIP)) is not None:  # a welcome first: pass it by
            controls.click(skip.box.center)
            return False
        return _find(words, _WORLD) is not None and _find(words, _SINGLE) is not None

    if (words := _wait_for(screen, controls, read_words, page, start_page, wait)) is None:
        print("Couldn't find World on OpenGuessr's start page")
        return False
    print("Starting a Singleplayer game")
    controls.click(_find(words, _WORLD).box.center)
    return True


def join_room(screen, controls, read_words: ReadWords, code: str, wait: float = WAIT_S) -> bool:
    """Join the room ``code`` (see :func:`room_code`) from OpenGuessr's multiplayer page, showing
    on ``screen``. Whether it got as far as pressing Join with the code in its boxes."""
    page = screen.virtual_desktop()

    def menu(words: Sequence[Word]) -> bool:
        if (skip := _find(words, _SKIP)) is not None:  # a welcome first: pass it by
            controls.click(skip.box.center)
            return False
        return _find(words, _JOIN) is not None and _find(words, _HOST) is not None

    if (words := _wait_for(screen, controls, read_words, page, menu, wait)) is None:
        print("Couldn't find Join on OpenGuessr's multiplayer page")
        return False
    controls.click(_find(words, _JOIN).box.center)

    def boxes(words: Sequence[Word]) -> bool:
        return _find(words, _JOIN) is not None and _find(words, _HOST) is None

    if (words := _wait_for(screen, controls, read_words, page, boxes, wait)) is None:
        print("Couldn't find the boxes for the room's code")
        return False
    join = _find(words, _JOIN).box
    unit = join.height
    controls.click(
        Point(
            round(join.center.x + FIRST_BOX[0] * unit), round(join.center.y + FIRST_BOX[1] * unit)
        )
    )
    for letter in code:
        controls.press(letter)
    controls.sleep(LOOK_S)
    button = Region(join.left - unit, join.top, join.width + 2 * unit, join.height)
    if not lit(screen.grab(button)):
        print(f"Couldn't type the code {code} into its boxes")
        return False
    controls.click(join.center)
    print(f"Joining room {code}")

    def asks_name(words: Sequence[Word]) -> bool:
        return _find(words, _CONFIRM) is not None

    if confirm := _wait_for(screen, controls, read_words, page, asks_name, CONFIRM_WAIT_S):
        controls.click(_find(confirm, _CONFIRM).box.center)
        controls.sleep(LOOK_S)
    again = _find(_words(screen, read_words, page), _JOIN)
    if again is not None and _near(again.box.center, join.center, STILL_THERE * unit):
        print(f"Couldn't join room {code}: check its code, and that the room is still open")
        return False
    print(f"In room {code}")
    return True


def lit(button: Image.Image) -> bool:
    """Whether a red button is bright, as the code page's Join is once it can be pressed."""
    rgb = np.asarray(button.convert("RGB")).reshape(-1, 3).astype(int)
    red = rgb[rgb[:, 0] > rgb[:, 1] + 40]
    return len(red) > 0 and float(np.median(red[:, 0])) >= LIT_RED


def _find(words: Sequence[Word], pattern: re.Pattern) -> Word | None:
    return next((word for word in words if pattern.search(word.text)), None)


def _near(a: Point, b: Point, within: float) -> bool:
    return abs(a.x - b.x) <= within and abs(a.y - b.y) <= within


def _words(screen, read_words: ReadWords, page: Region) -> list[Word]:
    """The words on ``page``, placed on the screen."""
    return [
        Word(w.text, replace(w.box, left=w.box.left + page.left, top=w.box.top + page.top))
        for w in read_words(screen.grab(page))
    ]


def _wait_for(
    screen,
    controls,
    read_words: ReadWords,
    page: Region,
    ready: Callable[[Sequence[Word]], bool],
    wait: float,
) -> list[Word] | None:
    """The words on ``page`` once ``ready`` says they are what was waited for, looking every
    :data:`LOOK_S` seconds for up to ``wait`` seconds; None if they never are. The site's check
    that a person is there takes none of that time: it says so once, and waits for them."""
    looks, said = 0, False
    while True:
        words = _words(screen, read_words, page)
        if ready(words):
            return words
        if _find(words, _CHECK) is not None:
            if not said:
                print(
                    "OpenGuessr is checking that a person is there: tick its box in the bot's "
                    "browser (with --hidden, at http://localhost:6080/vnc.html). The bot waits."
                )
            said = True
        else:
            looks += 1
            if looks > int(wait / LOOK_S):
                return None
        controls.sleep(LOOK_S)
