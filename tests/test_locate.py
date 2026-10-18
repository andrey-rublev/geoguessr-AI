import cv2
import numpy as np
import pytest

from geoguessr_ai.config import Point, Region
from geoguessr_ai.locate import (
    find_continue,
    find_game,
    find_tab,
    measure_map,
    shows_guess,
    shows_narrow_game,
)
from geoguessr_ai.ocr import Word

PANEL = (50, 55, 60)
"""The colour of the game's dark, see-through panels."""
OPENED = Region(622, 257, 844, 475)
"""Where the map opens out to, in a 1536 by 864 window, as OpenGuessr draws it."""


def draw_game(party=False, opened=False, seed=0):
    """OpenGuessr in a 1536 by 864 window, laid out as it really is at that size, over a blurred
    Street View with a horizon right across it, and the words read off it. The map is
    collapsed, unless ``opened``; a multiplayer room has its timer and chat box."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (864, 1536, 3), dtype=np.uint8)
    rgb = cv2.GaussianBlur(noise, (0, 0), 3)
    rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
    rgb[400:] = np.clip(rgb[400:].astype(int) + 60, 0, 255)  # the horizon
    if opened:
        rgb[OPENED.top : OPENED.bottom, OPENED.left : OPENED.right] = (95, 205, 228)
        rgb[400:500, 900:1000] = (240, 236, 226)  # land
    else:
        rgb[577:732, 1160:1466] = (95, 205, 228)
        rgb[640:690, 1250:1300] = (240, 236, 226)
    bar_right = OPENED.right if opened else 1466
    rgb[732:787, (OPENED.left if opened else 1160) : bar_right] = PANEL
    rgb[800:853, 1160:1466] = PANEL  # round the Guess button
    rgb[800:853, 12:138] = PANEL  # round Return
    words = [
        Word("OpenGuessr", Region(22, 22, 164, 30)),
        Word("Menu", Region(1428, 24, 81, 28)),
        Word("Multiplayer" if party else "Singleplayer", Region(1169, 747, 103, 30)),
        Word("World", Region(1278, 750, 65, 21)),
        Word("Guess", Region(1285, 813, 56, 30)),
        Word("Return", Region(58, 812, 64, 30)),
        Word("Ads keep OpenGuessr free.", Region(663, 797, 210, 27)),
    ]
    if party:
        rgb[597:787, 12:310] = (52, 55, 64)  # the chat box
        rgb[741:779, 18:188] = (75, 78, 88)  # its message field
        words += [
            Word("00:32", Region(1377, 748, 77, 25)),
            Word("Server: Starting round 1.", Region(17, 702, 193, 27)),
            Word("Your message...", Region(23, 747, 126, 31)),
        ]
    return rgb, words


def test_finds_a_singleplayer_round_by_its_words_and_panels():
    rgb, words = draw_game()

    game = find_game(words, rgb, origin=Point(1920, 100))

    def at(left, top, width, height):
        return Region(1920 + left, 100 + top, width, height)

    assert not game.party and game.timer is None and game.chat is None
    assert game.guess_button == Point(1920 + 1313, 100 + 828)
    assert game.panel == at(1160, 577, 306, 155)  # the collapsed map, below the horizon
    assert game.bar_top == 100 + 732
    assert game.map_hover == game.panel.center
    assert game.view == at(22, 70, 1129, 498)  # clear of Menu, the map and what's below
    assert game.page == at(0, 0, 1536, 864)  # past Menu and the Guess button, to the edge


def in_browser(rgb, words, tab="OpenGuessr - Free GeoGue", chrome=120):
    """The game as a browser shows it, below its tabs and address bar, one tab named ``tab``."""
    bar = np.full((chrome, rgb.shape[1], 3), 32, np.uint8)
    moved = [
        Word(w.text, Region(w.box.left, w.box.top + chrome, w.box.width, w.box.height))
        for w in words
    ]
    return np.vstack([bar, rgb]), [Word(tab, Region(500, 10, 200, 20)), *moved]


@pytest.mark.parametrize("tab", ["OpenGuessr - Free GeoGue", "OpenGuessr"])
@pytest.mark.parametrize("menu", [True, False])
def test_the_browsers_tab_is_no_logo(tab, menu):
    rgb, words = in_browser(*draw_game(party=True), tab=tab)
    if not menu:
        words = [word for word in words if word.text != "Menu"]

    game = find_game(words, rgb)

    assert game.page.left == 0 and game.page.top == 120  # round the logo, not the tab
    assert game.view.left == 22 and game.view.width == 1129


def test_the_tab_is_no_logo_even_when_the_logo_cant_be_read():
    rgb, words = in_browser(*draw_game(party=True), tab="OpenGuessr")
    words = [word for word in words if word.text != "OpenGuessr" or word.box.top < 100]

    game = find_game(words, rgb)

    assert game.page.left < 30 and game.view.left < 40  # by Return, below it


def test_finds_a_multiplayer_room_its_timer_and_its_chat_box():
    rgb, words = draw_game(party=True)

    game = find_game(words, rgb)

    assert game.party and game.timer == Point(1415, 760)
    chat, h = Region(12, 597, 298, 190), 30
    assert game.chat.left <= chat.left and game.chat.top <= chat.top
    assert game.chat.right >= chat.right and game.chat.bottom >= chat.bottom
    assert chat.left - game.chat.left < h and chat.top - game.chat.top < h
    assert game.chat.right - chat.right < h and game.chat.bottom - chat.bottom < h
    assert game.chat.bottom <= 800  # Return's panel, below it


def test_covers_the_whole_chat_box_however_what_shows_through_shades_it():
    rgb, words = draw_game(party=True)
    shade = np.linspace(0, 28, 298).astype(np.uint8)[None, :, None]
    rgb[597:787, 12:310] = (57, 52, 50) - np.minimum(shade, (28, 23, 12))  # as a real one did

    assert find_game(words, rgb).chat.right >= 310


def test_before_a_round_the_timer_is_taken_to_be_at_the_bars_right_end():
    rgb, words = draw_game(party=True)
    words = [word for word in words if word.text != "00:32"]  # a lobby shows none yet

    timer = find_game(words, rgb).timer

    assert abs(timer.x - 1415) <= 10 and timer.y == 762


def test_something_as_dark_as_the_panel_beside_it_doesnt_widen_it():
    rgb, words = draw_game()
    rgb[700:800, 1060:1160] = PANEL  # a dark car, right up against the bar

    game = find_game(words, rgb)

    assert game.panel.left == 1160 and game.panel.right == 1466


@pytest.mark.parametrize("missing", ["Guess", "Singleplayer", "Menu"])
def test_doesnt_see_the_game_without_its_words(missing):
    rgb, words = draw_game()
    words = [word for word in words if word.text != missing]
    if missing == "Menu":  # the logo will do instead, but not neither
        words = [word for word in words if word.text != "OpenGuessr"]

    assert find_game(words, rgb) is None


def test_measures_how_far_the_map_opens():
    collapsed, words = draw_game()
    opened, _ = draw_game(opened=True)
    game = find_game(words, collapsed)
    area = Region(0, 0, game.panel.right, game.bar_top)

    found = measure_map(collapsed[:732, :1466], opened[:732, :1466], area, game)

    assert found == OPENED


def test_finds_the_tab_continue_and_guess_by_name():
    words = [
        Word("OpenGuessr - Free GeoGue", Region(500, 10, 200, 20)),
        Word("Continue", Region(1398, 813, 68, 30)),
        Word("Locked in", Region(1269, 811, 86, 30)),
    ]

    assert find_tab(words, Point(100, 0)) == Point(700, 20)
    assert find_continue(words) == Point(1432, 828)
    assert shows_guess(words) and not shows_guess(words[:2])
    assert find_tab([Word("OpenGuessr", Region(0, 0, 9, 9))]) is None  # the logo is no tab


NARROW = [  # as read off an 800 by 600 window
    Word("OpenGuessr", Region(21, 21, 165, 31)),
    Word("Menu", Region(691, 24, 80, 27)),
    Word("Return", Region(59, 549, 61, 28)),
    Word("Map", Region(590, 550, 63, 28)),
    Word("Keyboard snortcuts", Region(500, 587, 90, 12)),
]


def test_tells_a_window_too_narrow_to_play_in():
    assert shows_narrow_game(NARROW)
    assert not shows_narrow_game(draw_game()[1])
    assert not shows_narrow_game([*NARROW, Word("Guess", Region(600, 500, 56, 30))])
    sign = Word("MAP", Region(300, 200, 60, 28))  # on a shop, say: nowhere near Google's line
    assert not shows_narrow_game([w for w in NARROW if w.text != "Map"] + [sign])
