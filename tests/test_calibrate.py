from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from test_locate import NARROW, OPENED, draw_game

from geoguessr_ai import calibrate
from geoguessr_ai.calibrate import WrongGame, ensure_layout, find_layout
from geoguessr_ai.config import Layout, Point, Region
from geoguessr_ai.ocr import Word

LAPTOP = Region(0, 0, 1920, 1080)
DOCKED = Region(1920, 100, 1536, 864)
"""A second screen, where the game shows: its pixels are those of :func:`draw_game`, moved."""
TAB = Word("OpenGuessr - Free GeoGue", Region(300, 8, 180, 18))


class Desk:
    """Two screens, a mouse, and what OCR reads on them. The game shows on the docked screen
    once ``hidden_looks`` looks have passed, and while ``showing`` is ``"game"``; with ``"tab"``
    the laptop's screen shows only its browser tab, which a click switches to, and with
    ``"window"`` nothing of it, until its window is brought forward. Its map opens while the
    mouse is over the collapsed one, or the opened one."""

    def __init__(self, showing="game", party=False, hidden_looks=0, narrow_looks=0):
        self.showing, self.party, self.hidden_looks = showing, party, hidden_looks
        self.narrow_looks = narrow_looks
        self.mouse, self.clicks, self.grabbed, self.raised = Point(0, 0), [], None, 0
        self.collapsed, self.words = draw_game(party=party)
        self.opened, _ = draw_game(party=party, opened=True)

    def monitors(self):
        return [LAPTOP, DOCKED]

    def virtual_desktop(self):
        return Region(0, 0, DOCKED.right, LAPTOP.bottom)

    def _shows(self):
        return self.showing == "game" and not self.hidden_looks

    def grab(self, region):
        self.grabbed = region
        if region == LAPTOP:
            self.hidden_looks = max(self.hidden_looks - 1, 0)
        desktop = np.zeros((LAPTOP.bottom, DOCKED.right, 3), np.uint8)
        if self._shows():
            local = Point(self.mouse.x - DOCKED.left, self.mouse.y - DOCKED.top)
            over = OPENED.contains(local) or Region(1160, 577, 306, 155).contains(local)
            game = self.opened if over else self.collapsed
            desktop[DOCKED.top : DOCKED.bottom, DOCKED.left :] = game
        crop = desktop[region.top : region.bottom, region.left : region.right]
        return Image.fromarray(crop)

    def read_words(self, image):
        if self.grabbed == LAPTOP and self.showing == "tab":
            return [TAB]
        if not self._shows():
            return []
        if self.grabbed == DOCKED:
            if self.narrow_looks:  # a window too narrow, until it's widened
                self.narrow_looks -= 1
                return list(NARROW)
            return list(self.words)
        guess = Region(DOCKED.left + 1285, DOCKED.top + 813, 56, 30)
        if self.grabbed.contains(guess.center):
            left, top = guess.left - self.grabbed.left, guess.top - self.grabbed.top
            return [Word("Guess", Region(left, top, guess.width, guess.height))]
        return []

    def bring_forward(self):
        self.raised += 1
        if self.showing == "window":
            self.showing = "game"
        return True

    def move(self, p, duration=0.15):
        self.mouse = p

    def click(self, p):
        self.mouse = p
        self.clicks.append(p)
        if self.showing == "tab" and abs(p.y - TAB.box.center.y) < 5:
            self.showing = "game"

    def sleep(self, seconds):
        pass


def docked(left, top, width, height):
    return Region(DOCKED.left + left, DOCKED.top + top, width, height)


def test_finds_the_game_on_whichever_screen_shows_it_and_how_far_its_map_opens():
    desk = Desk()

    layout = find_layout(desk, desk, False, desk.read_words, desk.bring_forward)

    assert layout.view == docked(22, 70, 1129, 498)
    assert layout.map_region == docked(OPENED.left, OPENED.top, OPENED.width, OPENED.height)
    assert layout.guess_button == Point(DOCKED.left + 1313, DOCKED.top + 828)
    assert layout.minimap == docked(1160, 577, 306, 155)
    assert layout.map_hover == layout.minimap.center
    assert layout.page == docked(0, 0, 1536, 864)
    assert layout.continue_button is None and not layout.party  # Continue is found later
    assert desk.clicks == [] and desk.raised == 0


def test_finds_a_multiplayer_rooms_timer_and_chat_box():
    desk = Desk(party=True)

    layout = find_layout(desk, desk, True, desk.read_words)

    assert layout.party and layout.timer == Point(DOCKED.left + 1415, DOCKED.top + 760)
    (chat,) = layout.covered  # all of the chat box, from corner to corner
    assert chat.contains(Point(DOCKED.left + 12, DOCKED.top + 597))
    assert chat.contains(Point(DOCKED.left + 309, DOCKED.top + 786))
    assert layout.found_by == calibrate.VERSION


def test_switches_to_the_games_tab_when_the_game_isnt_showing(capsys):
    desk = Desk(showing="tab")

    layout = find_layout(desk, desk, False, desk.read_words, desk.bring_forward)

    assert desk.clicks == [TAB.box.center] and layout.view == docked(22, 70, 1129, 498)
    assert "tab" in capsys.readouterr().out and desk.raised == 0


def test_brings_the_games_window_forward_when_nothing_of_it_shows():
    desk = Desk(showing="window")

    layout = find_layout(desk, desk, False, desk.read_words, desk.bring_forward)

    assert desk.raised == 1 and layout is not None and desk.clicks == []


def test_tells_which_command_plays_the_game_that_shows():
    desk = Desk()
    with pytest.raises(WrongGame, match="geoguessr-ai play"):
        find_layout(desk, desk, True, desk.read_words)
    desk = Desk(party=True)
    with pytest.raises(WrongGame, match="geoguessr-ai friends"):
        find_layout(desk, desk, False, desk.read_words)


@pytest.fixture
def preview(tmp_path, monkeypatch):
    path = tmp_path / "preview.png"
    monkeypatch.setattr(calibrate, "PREVIEW_PATH", path)
    return path


def test_waits_for_the_game_to_show_then_saves_where_it_is(tmp_path, preview, capsys):
    desk, path = Desk(hidden_looks=3), tmp_path / "layout.json"
    (tmp_path / "layout-continue.png").write_bytes(b"from where the game was before")

    layout = ensure_layout(desk, desk, path, False, desk.read_words)

    out = capsys.readouterr().out
    assert out.count("Can't see an OpenGuessr round") == 1 and "saved where it is" in out
    assert Layout.load(path) == layout and preview.exists()
    assert not (tmp_path / "layout-continue.png").exists()  # snapshotted afresh when it shows


def test_says_when_the_window_is_too_narrow_to_play_in(tmp_path, preview, capsys):
    desk = Desk(narrow_looks=3)

    layout = ensure_layout(desk, desk, tmp_path / "layout.json", False, desk.read_words)

    out = capsys.readouterr().out
    assert out.count("too narrow") == 1 and "Can't see" not in out
    assert layout.view == docked(22, 70, 1129, 498) and desk.raised == 0


def test_keeps_the_saved_layout_while_the_game_is_still_there(tmp_path, preview):
    desk, path = Desk(), tmp_path / "layout.json"
    layout = ensure_layout(desk, desk, path, False, desk.read_words)
    desk.mouse = Point(0, 0)

    assert ensure_layout(desk, desk, path, False, desk.read_words) == layout
    assert desk.mouse == Point(0, 0)  # didn't open the map to look again

    desk.showing = "tab"  # gone to another tab: found again
    assert ensure_layout(desk, desk, path, False, desk.read_words) == layout
    assert desk.clicks == [TAB.box.center]


@pytest.mark.parametrize("kept_map", [True, False])
def test_finds_the_game_again_when_an_older_finder_found_it(tmp_path, preview, kept_map):
    desk, path = Desk(), tmp_path / "layout.json"
    layout = ensure_layout(desk, desk, path, False, desk.read_words)
    older = replace(layout, view=docked(800, 70, 351, 498), found_by=0)  # took the tab for the logo
    (older if kept_map else replace(older, minimap=None)).save(path)

    assert ensure_layout(desk, desk, path, False, desk.read_words) == layout


def test_keeps_a_layout_pointed_out_by_hand(tmp_path, preview):
    desk, path = Desk(), tmp_path / "layout.json"
    found = ensure_layout(desk, desk, path, False, desk.read_words)
    by_hand = replace(found, page=None, minimap=None, found_by=0)
    by_hand.save(path)

    assert ensure_layout(desk, desk, path, False, desk.read_words) == by_hand
