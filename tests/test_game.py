import json

import cv2
import numpy as np
import pytest
from PIL import Image
from test_buttons import draw_button
from test_compass import draw_compass
from test_sun import sky

from geoguessr_ai import game as game_module
from geoguessr_ai.buttons import button_region
from geoguessr_ai.camera import Camera
from geoguessr_ai.clock import timer_region
from geoguessr_ai.config import Layout, Point, Region
from geoguessr_ai.game import BotSettings, OpenGuessrBot
from geoguessr_ai.knowledge.text import TextLine
from geoguessr_ai.mapcal import MapProjection
from geoguessr_ai.minimap import Placement
from geoguessr_ai.model.predictor import Guess
from geoguessr_ai.result import Answer, Reading, result_region

LAYOUT = Layout(
    view=Region(0, 100, 1600, 700),
    map_hover=Point(1800, 950),
    map_region=Region(1000, 300, 845, 633),
    guess_button=Point(1800, 1040),
    continue_button=Point(960, 1040),
)
DESKTOP = Region(0, 0, 2000, 1200)
LAGOS = (6.45, 3.39)


class FakeScreen:
    """Street View that stays black for the first ``blank_grabs`` looks, and is still blurred,
    its tiles loading, in the looks numbered in ``loading`` (from 1). No compass shows."""

    def __init__(self, blank_grabs=0, loading=()):
        self.blank_grabs, self.grabs, self.loading = blank_grabs, 0, set(loading)

    def virtual_desktop(self):
        return DESKTOP

    def grab(self, region):
        if region.left != LAYOUT.view.left:  # where the compass would be
            return Image.new("RGB", (region.width, region.height), (90, 110, 70))
        self.grabs += 1
        if self.blank_grabs:
            self.blank_grabs -= 1
            return Image.new("RGB", (region.width, region.height))
        if self.grabs in self.loading:  # light and dark, but no detail yet
            ramp = np.linspace(0, 255, region.width).astype(np.uint8)
            return Image.fromarray(np.tile(ramp, (region.height, 1))).convert("RGB")
        rng = np.random.default_rng(self.grabs)  # grey, so no blue sky to look up at
        noise = rng.integers(0, 256, (region.height, region.width, 1)).repeat(3, axis=2)
        return Image.fromarray(noise.astype(np.uint8))


class CornerSlam(Exception):
    """Stands in for what pyautogui raises when the mouse is slammed into a screen corner."""


class FakeControls:
    failsafe = CornerSlam

    def __init__(self, slam_on_click=None):
        self.clicks, self.drags, self.keys = [], [], []
        self.slam_on_click = slam_on_click

    def sleep(self, seconds):
        pass

    def press(self, key):
        self.keys.append(key)

    def move(self, p):
        pass

    def click(self, p):
        if self.slam_on_click is not None and len(self.clicks) == self.slam_on_click:
            raise CornerSlam("mouse in a corner")
        self.clicks.append(p)

    def drag(self, start, dx, dy=0):
        self.drags.append((start, dx, dy))


class TurningStreetView(FakeScreen, FakeControls):
    """Street View with Google's compass: pressing it faces north and levels the camera, its
    arrow turns 90 degrees and keeps the tilt, and dragging straight down tilts the camera up.
    The sun shows when facing ``sun_heading``: low in a level view, or with ``sun_high`` only
    once tilted up. With ``clear``, blue sky shows all round. The presses numbered in
    ``ignored`` (counting from 1) do nothing, as when Street View is busy."""

    def __init__(self, sun_heading=None, ignored=(), sun_high=False, clear=False):
        FakeScreen.__init__(self)
        FakeControls.__init__(self)
        self.heading, self.sun_heading, self.sun_high = 260.0, sun_heading, sun_high
        self.clear, self.looking_up = clear, False
        self.ignored, self.presses = set(ignored), 0

    def grab(self, region):
        if region.left != LAYOUT.view.left:
            self.compass_at = Point(region.left + 90, region.top + 355)
            rgb = np.full((region.height, region.width, 3), (90, 110, 70), np.uint8)
            rgb[300:420, :160] = draw_compass(self.heading)
            return Image.fromarray(rgb)
        sun = self.heading == self.sun_heading and self.looking_up == self.sun_high
        if not (sun or self.clear):
            return super().grab(region)
        if not sun:
            picture = sky()
        elif self.sun_high:  # glaring in the middle of the view
            picture = sky(sun_at=(700, 250), sun_radius=40, glow=90, ground=1.0)
        else:
            picture = sky(sun_at=(700, 150))
        rgb = np.full((region.height, region.width, 3), (70, 110, 60), np.uint8)
        rows = min(region.height, LAYOUT.view.height)
        rgb[:rows] = cv2.resize(picture, (region.width, LAYOUT.view.height))[:rows]
        return Image.fromarray(rgb)

    def drag(self, start, dx, dy=0):
        super().drag(start, dx, dy)
        if dx == 0 and dy:
            self.looking_up = dy > 0

    def click(self, p):
        super().click(p)
        self.presses += 1
        if self.presses in self.ignored:
            return
        if abs(p.x - self.compass_at.x) <= 3 and abs(p.y - self.compass_at.y) <= 3:
            self.heading, self.looking_up = 0.0, False
        elif self.compass_at.x + 15 < p.x < self.compass_at.x + 40:
            self.heading = (self.heading // 90 + 1) * 90 % 360  # on to the next quarter


class FakePredictor:
    """Guesses Lagos, expecting each of ``expected`` points in turn, then the last again."""

    def __init__(self, expected=(2500.0,)):
        self.view_counts, self.evidence, self.expected = [], [], list(expected)

    def predict(self, images, evidence=None):
        self.view_counts.append(len(images))
        self.evidence.append(evidence)
        expected = self.expected.pop(0) if len(self.expected) > 1 else self.expected[0]
        return Guess(*LAGOS, expected_score=expected, top_cells=[], countries=[("NG", 0.8)])


class FakeGuessMap:
    """Stands in for the minimap, which has its own tests."""

    def __init__(self, screen, controls, region, hover, **settings):
        self.controls, self.region, self.guesses, self.scale = controls, region, [], 1.5

    def place(self, lat, lon):
        self.guesses.append((lat, lon))
        click = self.region.to_screen(400, 300)
        self.controls.click(click)
        image = Image.new("RGB", (self.region.width, self.region.height))
        projection = MapProjection(1600, -70, -260)
        return Placement(lat + 0.01, lon, click, image, projection, (400.0, 300.0), True)


class FakeSignReader:
    """Reads one Portuguese street sign among the scenes it is shown, and a road's name from
    above among ground views, which it's shown with a lower bar for finding text."""

    def __init__(self):
        self.scenes, self.grounds = [], []

    def read(self, images, min_box_score=None):
        if min_box_score is not None:
            self.grounds.extend(image.size for image in images)
            return [TextLine("Avenida Paulista", "latin", 0.9)] if images else []
        self.scenes.extend(image.size for image in images)
        return [TextLine("Rua Augusta 12", "latin", 0.93)]


class FakeReader:
    """Stands in for the result-screen reader, which has its own tests."""

    def __init__(self, screen, controls, region):
        self.region, self.placed, self.scale = region, [], None

    def read(self, lat, lon):
        self.placed.append((lat, lon))
        return Reading(Answer(6.52, 3.38, zoom_outs=2), screenshot=Image.new("RGB", (40, 20)))


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(game_module, "GuessMap", FakeGuessMap)
    monkeypatch.setattr(game_module, "SignReader", FakeSignReader)


def test_round_looks_around_places_pin_reads_answer_and_advances(tmp_path, monkeypatch):
    monkeypatch.setattr(game_module, "ResultReader", FakeReader)
    controls, predictor = FakeControls(), FakePredictor()
    settings = BotSettings(rounds=2, views=4, debug_dir=tmp_path)
    bot = OpenGuessrBot(LAYOUT, predictor, settings, FakeScreen(), controls)

    bot.play()

    assert predictor.view_counts == [4, 4]
    assert len(controls.drags) == 6 and all(dx < 0 for _, dx, _ in controls.drags)  # no compass
    assert bot.guess_map.guesses == [LAGOS, LAGOS]
    assert controls.clicks[1:3] == [LAYOUT.guess_button, LAYOUT.continue_button]

    assert bot.reader.region == result_region(LAYOUT)
    assert bot.reader.placed[0] == pytest.approx((6.46, 3.39))  # where the pin really went
    assert bot.reader.scale == 1.5  # the minimap already showed how big pins are drawn
    folder = next(tmp_path.glob("*/round_01"))
    saved = json.loads((folder / "round.json").read_text(encoding="utf-8"))
    assert saved["guess"]["lat"] == 6.45 and saved["headings"] == [None] * 4
    assert (saved["placed"]["lat"], saved["placed"]["lon"]) == pytest.approx((6.46, 3.39))
    assert saved["answer"] == {"lat": 6.52, "lon": 3.38, "zoom_outs": 2}
    assert saved["pin_seen"] is True
    assert (folder / "result.png").exists() and (folder / "map.png").exists()
    assert Image.open(folder / "view_0.jpg").size == (1600, 700)


def test_turns_north_east_south_and_west_with_the_compass():
    street_view = TurningStreetView()
    settings = BotSettings(rounds=1, views=4, record_answers=False, debug_dir=None)

    bot = OpenGuessrBot(LAYOUT, FakePredictor(), settings, street_view, street_view)
    look = bot.look_around()

    assert [round(h) % 360 for h in look.headings] == [0, 90, 180, 270]
    assert len(look.views) == 4
    assert look.scenes[0].size == (1600, 860)  # the view plus the road below it


def test_looks_down_at_the_road_all_the_way_round_then_levels_the_camera(tmp_path):
    street_view = TurningStreetView()
    settings = BotSettings(
        rounds=1, record_answers=False, read_text=False, look_down=True, debug_dir=tmp_path
    )

    bot = OpenGuessrBot(LAYOUT, FakePredictor(), settings, street_view, street_view)
    bot.play()

    tilts = [(start, dy) for start, dx, dy in street_view.drags if dx == 0]
    assert tilts and all(dy < 0 for _, dy in tilts)  # dragging up the screen looks down
    assert all(LAYOUT.view.contains(start) for start, _ in tilts)
    folder = next(tmp_path.glob("*/round_01"))
    saved = json.loads((folder / "round.json").read_text(encoding="utf-8"))
    assert [round(h) % 360 for h in saved["down_headings"]] == [0, 90, 180, 270]
    assert (folder / "down_3.jpg").exists()
    assert street_view.heading == 0  # the compass pressed at the end, which levels the camera


@pytest.mark.parametrize("ignored", [(1,), (3,), (1, 2)])
def test_presses_the_compass_again_when_street_view_ignores_it(ignored):
    street_view = TurningStreetView(ignored=ignored)
    settings = BotSettings(rounds=1, record_answers=False, look_down=False, debug_dir=None)

    bot = OpenGuessrBot(LAYOUT, FakePredictor(), settings, street_view, street_view)
    look = bot.look_around()

    assert [round(h) % 360 for h in look.headings] == [0, 90, 180, 270]
    assert street_view.presses == 4 + len(ignored)


def test_walks_on_and_looks_again_when_unsure(tmp_path):
    controls, predictor = FakeControls(), FakePredictor(expected=(900.0, 3100.0))
    settings = BotSettings(rounds=1, record_answers=False, debug_dir=tmp_path)

    bot = OpenGuessrBot(LAYOUT, predictor, settings, FakeScreen(), controls)
    bot.play()

    assert controls.keys == ["up"] * settings.walk_steps
    sky = controls.clicks[0]  # gives Street View the keyboard
    assert LAYOUT.view.contains(sky) and sky.y < LAYOUT.view.top + LAYOUT.view.height // 10
    assert predictor.view_counts == [4, 8]  # the guess uses both spots
    assert len(bot.signs.scenes) == 4 + 8
    saved = json.loads(next(tmp_path.glob("*/round_01/round.json")).read_text(encoding="utf-8"))
    assert saved["walked"] is True and len(saved["headings"]) == 8
    assert saved["walks"] == 1 and saved["expected_each_look"] == [900.0, 3100.0]


@pytest.mark.parametrize(
    "expected,looks",
    [
        ((600.0, 800.0, 2000.0), 3),  # still very unsure after walking: walks on once more
        ((600.0, 800.0, 700.0, 650.0), 3),  # but not a third time
        ((1400.0, 1200.0), 2),  # unsure, but not very unsure after walking
    ],
)
def test_walks_on_once_more_when_still_very_unsure(tmp_path, expected, looks):
    controls, predictor = FakeControls(), FakePredictor(expected=expected)
    settings = BotSettings(rounds=1, record_answers=False, debug_dir=tmp_path)

    OpenGuessrBot(LAYOUT, predictor, settings, FakeScreen(), controls).play()

    assert predictor.view_counts == [4 * n for n in range(1, looks + 1)]
    assert controls.keys == ["up"] * settings.walk_steps * (looks - 1)
    saved = json.loads(next(tmp_path.glob("*/round_01/round.json")).read_text(encoding="utf-8"))
    assert saved["walks"] == looks - 1 and saved["expected_each_look"] == list(expected[:looks])


def test_sure_rounds_and_dry_runs_dont_walk():
    for settings, expected in (
        (BotSettings(rounds=1, record_answers=False, debug_dir=None), 1800.0),
        (BotSettings(rounds=1, dry_run=True, debug_dir=None), 900.0),
        (BotSettings(rounds=1, record_answers=False, walk_below=0.0, debug_dir=None), 600.0),
    ):
        controls = FakeControls()
        OpenGuessrBot(LAYOUT, FakePredictor((expected,)), settings, FakeScreen(), controls).play()
        assert not controls.keys


def test_a_sun_to_the_south_hints_at_the_northern_hemisphere():
    street_view = TurningStreetView(sun_heading=180.0)
    settings = BotSettings(rounds=1, record_answers=False, read_text=False, debug_dir=None)
    bot = OpenGuessrBot(LAYOUT, FakePredictor(), settings, street_view, street_view)

    evidence, _ = bot.notice(bot.look_around())

    assert any(note.startswith("sun to the south") for note in evidence.notes)
    assert evidence.latitude(45) > 3 * evidence.latitude(-35)


def test_looks_up_at_a_clear_sky_and_finds_the_sun(tmp_path):
    street_view = TurningStreetView(sun_heading=90.0, sun_high=True, clear=True)
    settings = BotSettings(rounds=1, record_answers=False, read_text=False, debug_dir=tmp_path)
    bot = OpenGuessrBot(LAYOUT, FakePredictor(), settings, street_view, street_view)

    look = bot.look_around()

    ups = [(start, dy) for start, dx, dy in street_view.drags if dx == 0]
    assert ups and all(dy > 0 and LAYOUT.view.contains(start) for start, dy in ups)
    assert [round(h) for h in look.up_headings] == [0, 90]  # stopped once it saw the sun
    assert look.up_pitches[0] == pytest.approx(45, abs=5)
    assert not street_view.looking_up and street_view.heading == 0  # levelled again
    evidence, _ = bot.notice(look)
    note = next(note for note in evidence.notes if note.startswith("sun"))
    assert note.startswith("sun to the east") and "45 deg up" in note


def test_a_high_sun_to_the_north_hints_at_the_southern_hemisphere(tmp_path):
    street_view = TurningStreetView(sun_heading=0.0, sun_high=True, clear=True)
    predictor = FakePredictor()
    settings = BotSettings(rounds=1, record_answers=False, read_text=False, debug_dir=tmp_path)

    OpenGuessrBot(LAYOUT, predictor, settings, street_view, street_view).play()

    evidence = predictor.evidence[0]
    assert evidence.latitude(-30) > 3 * evidence.latitude(45)
    folder = next(tmp_path.glob("*/round_01"))
    saved = json.loads((folder / "round.json").read_text(encoding="utf-8"))
    assert saved["up_headings"] == [0.0] and (folder / "up_0.jpg").exists()
    assert saved["camera"]["measured"] is False


@pytest.mark.parametrize("street_view, look_up", [(TurningStreetView, True), (None, False)])
def test_doesnt_look_up_under_grey_skies_or_when_told_not_to(street_view, look_up):
    street_view = street_view() if street_view else TurningStreetView(clear=True)
    settings = BotSettings(rounds=1, record_answers=False, look_up=look_up, debug_dir=None)

    look = OpenGuessrBot(LAYOUT, FakePredictor(), settings, street_view, street_view).look_around()

    assert not look.up_views and not any(dx == 0 for _, dx, _ in street_view.drags)


def test_measures_the_camera_once_by_dragging_sideways(monkeypatch):
    measured = Camera(700.0, 380.0, 500.0, measured=True)
    calls = []

    def fake_measure(before, after, view, guess):
        calls.append(guess)
        return None if len(calls) == 1 else (measured, 30.0)

    monkeypatch.setattr(game_module, "measure", fake_measure)
    street_view = TurningStreetView()
    settings = BotSettings(rounds=4, record_answers=False, read_text=False, debug_dir=None)

    bot = OpenGuessrBot(LAYOUT, FakePredictor(), settings, street_view, street_view)
    bot.play()

    assert len(calls) == 2 and bot.camera == measured  # tried again after failing once
    sideways = [(start, dx) for start, dx, dy in street_view.drags if dy == 0]
    assert len(sideways) == 2 and all(dx < 0 and LAYOUT.view.contains(s) for s, dx in sideways)


def test_reads_road_names_from_above_once_the_camera_is_measured(monkeypatch):
    measured = Camera(800.0, 400.0, 600.0, measured=True)
    monkeypatch.setattr(game_module, "measure", lambda *args: (measured, 30.0))
    street_view = TurningStreetView()
    settings = BotSettings(rounds=1, record_answers=False, debug_dir=None)
    bot = OpenGuessrBot(LAYOUT, FakePredictor(), settings, street_view, street_view)

    evidence, lines = bot.notice(bot.look_around())

    assert len(bot.signs.grounds) == 4  # the road round the car in each view, from above
    assert "Avenida Paulista" in [line.text for line in lines]
    assert any(note.startswith("Portuguese") for note in evidence.notes)


def test_signs_become_clues_for_the_model(tmp_path):
    predictor = FakePredictor()
    settings = BotSettings(rounds=1, record_answers=False, debug_dir=tmp_path)

    bot = OpenGuessrBot(LAYOUT, predictor, settings, FakeScreen(), FakeControls())
    bot.play()

    evidence = predictor.evidence[0]
    assert any(note.startswith("Portuguese") for note in evidence.notes)
    assert len(bot.signs.scenes) == 4
    saved = json.loads(next(tmp_path.glob("*/round_01/round.json")).read_text(encoding="utf-8"))
    assert saved["clues"] == evidence.notes and saved["text"][0]["text"] == "Rua Augusta 12"


def test_unreadable_answer_is_saved_as_missing(tmp_path, monkeypatch):
    class BlindReader(FakeReader):
        def read(self, lat, lon):
            return Reading(None, "couldn't see the flag on the result map")

    monkeypatch.setattr(game_module, "ResultReader", BlindReader)
    controls = FakeControls()
    settings = BotSettings(rounds=1, debug_dir=tmp_path)

    OpenGuessrBot(LAYOUT, FakePredictor(), settings, FakeScreen(), controls).play()

    saved = json.loads(next(tmp_path.glob("*/round_01/round.json")).read_text(encoding="utf-8"))
    assert saved["answer"] is None and "flag" in saved["answer_problem"]
    assert controls.clicks[-1] == LAYOUT.continue_button


class CoveredContinue(FakeScreen):
    """An advert covers the Continue button for the first ``covered_grabs`` looks at it."""

    def __init__(self, covered_grabs):
        super().__init__()
        self.covered_grabs = covered_grabs

    def grab(self, region):
        if region != button_region(LAYOUT.continue_button):
            return super().grab(region)
        if self.covered_grabs:
            self.covered_grabs -= 1
            return Image.new("RGB", (region.width, region.height), "white")
        return draw_button()


def test_waits_for_an_advert_to_uncover_continue():
    controls, screen = FakeControls(), CoveredContinue(covered_grabs=3)
    settings = BotSettings(rounds=1, record_answers=False, read_text=False, debug_dir=None)

    bot = OpenGuessrBot(LAYOUT, FakePredictor(), settings, screen, controls, draw_button())
    bot.play()

    assert controls.clicks[-1] == LAYOUT.continue_button and screen.covered_grabs == 0


def test_stops_rather_than_click_an_advert_covering_continue(tmp_path, capsys):
    controls, screen = FakeControls(), CoveredContinue(covered_grabs=10**6)
    settings = BotSettings(rounds=3, record_answers=False, read_text=False, debug_dir=tmp_path)

    OpenGuessrBot(LAYOUT, FakePredictor(), settings, screen, controls, draw_button()).play()

    assert LAYOUT.continue_button not in controls.clicks
    assert "Continue button stayed covered" in capsys.readouterr().out
    assert [p.name for p in tmp_path.glob("*/round_*")] == ["round_01"]
    assert next(tmp_path.glob("*/round_01/continue_blocked.png")).exists()


def test_a_corner_slam_ends_the_run_like_the_stop_key(tmp_path, capsys):
    controls = FakeControls(slam_on_click=3)  # part-way into the second round
    settings = BotSettings(rounds=3, record_answers=False, read_text=False, debug_dir=tmp_path)

    OpenGuessrBot(LAYOUT, FakePredictor(), settings, FakeScreen(), controls).play()

    assert "screen corner" in capsys.readouterr().out  # and no traceback: learning still follows
    assert [p.name for p in tmp_path.glob("*/round_*")] == ["round_01"]


def test_waits_for_street_view_to_stop_being_black():
    screen = FakeScreen(blank_grabs=3)
    settings = BotSettings(rounds=1, views=4, dry_run=True, debug_dir=None)

    OpenGuessrBot(LAYOUT, FakePredictor(), settings, screen, FakeControls()).play()

    assert screen.grabs == 3 + 1 + 4  # three black frames, the first good one, four views


def test_waits_for_each_view_to_finish_loading():
    screen, predictor = FakeScreen(loading={2, 3}), FakePredictor()
    settings = BotSettings(rounds=1, views=4, dry_run=True, debug_dir=None)

    OpenGuessrBot(LAYOUT, predictor, settings, screen, FakeControls()).play()

    assert screen.grabs == 1 + 2 + 4  # Street View drawn, the first view twice still blurred
    assert predictor.view_counts == [4]


def test_a_view_that_never_loads_isnt_shown_to_the_model(tmp_path):
    screen, predictor = FakeScreen(loading=range(3, 10)), FakePredictor()
    settings = BotSettings(rounds=1, views=4, record_answers=False, debug_dir=tmp_path)

    OpenGuessrBot(LAYOUT, predictor, settings, screen, FakeControls()).play()

    assert predictor.view_counts == [3]  # the second view stayed blurred for three seconds
    saved = json.loads(next(tmp_path.glob("*/round_01/round.json")).read_text(encoding="utf-8"))
    assert saved["undrawn_views"] == [1] and len(saved["headings"]) == 4


def test_dry_run_plays_one_round_without_continuing():
    controls = FakeControls()
    settings = BotSettings(rounds=5, dry_run=True, debug_dir=None)
    OpenGuessrBot(LAYOUT, FakePredictor(), settings, FakeScreen(), controls).play()
    assert controls.clicks[-1] == LAYOUT.guess_button
    assert LAYOUT.continue_button not in controls.clicks


PARTY = Layout(
    view=LAYOUT.view,
    map_hover=LAYOUT.map_hover,
    map_region=LAYOUT.map_region,
    guess_button=LAYOUT.guess_button,
    continue_button=None,  # the host presses it
    timer=Point(1800, 960),
    covered=(Region(0, 600, 300, 200),),  # the chat box
)


class MultiplayerRoom(TurningStreetView):
    """A multiplayer room. The host starts a round ``lobby`` seconds in, and the next one
    ``results`` seconds after each ends, ``rounds`` in all. Each lasts ``length`` seconds, or
    until everyone has guessed: the others do ``others_guess`` seconds in, which in a ``duel``
    cuts the time left to 15 seconds. The compass shows only during rounds, and every click,
    look and second waited passes time."""

    def __init__(self, rounds=2, length=60.0, others_guess=20.0, duel=False, results=8.0):
        super().__init__()
        self.t, self.length, self.others_guess, self.duel = 0.0, length, others_guess, duel
        self.results, self.rounds_left = results, rounds
        self.phase, self.phase_ends = "lobby", 2.0
        self.locked = []  # seconds into each round we locked in, or None
        self.idle_clicks = []  # clicks while no round was on

    def now(self):
        return self.t

    def _pass(self, seconds):
        self.t += seconds
        if self.t > 3600:
            raise TimeoutError("the bot has waited an hour for a round that never came")
        while self._step():
            pass

    def _step(self):
        if self.phase in ("lobby", "results") and self.t >= self.phase_ends:
            if not self.rounds_left:
                self.phase = "standings"
                return False
            self.rounds_left -= 1
            self.phase, self.start = "round", self.phase_ends
            self.deadline, self.locked_at, self.others_in = self.start + self.length, None, False
            return True
        if self.phase != "round":
            return False
        if not self.others_in and self.t >= self.start + self.others_guess:
            self.others_in = True
            if self.duel:
                self.deadline = min(self.deadline, self.t + 15.0)
            return True
        if self.t >= self.deadline or (self.others_in and self.locked_at is not None):
            self.locked.append(self.locked_at)
            self.phase, self.phase_ends = "results", self.t + self.results
            return True
        return False

    def timer_text(self):
        if self.phase != "round":
            return ""
        left = max(int(self.deadline - self.t), 0)
        return f"{left // 60:02d}:{left % 60:02d}"

    def sleep(self, seconds):
        self._pass(seconds)

    def move(self, p):
        self._pass(0.05)

    def press(self, key):
        super().press(key)
        self._pass(0.1)

    def drag(self, start, dx, dy=0):
        super().drag(start, dx, dy)
        self._pass(0.5)

    def click(self, p):
        if self.phase != "round":
            self.idle_clicks.append(p)
        super().click(p)
        if self.phase == "round" and p == PARTY.guess_button and self.locked_at is None:
            self.locked_at = self.t - self.start
        self._pass(0.2)

    def grab(self, region):
        self._pass(0.02)
        if region == timer_region(PARTY.timer):  # what it says is in timer_text
            return Image.new("RGB", (region.width, region.height), (50, 52, 60))
        if region.left != PARTY.view.left and self.phase != "round":  # no compass
            return Image.new("RGB", (region.width, region.height), (90, 110, 70))
        return super().grab(region)


class PartyReader(FakeReader):
    def read(self, lat, lon, crowded=False, abort=None):
        self.crowded, self.aborted = crowded, abort()
        return super().read(lat, lon)


@pytest.fixture
def party(monkeypatch):
    """Plays in a multiplayer room, reading its timer off what the room says is left."""

    def bot_in(room, predictor=None, **settings):
        class TimerReadingSigns(FakeSignReader):
            def read_line(self, image):
                return room.timer_text()

        monkeypatch.setattr(game_module, "SignReader", TimerReadingSigns)
        monkeypatch.setattr(game_module, "ResultReader", PartyReader)
        rounds = room.rounds_left  # all the host will start
        settings = {"party": True, "rounds": rounds, "look_up": False, "debug_dir": None} | settings
        return OpenGuessrBot(PARTY, predictor or FakePredictor(), BotSettings(**settings), room, room)

    return bot_in


def test_plays_each_round_the_host_starts_and_waits_for_the_host_to_go_on(party, capsys):
    room = MultiplayerRoom(rounds=2, length=60.0, others_guess=5.0)
    bot = party(room)

    bot.play()

    assert len(room.locked) == 2 and all(at is not None and at < 60 for at in room.locked)
    assert room.idle_clicks == []  # nothing clicked on result screens: no Continue
    assert bot.guess_map.guesses == [LAGOS, LAGOS]
    assert bot.reader.crowded and not bot.reader.aborted  # other players' pins show too
    assert "60 s on the clock" not in capsys.readouterr().out  # read after the round loaded


def test_hurries_when_little_time_is_left(party):
    room = MultiplayerRoom(rounds=1, length=12.0, others_guess=60.0)
    predictor = FakePredictor(expected=(1000.0,))  # unsure: would walk on, given time
    bot = party(room, predictor)

    bot.play()

    assert room.locked[0] is not None  # in time
    assert predictor.view_counts == [1] and bot.signs.scenes == [] and room.keys == []


def test_notices_the_timer_cut_short_when_another_player_guesses(party, capsys):
    room = MultiplayerRoom(rounds=1, length=120.0, others_guess=9.0, duel=True)
    predictor = FakePredictor(expected=(1000.0,))  # unsure: would walk on, given time
    bot = party(room, predictor)

    bot.play()

    out = capsys.readouterr().out  # cut while it looked round, two minutes down to 15 seconds
    assert "116 s on the clock" in out and "timer was cut to 1" in out
    assert room.locked[0] is not None and room.locked[0] < 9 + 15
    assert bot.signs.scenes == [] and room.keys == []  # no time to read signs or walk on


def test_gives_up_a_round_whose_time_runs_out_and_plays_the_next(party, capsys):
    room = MultiplayerRoom(rounds=2, length=5.0, others_guess=60.0)
    bot = party(room)
    bot.settings.round_load_wait = 4.5  # Street View took most of the round to load

    bot.play()

    assert room.locked[0] is None and "ended before the bot could guess" in capsys.readouterr().out
    assert PARTY.guess_button not in room.idle_clicks
    assert len(room.locked) == 2


def test_the_chat_box_is_blanked_out_of_what_the_bot_sees():
    street_view = TurningStreetView()
    settings = BotSettings(rounds=1, views=1, record_answers=False, debug_dir=None)
    bot = OpenGuessrBot(PARTY, FakePredictor(), settings, street_view, street_view)

    look = bot.look_around()

    chat = np.asarray(look.views[0])[500:700, 0:300]  # where it covers the view
    assert chat.std() == 0 and np.asarray(look.views[0])[500:700, 300:600].std() > 10
