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
    """Street View that stays black for the first ``blank_grabs`` looks. No compass shows."""

    def __init__(self, blank_grabs=0):
        self.blank_grabs, self.grabs = blank_grabs, 0

    def virtual_desktop(self):
        return DESKTOP

    def grab(self, region):
        if region.left != LAYOUT.view.left:  # where the compass would be
            return Image.new("RGB", (region.width, region.height), (90, 110, 70))
        self.grabs += 1
        if self.blank_grabs:
            self.blank_grabs -= 1
            return Image.new("RGB", (region.width, region.height))
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


def test_sure_rounds_and_dry_runs_dont_walk():
    for settings, expected in (
        (BotSettings(rounds=1, record_answers=False, debug_dir=None), 1600.0),
        (BotSettings(rounds=1, dry_run=True, debug_dir=None), 900.0),
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


def test_dry_run_plays_one_round_without_continuing():
    controls = FakeControls()
    settings = BotSettings(rounds=5, dry_run=True, debug_dir=None)
    OpenGuessrBot(LAYOUT, FakePredictor(), settings, FakeScreen(), controls).play()
    assert controls.clicks[-1] == LAYOUT.guess_button
    assert LAYOUT.continue_button not in controls.clicks
