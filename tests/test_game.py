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
        noise = np.random.default_rng(self.grabs).integers(0, 256, (region.height, region.width, 3))
        return Image.fromarray(noise.astype(np.uint8))


class FakeControls:
    def __init__(self):
        self.clicks, self.drags, self.keys = [], [], []

    def sleep(self, seconds):
        pass

    def press(self, key):
        self.keys.append(key)

    def move(self, p):
        pass

    def click(self, p):
        self.clicks.append(p)

    def drag(self, start, dx, dy=0):
        self.drags.append((start, dx, dy))


class TurningStreetView(FakeScreen, FakeControls):
    """Street View with Google's compass: pressing it faces north, its arrow turns 90 degrees.
    The sun shows low in the sky when facing ``sun_heading``. The presses numbered in
    ``ignored`` (counting from 1) do nothing, as when Street View is busy."""

    def __init__(self, sun_heading=None, ignored=()):
        FakeScreen.__init__(self)
        FakeControls.__init__(self)
        self.heading, self.sun_heading = 260.0, sun_heading
        self.ignored, self.presses = set(ignored), 0

    def grab(self, region):
        if region.left != LAYOUT.view.left:
            self.compass_at = Point(region.left + 90, region.top + 355)
            rgb = np.full((region.height, region.width, 3), (90, 110, 70), np.uint8)
            rgb[300:420, :160] = draw_compass(self.heading)
            return Image.fromarray(rgb)
        if self.heading != self.sun_heading:
            return super().grab(region)
        rgb = np.full((region.height, region.width, 3), (70, 110, 60), np.uint8)
        rows = min(region.height, LAYOUT.view.height)
        rgb[:rows] = cv2.resize(sky(sun_at=(700, 150)), (region.width, LAYOUT.view.height))[:rows]
        return Image.fromarray(rgb)

    def click(self, p):
        super().click(p)
        self.presses += 1
        if self.presses in self.ignored:
            return
        if abs(p.x - self.compass_at.x) <= 3 and abs(p.y - self.compass_at.y) <= 3:
            self.heading = 0.0
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
    """Reads one Portuguese street sign among everything it is shown."""

    def __init__(self):
        self.scenes = []

    def read(self, images):
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

    assert len(street_view.drags) == 2  # tilting down, straight up the screen
    assert all(dx == 0 and dy < 0 for _, dx, dy in street_view.drags)
    folder = next(tmp_path.glob("*/round_01"))
    saved = json.loads((folder / "round.json").read_text(encoding="utf-8"))
    assert [round(h) % 360 for h in saved["down_headings"]] == [270, 0, 90, 180]
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
