import json

import numpy as np
import pytest
from PIL import Image

from geoguessr_ai import game as game_module
from geoguessr_ai.config import Layout, Point, Region
from geoguessr_ai.game import BotSettings, OpenGuessrBot
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
LAGOS = (6.45, 3.39)


class FakeScreen:
    """Street View that stays black for the first ``blank_grabs`` grabs."""

    def __init__(self, blank_grabs=0):
        self.blank_grabs, self.grabs = blank_grabs, 0

    def grab(self, region):
        assert region == LAYOUT.view
        self.grabs += 1
        if self.blank_grabs:
            self.blank_grabs -= 1
            return Image.new("RGB", (region.width, region.height))
        noise = np.random.default_rng(self.grabs).integers(0, 256, (region.height, region.width, 3))
        return Image.fromarray(noise.astype(np.uint8))


class FakeControls:
    def __init__(self):
        self.clicks, self.drags = [], []

    def sleep(self, seconds):
        pass

    def move(self, p):
        pass

    def click(self, p):
        self.clicks.append(p)

    def drag(self, start, dx, dy=0):
        self.drags.append((start, dx))


class FakePredictor:
    def __init__(self):
        self.view_counts = []

    def predict(self, images):
        self.view_counts.append(len(images))
        return Guess(*LAGOS, expected_score=2500.0, top_cells=[])


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


class FakeReader:
    """Stands in for the result-screen reader, which has its own tests."""

    def __init__(self, screen, controls, region):
        self.region, self.placed, self.scale = region, [], None

    def read(self, lat, lon):
        self.placed.append((lat, lon))
        return Reading(Answer(6.52, 3.38, zoom_outs=2), screenshot=Image.new("RGB", (40, 20)))


@pytest.fixture(autouse=True)
def fake_guess_map(monkeypatch):
    monkeypatch.setattr(game_module, "GuessMap", FakeGuessMap)


def test_round_looks_around_places_pin_reads_answer_and_advances(tmp_path, monkeypatch):
    monkeypatch.setattr(game_module, "ResultReader", FakeReader)
    controls, predictor = FakeControls(), FakePredictor()
    settings = BotSettings(rounds=2, views=4, debug_dir=tmp_path)
    bot = OpenGuessrBot(LAYOUT, predictor, settings, FakeScreen(), controls)

    bot.play()

    assert predictor.view_counts == [4, 4]
    assert len(controls.drags) == 6 and all(dx < 0 for _, dx in controls.drags)
    assert bot.guess_map.guesses == [LAGOS, LAGOS]
    assert controls.clicks[1:3] == [LAYOUT.guess_button, LAYOUT.continue_button]

    assert bot.reader.region == result_region(LAYOUT)
    assert bot.reader.placed[0] == pytest.approx((6.46, 3.39))  # where the pin really went
    assert bot.reader.scale == 1.5  # the minimap already showed how big pins are drawn
    folder = next(tmp_path.glob("*/round_01"))
    saved = json.loads((folder / "round.json").read_text())
    assert saved["guess"]["lat"] == 6.45
    assert (saved["placed"]["lat"], saved["placed"]["lon"]) == pytest.approx((6.46, 3.39))
    assert saved["answer"] == {"lat": 6.52, "lon": 3.38, "zoom_outs": 2}
    assert saved["pin_seen"] is True
    assert (folder / "result.png").exists() and (folder / "map.png").exists()


def test_unreadable_answer_is_saved_as_missing(tmp_path, monkeypatch):
    class BlindReader(FakeReader):
        def read(self, lat, lon):
            return Reading(None, "couldn't see the flag on the result map")

    monkeypatch.setattr(game_module, "ResultReader", BlindReader)
    controls = FakeControls()
    settings = BotSettings(rounds=1, debug_dir=tmp_path)

    OpenGuessrBot(LAYOUT, FakePredictor(), settings, FakeScreen(), controls).play()

    saved = json.loads(next(tmp_path.glob("*/round_01/round.json")).read_text())
    assert saved["answer"] is None and "flag" in saved["answer_problem"]
    assert controls.clicks[-1] == LAYOUT.continue_button


def test_waits_for_street_view_to_stop_being_black():
    screen = FakeScreen(blank_grabs=3)
    settings = BotSettings(rounds=1, views=4, dry_run=True, debug_dir=None)

    OpenGuessrBot(LAYOUT, FakePredictor(), settings, screen, FakeControls()).play()

    assert screen.grabs == 3 + 4


def test_dry_run_plays_one_round_without_continuing(tmp_path):
    controls = FakeControls()
    settings = BotSettings(rounds=5, dry_run=True, debug_dir=None)
    OpenGuessrBot(LAYOUT, FakePredictor(), settings, FakeScreen(), controls).play()
    assert controls.clicks[-1] == LAYOUT.guess_button
    assert LAYOUT.continue_button not in controls.clicks
