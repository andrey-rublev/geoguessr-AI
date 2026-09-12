import json

import numpy as np
from PIL import Image
from test_mapcal import render_map

from geoguessr_ai.config import Layout, Point, Region
from geoguessr_ai.game import BotSettings, OpenGuessrBot
from geoguessr_ai.mapcal import MapProjection
from geoguessr_ai.model.predictor import Guess

TRUE_MAP = MapProjection(1600, -70, -260)
LAYOUT = Layout(
    view=Region(0, 100, 1600, 700),
    map_hover=Point(1800, 950),
    map_region=Region(1000, 300, 845, 633),
    guess_button=Point(1800, 1040),
    continue_button=Point(960, 1040),
)


class FakeScreen:
    def __init__(self):
        self.map_rgb = render_map(845, 633, TRUE_MAP)

    def grab(self, region):
        if region == LAYOUT.map_region:
            return Image.fromarray(self.map_rgb)
        return Image.new("RGB", (region.width, region.height), (120, 160, 90))


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

    def scroll(self, p, clicks):
        raise AssertionError("the synthetic map should be readable without zooming")


class FakePredictor:
    def __init__(self):
        self.view_counts = []

    def predict(self, images):
        self.view_counts.append(len(images))
        return Guess(
            lat=6.45, lon=3.39, expected_score=2500.0, top_cells=[]
        )  # Lagos, visible on TRUE_MAP


def test_round_looks_around_places_pin_and_advances(tmp_path):
    controls, predictor = FakeControls(), FakePredictor()
    settings = BotSettings(rounds=2, views=4, debug_dir=tmp_path)
    bot = OpenGuessrBot(LAYOUT, predictor, settings, FakeScreen(), controls)

    bot.play()

    assert predictor.view_counts == [4, 4]
    assert len(controls.drags) == 6 and all(dx < 0 for _, dx in controls.drags)
    pin, guess, cont = controls.clicks[:3]
    tx, ty = TRUE_MAP.to_pixel(6.45, 3.39)
    assert abs(pin.x - (LAYOUT.map_region.left + tx)) <= 2
    assert abs(pin.y - (LAYOUT.map_region.top + ty)) <= 2
    assert (guess, cont) == (LAYOUT.guess_button, LAYOUT.continue_button)

    saved = next(tmp_path.glob("*/round_01/round.json"))
    assert json.loads(saved.read_text())["guess"]["lat"] == 6.45
    assert np.asarray(Image.open(saved.parent / "map.png")).shape == (633, 845, 3)


def test_dry_run_plays_one_round_without_continuing(tmp_path):
    controls = FakeControls()
    settings = BotSettings(rounds=5, dry_run=True, debug_dir=None)
    OpenGuessrBot(LAYOUT, FakePredictor(), settings, FakeScreen(), controls).play()
    assert controls.clicks[-1] == LAYOUT.guess_button
    assert LAYOUT.continue_button not in controls.clicks
