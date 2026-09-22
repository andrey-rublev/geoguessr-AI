import numpy as np
import pytest
import torch
from PIL import Image

from geoguessr_ai.knowledge.countries import country_codes
from geoguessr_ai.knowledge.evidence import Evidence
from geoguessr_ai.model import predictor as predictor_module
from geoguessr_ai.model.geocells import GeoCells
from geoguessr_ai.model.head import Checkpoint, GeoHead

CENTROIDS = np.array([(48.86, 2.35), (35.68, 139.69), (-33.87, 151.21)])


class FakeEncoder:
    def __init__(self, name, device="auto"):
        self.name = name
        self.device = torch.device("cpu")
        self.seen = 0

    def encode(self, images):
        self.seen += len(images)
        return torch.ones(len(images), 8) / np.sqrt(8)


def test_predict_uses_every_crop_and_returns_likeliest_place(tmp_path, monkeypatch):
    head = GeoHead(embed_dim=8, n_cells=3, hidden=4)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-1].bias.copy_(torch.tensor([0.0, 6.0, 0.0]))  # always Tokyo
    path = tmp_path / "model.pt"
    Checkpoint(head, GeoCells(CENTROIDS), "fake/backbone").save(path)
    monkeypatch.setattr(predictor_module, "ImageEncoder", FakeEncoder)

    model = predictor_module.GeoPredictor(path)
    views = [Image.new("RGB", (1600, 800)), Image.new("RGB", (400, 400))]
    guess = model.predict(views)

    assert model.encoder.seen == 4  # a 2:1 view gives itself plus two square tiles
    assert (guess.lat, guess.lon) == pytest.approx(tuple(CENTROIDS[1]))
    assert guess.top_cells[0][2] > 0.9
    assert guess.expected_score > 4000


def test_prior_strength_corrects_for_crowded_training_regions(tmp_path, monkeypatch):
    head = GeoHead(embed_dim=8, n_cells=3, hidden=4)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-1].bias.copy_(torch.tensor([2.0, 2.5, 0.0]))  # leans Tokyo
    # ...but 85% of the training photos were in Tokyo, so the lean is mostly prior.
    log_prior = np.log(np.array([0.10, 0.85, 0.05]))
    path = tmp_path / "model.pt"
    Checkpoint(head, GeoCells(CENTROIDS), "fake/backbone", log_prior=log_prior).save(path)
    monkeypatch.setattr(predictor_module, "ImageEncoder", FakeEncoder)
    view = [Image.new("RGB", (200, 200))]

    raw = predictor_module.GeoPredictor(path, prior_strength=0.0).predict(view)
    debiased = predictor_module.GeoPredictor(path, prior_strength=1.0).predict(view)

    assert (raw.lat, raw.lon) == pytest.approx(tuple(CENTROIDS[1]))
    assert (debiased.lat, debiased.lon) == pytest.approx(tuple(CENTROIDS[0]))


def test_game_prior_from_played_rounds_tips_a_close_call(tmp_path, monkeypatch):
    head = GeoHead(embed_dim=8, n_cells=3, hidden=4)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-1].bias.copy_(torch.tensor([1.0, 1.2, 0.0]))  # a slight lean to Tokyo
    game = np.log(np.array([0.7, 0.2, 0.1]))  # but the game mostly sends players to Paris
    path = tmp_path / "model.pt"
    Checkpoint(head, GeoCells(CENTROIDS), "fake/backbone", game_log_prior=game).save(path)
    monkeypatch.setattr(predictor_module, "ImageEncoder", FakeEncoder)
    view = [Image.new("RGB", (200, 200))]

    ignored = predictor_module.GeoPredictor(path, game_prior_strength=0.0).predict(view)
    used = predictor_module.GeoPredictor(path).predict(view)

    assert (ignored.lat, ignored.lon) == pytest.approx(tuple(CENTROIDS[1]))
    assert (used.lat, used.lon) == pytest.approx(tuple(CENTROIDS[0]))


def test_street_view_coverage_and_clues_reweigh_places(tmp_path, monkeypatch):
    centroids = np.array([(48.86, 2.35), (35.68, 139.69), (39.90, 116.40)])  # Paris, Tokyo, Beijing
    head = GeoHead(embed_dim=8, n_cells=3, hidden=4)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-1].bias.copy_(torch.tensor([1.0, 1.5, 2.0]))  # leans Beijing, then Tokyo
    path = tmp_path / "model.pt"
    Checkpoint(head, GeoCells(centroids), "fake/backbone").save(path)
    monkeypatch.setattr(predictor_module, "ImageEncoder", FakeEncoder)
    view = [Image.new("RGB", (200, 200))]
    french_signs = Evidence(countries=np.where(np.array(country_codes()) == "FR", 1.0, 0.05))

    anywhere = predictor_module.GeoPredictor(path, coverage_strength=0.0).predict(view)
    covered = predictor_module.GeoPredictor(path, coverage_strength=1.0).predict(view)
    clued = predictor_module.GeoPredictor(path).predict(view, french_signs)

    assert (anywhere.lat, anywhere.lon) == pytest.approx(tuple(centroids[2]))
    # China has no Street View, so leaning on coverage moves the bet to Tokyo.
    assert (covered.lat, covered.lon) == pytest.approx(tuple(centroids[1]))
    assert covered.countries[0][0] == "JP" and covered.drives_left > 0.5
    assert (clued.lat, clued.lon) == pytest.approx(tuple(centroids[0]))


def test_predict_requires_images(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    Checkpoint(GeoHead(8, 3, hidden=4), GeoCells(CENTROIDS), "fake").save(path)
    monkeypatch.setattr(predictor_module, "ImageEncoder", FakeEncoder)
    with pytest.raises(ValueError):
        predictor_module.GeoPredictor(path).predict([])
