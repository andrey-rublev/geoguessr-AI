import numpy as np
import pytest
import torch
from PIL import Image

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

    assert model.encoder.seen == 3  # a 2:1 view is split into two square crops
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


def test_predict_requires_images(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    Checkpoint(GeoHead(8, 3, hidden=4), GeoCells(CENTROIDS), "fake").save(path)
    monkeypatch.setattr(predictor_module, "ImageEncoder", FakeEncoder)
    with pytest.raises(ValueError):
        predictor_module.GeoPredictor(path).predict([])
