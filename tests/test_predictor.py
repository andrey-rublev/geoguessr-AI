import numpy as np
import pytest
import torch
from PIL import Image

from geoguessr_ai.knowledge.countries import country_codes
from geoguessr_ai.knowledge.evidence import Evidence
from geoguessr_ai.model import predictor as predictor_module
from geoguessr_ai.model.detail import COLS, ROWS, fit
from geoguessr_ai.model.geocells import GeoCells
from geoguessr_ai.model.head import Checkpoint, Ensemble, GeoHead
from geoguessr_ai.model.predictor import detail_path

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
    assert model.source.startswith("model.pt, saved 20")  # which model, saved with each round
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


def sure_of(cell: int, centroids: np.ndarray) -> Checkpoint:
    head = GeoHead(embed_dim=8, n_cells=len(centroids), hidden=4)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-1].bias[cell] = 9.0
    return Checkpoint(head, GeoCells(centroids), "fake/backbone")


def test_an_ensemble_averages_its_members_on_the_newest_ones_cells(tmp_path, monkeypatch):
    paris = sure_of(0, CENTROIDS)
    tokyo = sure_of(1, CENTROIDS + 0.3)  # cells refitted, a little elsewhere
    path = tmp_path / "model.pt"
    Ensemble([paris, tokyo]).save(path)
    monkeypatch.setattr(predictor_module, "ImageEncoder", FakeEncoder)

    guess = predictor_module.GeoPredictor(path).predict([Image.new("RGB", (200, 200))])

    shares = {(round(lat), round(lon)): p for lat, lon, p in guess.top_cells}
    assert shares[(49, 2)] == pytest.approx(0.5, abs=0.01)  # on the first member's cells
    assert shares[(36, 140)] == pytest.approx(0.5, abs=0.01)
    with pytest.raises(ValueError, match="Ensemble"):
        Checkpoint.load(path)
    single = tmp_path / "single.pt"
    paris.save(single)
    assert len(Ensemble.load(single).members) == 1  # a checkpoint on its own is an ensemble too


def test_close_ups_of_the_views_reweigh_places_once_a_detail_expert_is_learned(
    tmp_path, monkeypatch
):
    head = GeoHead(embed_dim=8, n_cells=3, hidden=4)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-1].bias.copy_(torch.tensor([1.0, 1.5, 0.0]))  # leans Tokyo
    path = tmp_path / "model.pt"
    Checkpoint(head, GeoCells(CENTROIDS), "fake/backbone").save(path)
    monkeypatch.setattr(predictor_module, "ImageEncoder", FakeEncoder)
    view = [Image.new("RGB", (800, 400))]
    before = predictor_module.GeoPredictor(path).predict(view)

    # Close-ups like the fake encoder's every tile were French in training: a French bollard.
    rng = np.random.default_rng(0)
    seen = rng.normal(size=(200, 8)) * 0.3
    country = np.where(np.arange(200) % 2 == 0, "FR", "JP")
    seen[country == "FR"] += 1 / np.sqrt(8)
    codes = [country_codes().index(c) for c in country]
    fit(seen, codes, "fake/backbone").save(detail_path(path))
    model = predictor_module.GeoPredictor(path)
    after = model.predict(view)

    assert (before.lat, before.lon) == pytest.approx(tuple(CENTROIDS[1]))
    assert (after.lat, after.lon) == pytest.approx(tuple(CENTROIDS[0]))
    assert model.encoder.seen == 3 + ROWS * COLS  # the model's crops, then the close-ups
    looked = [Image.new("RGB", (800, 400)) for _ in range(4)]
    model.predict(looked)
    seen = model.encoder.seen
    model.predict([*looked, *(Image.new("RGB", (800, 400)) for _ in range(4))])  # walked on
    assert model.encoder.seen == seen + 8 * 3  # the first look's close-ups weren't read again


def test_places_the_guess_by_the_photos_that_look_most_alike(tmp_path, monkeypatch):
    head = GeoHead(embed_dim=8, n_cells=3, hidden=4)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-1].bias.copy_(torch.tensor([0.0, 6.0, 0.0]))  # always Tokyo's cell
    path = tmp_path / "model.pt"
    Checkpoint(head, GeoCells(CENTROIDS), "fake/backbone").save(path)
    monkeypatch.setattr(predictor_module, "ImageEncoder", FakeEncoder)
    folder = tmp_path / "embeddings" / "fake__backbone"
    folder.mkdir(parents=True)
    alike, unlike = np.ones(8) / np.sqrt(8), np.eye(8)[0]  # FakeEncoder sees every view as alike
    yokohama, saitama = (35.44, 139.64), (35.86, 139.65)
    np.savez(
        folder / "osv5m-train-00.npz",
        ids=np.array(["a", "b"]),
        lat=np.array([yokohama[0], saitama[0]], np.float32),
        lon=np.array([yokohama[1], saitama[1]], np.float32),
        embeddings=np.stack([alike, unlike]).astype(np.float16),
        backbone=np.array("fake/backbone"),
    )

    model = predictor_module.GeoPredictor(path, embeddings=tmp_path / "embeddings")
    guess = model.predict([Image.new("RGB", (400, 400))])

    assert len(model.neighbours) == 2
    assert (guess.lat, guess.lon) == pytest.approx(yokohama, abs=0.01)
    assert guess.top_cells[0][:2] == pytest.approx(tuple(CENTROIDS[1]))  # the cell, as chosen


def test_photos_that_look_alike_vote_for_their_cell(tmp_path, monkeypatch):
    head = GeoHead(embed_dim=8, n_cells=3, hidden=4)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-1].bias.copy_(torch.tensor([1.0, 1.2, 0.0]))  # a slight lean to Tokyo
    path = tmp_path / "model.pt"
    Checkpoint(head, GeoCells(CENTROIDS), "fake/backbone").save(path)
    monkeypatch.setattr(predictor_module, "ImageEncoder", FakeEncoder)
    folder = tmp_path / "embeddings" / "fake__backbone"
    folder.mkdir(parents=True)
    alike, unlike = np.ones(8) / np.sqrt(8), np.eye(8)[0]  # FakeEncoder sees every view as alike
    versailles, yokohama = (48.80, 2.13), (35.44, 139.64)
    np.savez(
        folder / "osv5m-train-00.npz",
        ids=np.array(["a", "b"]),
        lat=np.array([versailles[0], yokohama[0]], np.float32),
        lon=np.array([versailles[1], yokohama[1]], np.float32),
        embeddings=np.stack([alike, unlike]).astype(np.float16),
        backbone=np.array("fake/backbone"),
    )
    view = [Image.new("RGB", (400, 400))]

    alone = predictor_module.GeoPredictor(path).predict(view)
    voted = predictor_module.GeoPredictor(path, embeddings=tmp_path / "embeddings").predict(view)

    assert (alone.lat, alone.lon) == pytest.approx(tuple(CENTROIDS[1]))
    assert (voted.lat, voted.lon) == pytest.approx(versailles, abs=0.01)
    assert voted.countries[0][0] == "FR"
