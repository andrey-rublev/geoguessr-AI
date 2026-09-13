import numpy as np
import pytest

from geoguessr_ai.geo import haversine_km
from geoguessr_ai.model.geocells import GeoCells, debias


def test_log_prior_counts_training_photos_per_cell():
    cells = GeoCells(np.array([PARIS, TOKYO]))
    lat = np.array([48.9, 48.8, 49.0, 35.7])
    lon = np.array([2.3, 2.4, 2.2, 139.7])
    np.testing.assert_allclose(np.exp(cells.log_prior(lat, lon)), [4 / 6, 2 / 6])


def test_debias_removes_the_prior():
    log_prior = np.log(np.array([0.7, 0.2, 0.1]))
    np.testing.assert_allclose(debias(log_prior, log_prior, strength=1.0), [1 / 3] * 3)
    np.testing.assert_allclose(debias(log_prior, log_prior, strength=0.0), [0.7, 0.2, 0.1])
    np.testing.assert_allclose(debias(log_prior, None, strength=1.0), [0.7, 0.2, 0.1])
    batch = debias(np.stack([log_prior, log_prior]), log_prior, strength=0.5)
    assert batch.shape == (2, 3) and np.allclose(batch.sum(axis=1), 1)


def test_debias_can_add_the_games_own_prior():
    even = np.log(np.full(3, 1 / 3))
    game = np.log(np.array([0.6, 0.3, 0.1]))
    np.testing.assert_allclose(debias(even, None, 1.0, game, 1.0), [0.6, 0.3, 0.1])
    np.testing.assert_allclose(debias(even, None, 1.0, game, 0.0), [1 / 3] * 3)


PARIS = (48.86, 2.35)
BRUSSELS = (50.85, 4.35)
TOKYO = (35.68, 139.69)


def test_spread_log_prior_favours_played_places_without_overfitting():
    cells = GeoCells(np.array([PARIS, BRUSSELS, TOKYO]))
    one = np.exp(cells.spread_log_prior([48.9], [2.4]))
    many = np.exp(cells.spread_log_prior(np.full(500, 48.9), np.full(500, 2.4)))
    assert one.sum() == pytest.approx(1.0) and many.sum() == pytest.approx(1.0)
    assert one.max() < 0.4  # a single round barely moves it off an even third each
    assert many[0] > many[1] > many[2]  # Brussels is near Paris, Tokyo is not
    assert many[2] < 0.05


def test_fit_recovers_clusters():
    rng = np.random.default_rng(0)
    centres = np.array([PARIS, TOKYO, (-33.87, 151.21)])
    points = np.concatenate([c + rng.normal(0, 0.5, (200, 2)) for c in centres])
    cells = GeoCells.fit(points[:, 0], points[:, 1], n_cells=3)
    assert len(cells) == 3
    for lat, lon in centres:
        idx = cells.assign(lat, lon)
        assert haversine_km(lat, lon, *cells.centroids[idx]) < 50


def test_soft_targets_favour_nearest_cells():
    cells = GeoCells(np.array([PARIS, BRUSSELS, TOKYO]))
    targets = cells.soft_targets(np.array([49.0]), np.array([2.5]), tau_km=150)
    assert targets.shape == (1, 3)
    assert targets.sum() == pytest.approx(1.0)
    assert targets[0, 0] > targets[0, 1] > targets[0, 2]


def test_best_guess_hedges_between_nearby_cells():
    cells = GeoCells(np.array([PARIS, BRUSSELS, TOKYO]))
    # Tokyo is the single most likely cell, but Europe holds more probability overall.
    lat, lon, expected = cells.best_guess(np.array([0.3, 0.3, 0.4]))
    assert (lat, lon) in (PARIS, BRUSSELS)
    assert 0 < expected < 5000


def test_best_guess_confident_prediction():
    cells = GeoCells(np.array([PARIS, BRUSSELS, TOKYO]))
    lat, lon, expected = cells.best_guess(np.array([0.0, 0.0, 1.0]))
    assert (lat, lon) == TOKYO
    assert expected == pytest.approx(5000.0)
