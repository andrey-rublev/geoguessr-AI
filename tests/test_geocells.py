import numpy as np
import pytest

from geoguessr_ai.geo import haversine_km
from geoguessr_ai.model.geocells import GeoCells

PARIS = (48.86, 2.35)
BRUSSELS = (50.85, 4.35)
TOKYO = (35.68, 139.69)


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
