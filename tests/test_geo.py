import numpy as np
import pytest

from geoguessr_ai.geo import (
    from_unit_vectors,
    geoguessr_score,
    haversine_km,
    inverse_mercator,
    mercator_xy,
    to_unit_vectors,
)


def test_haversine_paris_london():
    assert haversine_km(48.8566, 2.3522, 51.5074, -0.1278) == pytest.approx(343.5, abs=1.0)


def test_haversine_broadcasts():
    d = haversine_km(0.0, 0.0, np.array([0.0, 0.0]), np.array([0.0, 180.0]))
    assert d.shape == (2,)
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(np.pi * 6371.0088, rel=1e-6)


def test_unit_vector_roundtrip():
    lat = np.array([-60.0, 0.0, 45.5, 89.0])
    lon = np.array([-170.0, 0.0, 12.25, 179.0])
    back_lat, back_lon = from_unit_vectors(to_unit_vectors(lat, lon))
    np.testing.assert_allclose(back_lat, lat, atol=1e-9)
    np.testing.assert_allclose(back_lon, lon, atol=1e-9)


def test_mercator_known_points_and_roundtrip():
    assert mercator_xy(0.0, 0.0) == pytest.approx((0.5, 0.5))
    x, y = mercator_xy(85.05112878, -180.0)
    assert x == pytest.approx(0.0) and y == pytest.approx(0.0, abs=1e-6)
    lat, lon = inverse_mercator(*mercator_xy(53.235, 7.462))
    assert (lat, lon) == pytest.approx((53.235, 7.462))


def test_geoguessr_score():
    assert geoguessr_score(0.0) == pytest.approx(5000.0)
    assert geoguessr_score(1492.7) == pytest.approx(5000.0 / np.e)
