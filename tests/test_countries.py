import numpy as np
import pytest

from geoguessr_ai.knowledge.countries import country_at, country_codes, country_index


@pytest.mark.parametrize(
    "lat,lon,code",
    [
        (37.57, 126.98, "KR"),  # Seoul
        (35.68, 139.69, "JP"),  # Tokyo
        (-29.31, 27.48, "LS"),  # Maseru, surrounded by South Africa
        (43.94, 12.45, "SM"),  # San Marino, surrounded by Italy
        (1.35, 103.82, "SG"),
        (-33.87, 151.21, "AU"),
        (38.70, -9.60, "PT"),  # just off Lisbon: the sea belongs to the nearest country
    ],
)
def test_country_at(lat, lon, code):
    assert country_at(lat, lon) == code


def test_country_index_takes_arrays_and_wraps_longitude():
    idx = country_index(np.array([37.57, 37.57]), np.array([126.98, 126.98 - 360]))
    assert idx[0] == idx[1] and country_codes()[idx[0]] == "KR"
