import numpy as np

from geoguessr_ai.knowledge.countries import country_codes
from geoguessr_ai.knowledge.evidence import Evidence, cell_countries, cell_weights

CODES = country_codes()
# Geocells every 2 degrees across Europe and Asia.
GRID = np.array([(lat, lon) for lat in np.arange(30, 52, 2.0) for lon in np.arange(0, 142, 2.0)])


def cell(lat, lon):
    return int(np.argmin(np.hypot(GRID[:, 0] - lat, GRID[:, 1] - lon)))


def test_cells_belong_to_the_countries_around_them():
    shares = cell_countries(GRID)

    assert np.allclose(shares.sum(axis=1), 1)
    assert shares[cell(36, 140), CODES.index("JP")] > 0.8
    assert shares[cell(36, 128), CODES.index("KR")] > 0.8
    border = shares[cell(48, 8)]  # the Rhine between Germany and France
    assert border[CODES.index("DE")] > 0.3 and border[CODES.index("FR")] > 0.1


def test_places_without_street_view_weigh_little():
    weights = cell_weights(GRID)
    assert weights[cell(40, 116)] < 0.05 * weights[cell(36, 140)]  # Beijing vs Tokyo


def test_evidence_about_country_and_latitude_reweighs_cells():
    korean = Evidence(countries=np.where(np.array(CODES) == "KR", 1.0, 0.05))
    weights = cell_weights(GRID, korean)
    assert weights[cell(36, 128)] > 10 * weights[cell(36, 140)]

    south_of_40 = Evidence(latitude=lambda lat: np.where(lat < 40, 1.0, 0.1))
    weights = cell_weights(GRID, south_of_40)
    assert weights[cell(36, 140)] > 5 * weights[cell(44, 142)]
