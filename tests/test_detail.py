import numpy as np
import pytest
from PIL import Image

from geoguessr_ai.knowledge.countries import country_codes
from geoguessr_ai.model.detail import DetailExpert, fit, tiles

FR, DE = country_codes().index("FR"), country_codes().index("DE")


def test_views_are_cut_into_close_up_tiles_row_by_row():
    view = Image.new("RGB", (800, 300))
    view.paste((255, 0, 0), (0, 150, 200, 300))  # bottom-left corner
    cut = tiles(view, rows=2, cols=4)
    assert len(cut) == 8 and all(t.size == (200, 150) for t in cut)
    assert cut[4].getpixel((10, 10)) == (255, 0, 0) and cut[0].getpixel((10, 10)) == (0, 0, 0)


def bollards():
    """Tiles of two countries' roadsides, told apart by one feature, like a bollard's cap."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 8))
    country = np.where(np.arange(200) % 2 == 0, FR, DE)
    emb[:, 0] += np.where(country == FR, 2.0, -2.0)
    return emb, country


def test_learns_which_country_details_belong_to_and_says_so_as_a_likelihood():
    emb, country = bollards()
    expert = fit(emb, country, "fake/backbone")
    french = np.zeros((4, 8))
    french[:, 0] = 2.0
    like = expert.likelihood(french)
    assert like.shape == (len(country_codes()),) and like.max() == pytest.approx(1.0)
    assert like[FR] == 1.0 and like[DE] < 0.1
    japan = like[country_codes().index("JP")]
    assert like[DE] < japan < like[FR]  # never seen, so neither for nor against
    assert expert.likelihood(french, weight=0.0)[DE] == pytest.approx(1.0)  # weight 0: no say


def test_saves_and_loads_with_its_weight(tmp_path):
    emb, country = bollards()
    expert = fit(emb, country, "fake/backbone")
    expert.weight = 0.7
    expert.save(tmp_path / "detail.npz")
    loaded = DetailExpert.load(tmp_path / "detail.npz")
    assert loaded.weight == 0.7 and loaded.backbone == "fake/backbone"
    np.testing.assert_allclose(loaded.likelihood(emb[:5]), expert.likelihood(emb[:5]))
