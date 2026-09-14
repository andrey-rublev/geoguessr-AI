import numpy as np
from PIL import Image

from geoguessr_ai.buttons import (
    MIN_SIMILARITY,
    button_image_path,
    button_region,
    similarity,
)
from geoguessr_ai.config import Point, Region


def draw_button():
    """A dark button with light lettering."""
    rgb = np.full((48, 160, 3), 40, np.uint8)
    for left in range(20, 140, 24):
        rgb[16:32, left : left + 12] = 230
    return Image.fromarray(rgb)


def test_the_same_button_matches_even_tinted_by_the_map_behind():
    button = draw_button()
    tinted = Image.fromarray((np.asarray(button) * 0.7 + (20, 60, 90)).astype(np.uint8))
    assert similarity(tinted, button) > 0.95


def test_an_advert_over_the_button_does_not_match():
    advert = np.full((48, 160, 3), 250, np.uint8)
    advert[4:12, 8:150] = 30  # a line of dark writing across the top
    assert similarity(Image.fromarray(advert), draw_button()) < MIN_SIMILARITY
    assert similarity(Image.new("RGB", (160, 48), "white"), draw_button()) == 0


def test_button_patch_and_snapshot_file():
    assert button_region(Point(960, 1040)) == Region(880, 1016, 160, 48)
    assert button_image_path("configs/layout.json").as_posix() == "configs/layout-continue.png"
