import pytest

from geoguessr_ai.config import Layout, Point, Region


def test_region_from_corners_normalises_order():
    region = Region.from_corners(Point(500, 400), Point(100, 50))
    assert region == Region(100, 50, 400, 350)
    assert region.center == Point(300, 225)
    assert region.contains(Point(100, 50)) and not region.contains(Point(500, 400))
    assert region.to_screen(10.4, 20.6) == Point(110, 71)


def test_region_rejects_tiny_selection():
    with pytest.raises(ValueError):
        Region.from_corners(Point(10, 10), Point(15, 200))


def test_layout_roundtrip(tmp_path):
    layout = Layout(
        view=Region(0, 80, 1920, 800),
        map_hover=Point(1700, 850),
        map_region=Region(1000, 400, 880, 600),
        guess_button=Point(1700, 1040),
        continue_button=Point(960, 1040),
    )
    path = tmp_path / "layout.json"
    layout.save(path)
    assert Layout.load(path) == layout


def test_multiplayer_layout_roundtrip_without_continue(tmp_path):
    layout = Layout(
        view=Region(0, 80, 1920, 800),
        map_hover=Point(1700, 850),
        map_region=Region(1000, 400, 880, 600),
        guess_button=Point(1700, 1040),
        continue_button=None,
        timer=Point(1850, 1010),
        covered=(Region(0, 600, 300, 200),),
        page=Region(0, 0, 1920, 1080),
        party=True,
        minimap=Region(1530, 760, 320, 160),
    )
    path = tmp_path / "layout-party.json"
    layout.save(path)
    assert Layout.load(path) == layout


def test_older_layouts_are_multiplayer_ones_when_they_have_no_continue(tmp_path):
    path = tmp_path / "layout-party.json"
    path.write_text(
        '{"view": {"left": 0, "top": 80, "width": 1920, "height": 800},'
        ' "map_hover": {"x": 1700, "y": 850},'
        ' "map_region": {"left": 1000, "top": 400, "width": 880, "height": 600},'
        ' "guess_button": {"x": 1700, "y": 1040}, "continue_button": null}'
    )
    layout = Layout.load(path)
    assert layout.party and layout.page is None


def test_layout_load_missing_file_explains_next_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        Layout.load(tmp_path / "nope.json")


def test_the_picture_reaches_down_to_the_bottom_of_the_collapsed_map():
    # As found in a 1280 by 650 window, where the map comes halfway down
    layout = Layout(
        view=Region(22, 70, 924, 290),
        map_hover=Point(1082, 444),
        map_region=Region(506, 161, 703, 358),
        guess_button=Point(1082, 615),
        continue_button=None,
        covered=(Region(12, 380, 298, 190),),
        minimap=Region(954, 369, 256, 150),
    )
    assert layout.picture == Region(22, 70, 1188, 449)
    assert layout.hidden == (Region(12, 380, 298, 190), Region(954, 369, 256, 150))
    by_hand = Layout(**{**layout.__dict__, "minimap": None})
    assert by_hand.picture == by_hand.view and by_hand.hidden == by_hand.covered
