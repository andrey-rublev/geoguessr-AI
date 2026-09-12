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


def test_layout_load_missing_file_explains_next_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        Layout.load(tmp_path / "nope.json")
