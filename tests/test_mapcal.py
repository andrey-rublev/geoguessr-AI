import cv2
import numpy as np
import pytest

from geoguessr_ai.mapcal import MapNotFound, MapProjection, locate_world, water_mask, world_mask

WATER_RGB = (90, 200, 235)
LAND_RGB = (245, 240, 229)
OFF_WORLD_RGB = (229, 227, 223)


def render_map(width: int, height: int, proj: MapProjection, seed: int = 0) -> np.ndarray:
    """Draw a fake Google-style map screenshot for a known projection."""
    rng = np.random.default_rng(seed)
    size = round(proj.world_px)
    world = world_mask(size, size) > 0
    world_rgb = np.where(world[..., None], WATER_RGB, LAND_RGB).astype(np.uint8)
    canvas = np.full((height, width, 3), OFF_WORLD_RGB, dtype=np.uint8)
    x, y = round(proj.origin_x), round(proj.origin_y)
    copies = range(-3, 4) if proj.wraps else [0]
    for k in copies:
        xx = x + k * size
        x0, y0, x1, y1 = max(xx, 0), max(y, 0), min(xx + size, width), min(y + size, height)
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = world_rgb[y0 - y : y1 - y, x0 - xx : x1 - xx]
    # Borders, roads and labels that the matcher has to ignore.
    for _ in range(40):
        p1 = tuple(int(v) for v in rng.integers(0, [width, height]))
        p2 = tuple(int(v) for v in rng.integers(0, [width, height]))
        cv2.line(canvas, p1, p2, (120, 120, 120), 1)
    for i in range(15):
        org = tuple(int(v) for v in rng.integers(0, [width - 60, height]))
        cv2.putText(canvas, f"Label{i}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    noise = rng.normal(0, 4, canvas.shape)
    return np.clip(canvas + noise, 0, 255).astype(np.uint8)


def assert_close(found: MapProjection, truth: MapProjection, width: int, height: int):
    # What matters is where a pin lands: compare projected pixels across the view.
    for lat, lon in [(51.5, -0.13), (-33.9, 18.4), (35.7, 139.7), (-23.5, -46.6), (40.7, -74.0)]:
        fx, fy = found.to_pixel(lat, lon, width)
        tx, ty = truth.to_pixel(lat, lon, width)
        if 0 <= tx < width and 0 <= ty < height:
            assert abs(fx - tx) < 2.0 and abs(fy - ty) < 2.0, (lat, lon, (fx, fy), (tx, ty))


def test_water_mask_classifies_map_colours():
    rgb = np.array([[WATER_RGB, LAND_RGB, OFF_WORLD_RGB, (200, 230, 200)]], dtype=np.uint8)
    np.testing.assert_array_equal(water_mask(rgb), [[1, -1, -1, -1]])


def test_projection_roundtrip_and_wrapping():
    proj = MapProjection(world_px=1000, origin_x=-200, origin_y=-150, wraps=True)
    x, y = proj.to_pixel(10.0, 170.0, view_width=800)
    assert 0 <= x < 800
    lat, lon = proj.to_latlon(x, y)
    assert (lat, lon) == pytest.approx((10.0, 170.0), abs=1e-6)


@pytest.mark.parametrize(
    "width,height,truth",
    [
        # OpenGuessr's expanded map at 150% display scale: most of the world, no wrapping.
        (845, 633, MapProjection(1600, -70, -260)),
        # Zoomed out past the whole world, with repeated copies either side.
        (640, 480, MapProjection(400, 20, 40, wraps=True)),
        # A smaller, zoomed-in view of Europe and Africa.
        (640, 480, MapProjection(2400, -1000, -500)),
    ],
)
def test_locate_world_recovers_projection(width, height, truth):
    rgb = render_map(width, height, truth)
    found = locate_world(rgb)
    assert found.score > 0.8
    assert_close(found, truth, width, height)


def test_locate_world_rejects_non_map():
    with pytest.raises(MapNotFound):
        locate_world(np.full((300, 400, 3), LAND_RGB, dtype=np.uint8))
