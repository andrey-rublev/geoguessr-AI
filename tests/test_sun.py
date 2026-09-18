import cv2
import numpy as np
import pytest

from geoguessr_ai.knowledge.sun import FLOOR, clear_sky, find_sun, latitude_likelihood


def sky(sun_at=None, clouds=(), size=(500, 1400), sun_radius=12, glow=30, ground=0.64):
    """A blue sky over green ground, maybe with a glowing sun and white clouds."""
    img = np.zeros((*size, 3), np.uint8)
    img[:] = (110, 160, 220)
    img[round(size[0] * ground) :] = (70, 110, 60)
    if sun_at is not None:
        halo = np.zeros(img.shape[:2], np.float32)
        cv2.circle(halo, sun_at, 1, 1.0, -1)
        halo = cv2.GaussianBlur(halo, (0, 0), glow)
        halo /= halo.max()
        img = np.clip(img + halo[..., None] * 255, 0, 255).astype(np.uint8)
        cv2.circle(img, sun_at, sun_radius, (255, 255, 255), -1)
    for x, y in clouds:
        cv2.ellipse(img, (x, y), (120, 40), 0, 0, 360, (255, 255, 255), -1)
    return img


def test_finds_a_low_sun():
    found = find_sun(sky(sun_at=(900, 150)))
    assert found is not None and np.hypot(found[0] - 900, found[1] - 150) < 3


def test_finds_a_high_sun_glaring_in_a_view_tilted_up():
    glare = sky(sun_at=(500, 250), size=(600, 1000), sun_radius=50, glow=110, ground=1.0)
    found = find_sun(glare)
    assert found is not None and np.hypot(found[0] - 500, found[1] - 250) < 5


@pytest.mark.parametrize("clouds", [[], [(300, 100), (1000, 200)]])
def test_no_sun_in_a_plain_or_cloudy_sky(clouds):
    assert find_sun(sky(clouds=clouds)) is None


def test_no_sun_in_a_white_overcast_sky():
    overcast = np.full((500, 900, 3), 238, np.uint8)
    cv2.ellipse(overcast, (450, 200), (160, 120), 0, 0, 360, (255, 255, 255), -1)
    overcast = cv2.GaussianBlur(overcast, (0, 0), 25)
    overcast[overcast >= 246] = 255
    assert find_sun(overcast) is None


def test_no_sun_cut_off_by_the_edge_of_the_view():
    assert find_sun(sky(sun_at=(700, 5), sun_radius=40)) is None


def test_clear_sky_is_blue_sky_at_the_top():
    assert clear_sky(sky()) > 0.9
    assert clear_sky(np.full((300, 400, 3), 200, np.uint8)) == 0


def test_a_southern_sun_means_the_north_and_vice_versa():
    south = latitude_likelihood(180, 20)
    north = latitude_likelihood(0, 20)
    assert south(45) > 3 * south(-35) and north(-35) > 3 * north(45)
    everywhere = south(np.arange(-60, 76))
    assert everywhere.min() >= FLOOR and everywhere.max() == 1


def test_a_sun_overhead_means_the_tropics():
    overhead = latitude_likelihood(180, 85)
    assert overhead(5) == 1 and overhead(50) == FLOOR and overhead(-45) == FLOOR
