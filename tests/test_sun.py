import cv2
import numpy as np
import pytest

from geoguessr_ai.knowledge.sun import find_sun, latitude_likelihood, sun_azimuth


def sky(sun_at=None, clouds=()):
    """A blue sky over green ground, maybe with a glowing sun and white clouds."""
    img = np.zeros((500, 1400, 3), np.uint8)
    img[:] = (110, 160, 220)
    img[320:] = (70, 110, 60)
    if sun_at is not None:
        glow = np.zeros(img.shape[:2], np.float32)
        cv2.circle(glow, sun_at, 1, 1.0, -1)
        glow = cv2.GaussianBlur(glow, (0, 0), 30)
        glow /= glow.max()
        img = np.clip(img + glow[..., None] * 255, 0, 255).astype(np.uint8)
        cv2.circle(img, sun_at, 12, (255, 255, 255), -1)
    for x, y in clouds:
        cv2.ellipse(img, (x, y), (120, 40), 0, 0, 360, (255, 255, 255), -1)
    return img


def test_finds_a_low_sun():
    found = find_sun(sky(sun_at=(900, 150)))
    assert found is not None and np.hypot(found[0] - 900, found[1] - 150) < 3


@pytest.mark.parametrize("clouds", [[], [(300, 100), (1000, 200)]])
def test_no_sun_in_a_plain_or_cloudy_sky(clouds):
    assert find_sun(sky(clouds=clouds)) is None


def test_azimuth_from_position_in_a_view():
    assert sun_azimuth(700, 1400, 180) == pytest.approx(180)
    assert sun_azimuth(0, 1400, 0) == pytest.approx(360 - 52.5)


def test_a_southern_sun_means_the_north_and_vice_versa():
    south = latitude_likelihood(180)
    north = latitude_likelihood(0)
    assert south(45) > 3 * south(-35) and north(-35) > 3 * north(45)
    assert 0.25 <= min(south(np.arange(-60, 76))) and max(south(np.arange(-60, 76))) == 1
