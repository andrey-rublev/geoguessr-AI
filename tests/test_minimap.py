import numpy as np
from PIL import Image
from test_result import LAND, OCEAN, OFF_WORLD, PIN_RED, WATER, paint

from geoguessr_ai.config import Point, Region
from geoguessr_ai.geo import haversine_km
from geoguessr_ai.mapcal import MapProjection
from geoguessr_ai.minimap import GuessMap

REGION = Region(300, 200, 692, 348)
HOVER = Point(900, 500)
# Zoomed out as far as the game allows: about two thirds of the world, centred on the Atlantic.
ATLANTIC = MapProjection(1024, -195, -262, wraps=True)
MADRID, TOKYO = (40.42, -3.70), (35.68, 139.69)


class FakeMinimap:
    """Screen and mouse for a Leaflet map: the wheel zooms around the cursor, down to a minimum
    zoom; drags pan; clicks drop the pin."""

    def __init__(self, projection, min_world=1024, lost_clicks=0):
        self.projection, self.min_world, self.lost_clicks = projection, min_world, lost_clicks
        self.pin, self.drags = None, 0

    def grab(self, region):
        assert region == REGION
        p = self.projection
        ys, xs = np.mgrid[0 : region.height, 0 : region.width]
        mx, my = (xs - p.origin_x) / p.world_px, (ys - p.origin_y) / p.world_px
        rows = np.clip(np.floor(my * 4096), 0, 4095).astype(int)
        cols = np.floor(mx * 4096).astype(np.int64) % 4096
        rgb = np.where(OCEAN[rows, cols][..., None], WATER, LAND).astype(np.float32)
        rgb[(my < 0) | (my >= 1)] = OFF_WORLD
        if self.pin:
            paint(rgb, "pin", *p.to_pixel(*self.pin, region.width), PIN_RED)
        return Image.fromarray(rgb.round().astype(np.uint8))

    def sleep(self, seconds):
        pass

    def move(self, p):
        pass

    def click(self, p):
        if self.lost_clicks:
            self.lost_clicks -= 1
        else:
            self.pin = self.projection.to_latlon(p.x - REGION.left, p.y - REGION.top)

    def drag(self, start, dx, dy=0):
        assert REGION.contains(start) and REGION.contains(start.offset(dx, dy))
        self.projection = self.projection.panned(dx, dy)
        self.drags += 1

    def scroll(self, p, clicks):
        for _ in range(abs(clicks)):
            factor = 2.0 if clicks > 0 else 0.5
            zoomed = self.projection.zoomed(p.x - REGION.left, p.y - REGION.top, factor)
            if zoomed.world_px >= self.min_world:
                self.projection = zoomed


def test_zooms_in_to_place_the_pin_and_back_out():
    fake = FakeMinimap(ATLANTIC)

    placement = GuessMap(fake, fake, REGION, HOVER).place(*MADRID)

    assert placement.confirmed and fake.drags == 0
    assert haversine_km(*fake.pin, *MADRID) < 40
    assert haversine_km(placement.lat, placement.lon, *fake.pin) < 40
    assert fake.projection.world_px == ATLANTIC.world_px  # ready for the next round


def test_drags_a_guess_beyond_the_edge_into_view():
    fake = FakeMinimap(ATLANTIC)
    assert ATLANTIC.to_pixel(*TOKYO, REGION.width)[0] > REGION.width

    placement = GuessMap(fake, fake, REGION, HOVER).place(*TOKYO)

    assert fake.drags >= 1 and placement.confirmed
    assert haversine_km(*fake.pin, *TOKYO) < 40


def test_zooms_out_first_when_the_map_was_left_zoomed_in():
    fake = FakeMinimap(ATLANTIC.zoomed(346, 174, 16))  # all Sahara: nothing to recognise

    GuessMap(fake, fake, REGION, HOVER).place(*MADRID)

    assert haversine_km(*fake.pin, *MADRID) < 40


def test_clicks_again_when_the_first_click_is_lost():
    fake = FakeMinimap(ATLANTIC, lost_clicks=1)

    placement = GuessMap(fake, fake, REGION, HOVER).place(*MADRID)

    assert placement.confirmed and haversine_km(*fake.pin, *MADRID) < 40


def test_reports_a_pin_that_never_appears():
    fake = FakeMinimap(ATLANTIC, lost_clicks=5)
    assert GuessMap(fake, fake, REGION, HOVER).place(*MADRID).confirmed is False
