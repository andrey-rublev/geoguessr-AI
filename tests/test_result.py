import math

import cv2
import numpy as np
import pytest
from PIL import Image

from geoguessr_ai import result as result_module
from geoguessr_ai.config import Layout, Point, Region
from geoguessr_ai.geo import haversine_km, mercator_xy
from geoguessr_ai.mapcal import MapProjection, world_mask
from geoguessr_ai.ocr import Word
from geoguessr_ai.result import (
    MARKERS,
    Marker,
    ResultReader,
    find_marker,
    find_scale,
    map_by_coasts,
    map_by_towns,
    marker_shades,
    our_labelled_pin,
    result_region,
)

WATER, LAND, OFF_WORLD = (90, 200, 235), (245, 240, 229), (229, 227, 223)
OCEAN = world_mask(4096, 4096) > 0
MADRID, LISBON = (40.42, -3.70), (38.72, -9.14)
REGION = Region(100, 200, 1200, 500)
PIN_RED, FLAG_BLACK = (235, 35, 35), (15, 15, 15)


def paint(rgb, name, x, y, colour, scale=1.5):
    """Draw a marker from its template, anchored at (x, y), the way the detector expects it."""
    template, (ax, ay) = MARKERS[name]
    th, tw = template.shape
    alpha = cv2.resize(template, (round(tw * scale), round(th * scale)))
    left, top = round(x - (ax + 0.5) * scale + 0.5), round(y - (ay + 0.5) * scale + 0.5)
    y0, x0 = max(top, 0), max(left, 0)
    y1 = min(top + alpha.shape[0], rgb.shape[0])
    x1 = min(left + alpha.shape[1], rgb.shape[1])
    if y1 > y0 and x1 > x0:
        a = alpha[y0 - top : y1 - top, x0 - left : x1 - left, None]
        rgb[y0:y1, x0:x1] = rgb[y0:y1, x0:x1] * (1 - a) + np.array(colour) * a


def fitted(world, a=MADRID, b=LISBON):
    """The view the game opens with: both places centred at the given zoom."""
    (ax, ay), (bx, by) = mercator_xy(*a), mercator_xy(*b)
    cx, cy = (ax + bx) / 2, (ay + by) / 2
    return MapProjection(world, REGION.width / 2 - cx * world, REGION.height / 2 - cy * world, True)


class FakeResultScreen:
    """A result map at any zoom, drawn from the land mask, with our pin and the answer's flag,
    and in a multiplayer room the pins of ``others``."""

    def __init__(self, projection, pin=MADRID, flag=LISBON, show_flag=True, others=()):
        self.projection, self.pin, self.flag, self.show_flag = projection, pin, flag, show_flag
        self.others = others

    def grab(self, region):
        assert region == REGION
        p = self.projection
        ys, xs = np.mgrid[0 : region.height, 0 : region.width]
        mx, my = (xs - p.origin_x) / p.world_px, (ys - p.origin_y) / p.world_px
        rows = np.clip(np.floor(my * 4096), 0, 4095).astype(int)
        cols = np.floor(mx * 4096).astype(np.int64) % 4096
        rgb = np.where(OCEAN[rows, cols][..., None], WATER, LAND).astype(np.float32)
        rgb[(my < 0) | (my >= 1)] = OFF_WORLD
        for place in (*self.others, self.pin):
            paint(rgb, "pin", *p.to_pixel(*place, region.width), PIN_RED)
        if self.show_flag:  # the game draws the flag over the pin
            paint(rgb, "flag", *p.to_pixel(*self.flag, region.width), FLAG_BLACK)
        return Image.fromarray(rgb.round().astype(np.uint8))


class FakeControls:
    """Each wheel notch halves the map's scale around the cursor, like Leaflet does, until the
    world is ``widest`` pixels across: the notch that gets there goes only part of the way."""

    def __init__(self, screen, widest=0.0):
        self.screen, self.notches, self.widest, self.t = screen, 0, widest, 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds

    def move(self, p):
        pass

    def scroll(self, p, clicks):
        assert clicks < 0, "the reader only ever zooms out"
        for _ in range(-clicks):
            cx, cy = p.x - REGION.left, p.y - REGION.top
            q = self.screen.projection
            if q.world_px <= self.widest:
                return
            self.screen.projection = q.zoomed(cx, cy, max(0.5, self.widest / q.world_px))
            self.notches += 1


def test_markers_are_found_at_any_display_scale():
    rgb = np.full((300, 400, 3), LAND, np.float32)
    paint(rgb, "pin", 150.0, 200.0, PIN_RED, scale=2.0)
    paint(rgb, "flag", 300.0, 120.0, FLAG_BLACK, scale=2.0)
    red, dark = marker_shades(rgb.astype(np.uint8))

    scale = find_scale(red)

    assert scale == pytest.approx(2.0, rel=0.05)
    pin, flag = find_marker(red, "pin", scale), find_marker(dark, "flag", scale)
    assert pin.score > 0.9 and math.dist((pin.x, pin.y), (150, 200)) < 2.5
    assert flag.score > 0.9 and math.dist((flag.x, flag.y), (300, 120)) < 2.5


def test_reads_the_answer_after_zooming_out():
    screen = FakeResultScreen(fitted(24_576))
    controls = FakeControls(screen)

    reading = ResultReader(screen, controls, REGION).read(*MADRID)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 10
    assert controls.notches >= 2 and reading.answer.zoom_outs == 0
    assert reading.screenshot.size == (REGION.width, REGION.height)
    assert reading.zoomed_out == []  # kept only to see why an answer couldn't be read


def reader_recognising_only_the_widest_view():
    """The world is only recognised at the map's widest, which the last notch zooms out to only
    2^0.62 times: counted as a halving, Lisbon would come out 1.3 times too far from Madrid."""
    screen = FakeResultScreen(fitted(24_576))
    widest = 3_072 / 2**0.62
    reader = ResultReader(screen, FakeControls(screen, widest=widest), REGION)
    locate = reader._locate
    reader._locate = lambda rgb: locate(rgb) if screen.projection.world_px == widest else None
    return reader


def test_the_notch_that_reaches_the_widest_view_counts_for_what_it_zoomed():
    reading = reader_recognising_only_the_widest_view().read(*MADRID)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 10


def test_a_miscounted_zoom_is_put_right_by_the_whole_zoom_levels(monkeypatch):
    reader = reader_recognising_only_the_widest_view()
    monkeypatch.setattr(reader, "_halvings_between", lambda *args: 1.0)  # the last was 0.62

    reading = reader.read(*MADRID)

    assert reading.answer is not None and reading.answer.zoom_outs == 0, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 10


def test_zoom_steps_counted_far_wrong_are_caught_by_the_recognised_map(monkeypatch):
    reader = reader_recognising_only_the_widest_view()
    monkeypatch.setattr(reader, "_halvings_between", lambda *args: 2.0)

    reading = reader.read(*MADRID)

    # Read where the world was recognised, no zoom steps counted, if not as finely as further in.
    assert reading.answer is not None and reading.answer.zoom_outs == 4, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 30


def test_reads_the_answer_off_a_map_recognised_a_few_pixels_out():
    screen = FakeResultScreen(fitted(24_576))
    reader = ResultReader(screen, FakeControls(screen), REGION)
    locate = reader._locate

    def a_little_out(rgb):  # the pin is still within reach of where we clicked
        p = locate(rgb)
        return p and MapProjection(p.world_px, p.origin_x + 7, p.origin_y, p.wraps, p.score)

    reader._locate = a_little_out
    reading = reader.read(*MADRID)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 10


def test_a_first_view_recognised_off_a_whole_zoom_level_is_looked_at_again():
    screen = FakeResultScreen(fitted(3_072))  # recognisable at once, at zoom level 3
    reader = ResultReader(screen, FakeControls(screen), REGION)
    locate = reader._locate
    seen = []

    def wrong_at_first(rgb):  # the world 0.77 times its size, our pin landing right by chance
        p = locate(rgb)
        seen.append(p)
        if p is None or len(seen) > 1:
            return p
        world = p.world_px * 0.77
        (x, y), (mx, my) = p.to_pixel(*MADRID, REGION.width), mercator_xy(*MADRID)
        return MapProjection(world, x - mx * world, y - my * world, p.wraps, p.score)

    reader._locate = wrong_at_first
    reading = reader.read(*MADRID)

    assert seen[0] is not None, "the first view should be recognisable"
    assert reading.answer is not None and reading.answer.zoom_outs == 0, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 20


def test_reads_the_answer_among_other_players_pins():
    paris, porto = (48.86, 2.35), (41.15, -8.61)  # the others guessed here, nearer the flag
    screen = FakeResultScreen(fitted(3_072, MADRID, paris), others=(paris, porto))
    reader = ResultReader(screen, FakeControls(screen), REGION)
    reader.scale = 1.5  # as the guess map measured it

    reading = reader.read(*MADRID, crowded=True)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 20


def test_among_other_players_waits_for_a_pin_to_show_before_telling_ours():
    tokyo = (35.68, 139.69)  # our pin off the map at first
    screen = FakeResultScreen(fitted(3_072), pin=tokyo)
    reader = ResultReader(screen, FakeControls(screen), REGION, max_zoom_outs=2)
    reader.scale = 1.5  # as the guess map measured it

    reading = reader.read(*tokyo, crowded=True)  # zoomed out, Tokyo comes into view

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 50


class GoingResultScreen(FakeResultScreen):
    """A multiplayer result map that goes to the standings after ``notches`` of the wheel."""

    def __init__(self, *args, controls=None, notches=3, shows_after=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.controls, self.notches, self.shows_after = controls, notches, shows_after
        self.looks = 0

    def grab(self, region):
        self.looks += 1
        gone = self.controls.notches >= self.notches
        if gone or self.looks <= self.shows_after:
            return Image.new("RGB", (region.width, region.height), (40, 40, 50))
        return super().grab(region)


def going_reader(world=24_576, **kwargs):
    screen = GoingResultScreen(fitted(world), **kwargs)
    screen.controls = FakeControls(screen)
    reader = ResultReader(screen, screen.controls, REGION)
    reader.scale = 1.5  # as the guess map measured it
    reader._answer_on_view = lambda *args: None  # as when one view doesn't say enough
    return screen, reader


def test_in_a_room_one_view_is_enough_by_its_coasts():
    screen = GoingResultScreen(fitted(256 * 1.5 * 2**6), notches=99)
    screen.controls = FakeControls(screen)
    reader = ResultReader(screen, screen.controls, REGION)
    reader.scale = 1.5

    reading = reader.read(*MADRID, crowded=True)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 10
    assert screen.controls.notches == 0  # no need to zoom out at all


def test_coasts_tell_a_maps_zoom_only_if_one_level_fits_best():
    view = fitted(256 * 1.5 * 2**6)
    rgb = np.asarray(FakeResultScreen(view, show_flag=False).grab(REGION))
    x, y = view.to_pixel(*MADRID, REGION.width)
    pin = Marker(x, y, 1.0)

    assert map_by_coasts(rgb, pin, *MADRID, 1.5) == view.world_px
    assert map_by_coasts(rgb, pin, 35.68, 139.69, 1.5) is None  # we clicked on Tokyo
    all_land = np.full_like(rgb, LAND)
    assert map_by_coasts(all_land, pin, 48.0, 90.0, 1.5) is None  # Mongolia: land at any zoom


def test_tells_our_pin_by_its_label():
    ours, theirs = Marker(400, 300, 1.0), Marker(900, 500, 1.0)
    label = Word("You", Region(400 + 54 - 20, 300 - 22 - 10, 40, 20))

    other_word = Word("Madrid", Region(1, 1, 50, 20))
    assert our_labelled_pin([other_word, label], [theirs, ours], 1.5) == ours
    assert our_labelled_pin([label], [theirs], 1.5) is None  # it stands beside no pin
    assert our_labelled_pin([Word("Younde", Region(400, 300, 50, 20))], [ours], 1.5) is None


def test_in_a_room_a_players_name_across_the_flags_foot_doesnt_hide_it():
    rgb = np.full((500, 1200, 3), LAND, np.float32)
    paint(rgb, "flag", 600, 200, FLAG_BLACK)
    rgb[200 - 18 : 200 + 5, 560:680] = (255, 255, 255)  # a name's label, over the pole
    reader = ResultReader(None, None, REGION)
    reader.scale = 1.5
    pixels = rgb.astype(np.uint8)

    assert reader._markers(pixels)[1] is None
    flag = reader._markers(pixels, crowded=True)[1]
    assert flag is not None and abs(flag.x - 600) <= 2 and abs(flag.y - 200) <= 2


def test_the_map_on_its_way_to_the_result_isnt_the_first_view():
    """A pin alone may be the one we just placed: the view to read is the one with the flag."""
    screen = FakeResultScreen(fitted(256 * 1.5 * 2**6), show_flag=False)
    controls = FakeControls(screen)
    grabbed = {"looks": 0}
    plain = screen.grab

    def grab(region):
        grabbed["looks"] += 1
        screen.show_flag = grabbed["looks"] > 3
        return plain(region)

    screen.grab = grab
    reader = ResultReader(screen, controls, REGION)
    reader.scale = 1.5

    first = reader._first_view(None)

    assert reader._markers(first, crowded=True)[1] is not None and grabbed["looks"] > 3


def test_in_a_room_takes_all_the_views_it_can_before_working_any_out():
    screen, reader = going_reader(notches=99)
    locate, located_at = reader._locate, []

    def locating(rgb):
        located_at.append(screen.controls.notches)
        return locate(rgb)

    reader._locate = locating

    reading = reader.read(*MADRID, crowded=True)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 10
    assert screen.controls.notches == reader.max_zoom_outs or len(set(located_at)) == 1
    assert len(set(located_at)) == 1  # only once it had finished zooming


def test_in_a_room_the_map_going_ends_the_views_but_not_the_working_out():
    world = 3_072  # recognisable as it first shows
    screen, reader = going_reader(world=world, notches=1)

    reading = reader.read(*MADRID, crowded=True)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 20
    assert reading.screenshot is not None


def test_in_a_room_says_when_the_map_went_before_it_could_be_read():
    screen, reader = going_reader(notches=2)

    reading = reader.read(*MADRID, crowded=True)

    assert reading.answer is None and "went" in reading.problem
    assert reading.screenshot is not None and reading.zoomed_out


def test_in_a_room_waits_for_the_map_to_show():
    screen, reader = going_reader(notches=99, shows_after=5)

    reading = reader.read(*MADRID, crowded=True)

    assert reading.answer is not None, reading.problem


def test_in_a_room_says_when_the_map_never_shows():
    screen, reader = going_reader(notches=99, shows_after=10_000)

    reading = reader.read(*MADRID, crowded=True)

    assert reading.answer is None and "never showed" in reading.problem
    assert screen.controls.notches == 0


SPAIN = {
    "toledo": ((39.86, -4.02),),
    "valladolid": ((41.65, -4.72),),
    "porto": ((41.15, -8.61),),
    "sevilla": ((37.39, -5.98),),
    "salamanca": ((40.97, -5.66),),
    "zaragoza": ((41.65, -0.88),),
}


def named_on(projection, names=tuple(SPAIN), width=REGION.width):
    """The words a map at ``projection`` shows for these towns, each just right of its dot."""
    words = []
    for name in names:
        x, y = projection.to_pixel(*SPAIN[name][0], width)
        words.append(Word(name.title(), Region(round(x) + 8, round(y) - 10, 80, 20)))
    return words


def whole_zoom_view():
    """A view of Madrid and Lisbon at a whole zoom level, at the markers' 1.5 scale."""
    return fitted(256 * 1.5 * 2**7)


def reader_by_towns(monkeypatch, words, others=()):
    """A reader in a room whose map goes before the world can be recognised on it."""
    monkeypatch.setattr(result_module, "towns", lambda: SPAIN)
    screen = GoingResultScreen(whole_zoom_view(), notches=1, others=others)
    screen.controls = FakeControls(screen)
    reader = ResultReader(screen, screen.controls, REGION, read_words=lambda image: words)
    reader.scale = 1.5  # as the guess map measured it
    reader._locate = lambda rgb: None
    return reader


def test_towns_agree_on_where_a_map_is_and_its_zoom(monkeypatch):
    monkeypatch.setattr(result_module, "towns", lambda: SPAIN)
    view = whole_zoom_view()

    found = map_by_towns(named_on(view), 1.5)

    assert found.world_px == view.world_px
    assert abs(found.origin_x - view.origin_x) < 60 and abs(found.origin_y - view.origin_y) < 20


def test_towns_too_few_or_too_close_say_nothing(monkeypatch):
    monkeypatch.setattr(result_module, "towns", lambda: SPAIN)
    view = whole_zoom_view()

    assert map_by_towns(named_on(view, ("toledo", "valladolid")), 1.5) is None
    assert map_by_towns([], 1.5) is None
    close = [Word(w.text, Region(500 + i, 300 + i, 80, 20)) for i, w in enumerate(named_on(view))]
    assert map_by_towns(close, 1.5) is None


def test_in_a_room_the_towns_a_map_names_give_the_answer_without_the_world(monkeypatch):
    reader = reader_by_towns(monkeypatch, named_on(whole_zoom_view()))

    reading = reader.read(*MADRID, crowded=True)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 10


def test_towns_are_asked_before_the_world_is_recognised(monkeypatch):
    reader = reader_by_towns(monkeypatch, named_on(whole_zoom_view()))
    reader._locate = lambda rgb: pytest.fail("the towns had already said")

    assert reader.read(*MADRID, crowded=True).answer is not None


def test_our_pin_is_the_one_the_towns_put_where_we_clicked(monkeypatch):
    badajoz = (38.88, -6.97)  # another player's guess, between ours and the flag
    reader = reader_by_towns(monkeypatch, named_on(whole_zoom_view()), others=(badajoz,))

    reading = reader.read(*MADRID, crowded=True)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 10


def test_towns_that_dont_put_our_pin_where_we_clicked_are_not_believed(monkeypatch):
    reader = reader_by_towns(monkeypatch, named_on(whole_zoom_view()))

    assert reader.read(35.68, 139.69, crowded=True).answer is None  # we clicked on Tokyo


def test_with_our_pin_under_the_advert_the_towns_place_the_flag_roughly(monkeypatch):
    view = whole_zoom_view()
    reader = reader_by_towns(monkeypatch, named_on(view))
    # An advert over where we clicked, so that our pin can't show.
    x, y = view.to_pixel(*MADRID, REGION.width)
    reader._advert = lambda: (round(x) - 100, round(y) - 40, round(x) + 100, round(y) + 40)

    reading = reader.read(*MADRID, crowded=True)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *LISBON) < 60


def test_an_answer_the_towns_place_on_a_wide_map_isnt_to_learn_from():
    world = 256 * 1.5 * 2**4  # a town's name is many kilometres across, this far out
    view = MapProjection(world, 0.0, 0.0, True)
    flag = Marker(*view.to_pixel(*LISBON, REGION.width), 1.0)
    reader = ResultReader(None, None, REGION)
    reader.scale = 1.5
    x, y = view.to_pixel(*MADRID, REGION.width)
    reader._advert = lambda: (round(x) - 100, round(y) - 40, round(x) + 100, round(y) + 40)
    rgb = np.zeros((REGION.height, REGION.width, 3), np.uint8)

    answer = reader._answer_by_towns(rgb, view, flag, None, [], *MADRID, 0)

    assert answer is not None and answer.rough


def test_with_the_flag_over_our_pin_the_answer_is_where_we_clicked(monkeypatch):
    view = whole_zoom_view()
    near = (39.0, -9.0)
    monkeypatch.setattr(result_module, "towns", lambda: SPAIN)
    screen = FakeResultScreen(view, pin=near, flag=near)
    screen.controls = FakeControls(screen)
    reader = ResultReader(screen, screen.controls, REGION, read_words=lambda image: named_on(view))
    reader.scale = 1.5
    reader._pins = lambda rgb: []  # the flag hides it

    reading = reader.read(*near, crowded=True)

    assert reading.answer is not None and (reading.answer.lat, reading.answer.lon) == near


def test_in_a_room_a_flag_in_the_advert_is_no_flag():
    page = Region(0, 0, 1400, 1000)  # so the advert is from x = 335 to 1065, y = 764 to 864
    seen = ResultReader(None, None, Region(100, 200, 1200, 700))
    seen.scale = 1.5
    blanked = ResultReader(None, None, Region(100, 200, 1200, 700), page=page)
    blanked.scale = 1.0
    rgb = np.full((700, 1200, 3), LAND, np.float32)
    paint(rgb, "flag", 600 - 100, 830 - 200, FLAG_BLACK, scale=1.5)
    rgb = rgb.astype(np.uint8)

    assert seen._markers(rgb)[1] is not None
    assert blanked._shades(rgb)[1][830 - 200 - 5, 600 - 100].max() == 0
    assert blanked._shades(rgb)[1][100, 100] == seen._shades(rgb)[1][100, 100]


def test_stops_reading_once_the_result_map_goes():
    screen = FakeResultScreen(fitted(24_576))
    controls = FakeControls(screen)
    reading = ResultReader(screen, controls, REGION).read(*MADRID, abort=lambda: True)
    assert reading.answer is None and "went" in reading.problem and controls.notches == 0


def test_close_guess_is_measured_before_the_markers_overlap():
    near = (40.52, -3.52)  # about 19 km from Madrid
    screen = FakeResultScreen(fitted(786_432, MADRID, near), pin=near, flag=MADRID)

    reading = ResultReader(screen, FakeControls(screen), REGION).read(*near)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *MADRID) < 3


def test_no_answer_when_the_flag_never_shows():
    screen = FakeResultScreen(fitted(6_144), show_flag=False)
    reading = ResultReader(screen, FakeControls(screen), REGION, max_zoom_outs=2).read(*MADRID)
    assert reading.answer is None and "flag" in reading.problem


def test_no_answer_when_the_map_disagrees_with_our_pin():
    screen = FakeResultScreen(fitted(6_144))
    paris = (48.86, 2.35)  # we think we clicked here, but the pin is in Madrid
    reading = ResultReader(screen, FakeControls(screen), REGION, max_zoom_outs=2).read(*paris)
    assert reading.answer is None and "lined up" in reading.problem
    assert [shot.size for shot in reading.zoomed_out] == [(REGION.width, REGION.height)] * 2


def test_result_region_covers_the_view_and_the_expanded_map():
    layout = Layout(
        view=Region(14, 262, 1405, 513),
        map_hover=Point(1626, 869),
        map_region=Region(768, 450, 1039, 522),
        guess_button=Point(1626, 1150),
        continue_button=Point(987, 1141),
    )
    assert result_region(layout) == Region(14, 262, 1793, 710)


def test_close_guess_among_other_players_pins_is_measured_zoomed_in():
    # Which pin is ours only the recognised world can say, by when the flag hides under it.
    near, paris = (40.52, -3.52), (48.86, 2.35)
    screen = FakeResultScreen(fitted(786_432, MADRID, near), pin=near, flag=MADRID, others=(paris,))
    reader = ResultReader(screen, FakeControls(screen), REGION)
    reader.scale = 1.5  # as the guess map measured it

    reading = reader.read(*near, crowded=True)

    assert reading.answer is not None, reading.problem
    assert haversine_km(reading.answer.lat, reading.answer.lon, *MADRID) < 3
