"""The bot: look around, ask the model where we are, drop the pin, read the answer, next round."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .buttons import MIN_SIMILARITY, button_region, similarity
from .camera import Camera, default_camera, ground_view, measure
from .compass import Compass, CompassReader, compass_region
from .config import Layout, Point, Region
from .controls import StopRequested
from .geo import haversine_km
from .knowledge.evidence import Evidence
from .knowledge.sun import clear_sky, find_sun, latitude_likelihood
from .knowledge.text import TextLine, text_clues
from .minimap import GuessMap, Placement
from .model.predictor import Guess
from .ocr import SignReader
from .result import Reading, ResultReader, result_region

PRESSES_PER_TURN = 3
"""Street View sometimes ignores a press on the compass; press again up to this many times."""
HEADING_TOLERANCE = 30.0
"""How far from north, east, south or west a view may face and still count as that way."""
LOOK_UP_PITCH = 45.0
"""Degrees to tilt the camera up to look for the sun: level views see up to about 30 degrees
above the horizon, so tilted this far they see from about 15 to 75."""
LOOK_DOWN_PITCH = 45.0
CLEAR_SKY = 0.15
"""Look up for the sun when at least this share of the top of some level view is blue sky. Of
15 live rounds, the three where looking up found the sun had 0.5 to 0.9, overcast ones 0.08
or less."""
TILT_DRAGS, TILT_TOLERANCE = 3, 5.0
"""Tilting takes at most this many drags, stopping within this many degrees."""
CAMERA_TRIES = 3
"""Times per game to try measuring Street View's camera (see :mod:`.camera`)."""
CAMERA_DRAG = 0.35
"""How far to drag the view sideways to measure the camera, as a fraction of its width."""
GROUND_BOX_SCORE = 0.5
"""Road names seen from above are found less surely than signs: a Paraguayan one scored 0.52
and 0.55, under the 0.6 that signs need."""


class ContinueBlocked(Exception):
    """Something, like an advert, kept covering the Continue button."""


class Predictor(Protocol):
    def predict(self, images: Sequence[Image.Image], evidence: Evidence | None = None) -> Guess: ...


@dataclass
class BotSettings:
    rounds: int = 5
    views: int = 4
    """Screenshots per round. With the compass they face north, east, south and west (so at
    most four); without it the camera is dragged round between them."""
    drag_fraction: float = 0.6
    """Without the compass, how far to drag per turn, as a fraction of the view width."""
    round_load_wait: float = 3.0
    view_settle_wait: float = 0.8
    view_load_timeout: float = 10.0
    """How long to wait for Street View while it is still a black screen."""
    turn_wait: float = 1.0
    """How long the view takes to turn after pressing the compass."""
    map_expand_wait: float = 1.0
    zoom_levels: int = 3
    """Wheel notches to zoom the minimap in by before clicking the guess."""
    result_wait: float = 3.5
    continue_timeout: float = 20.0
    """How long to wait for something covering the Continue button, like an advert, to go."""
    min_map_score: float = 0.6
    read_text: bool = True
    """Read signs and road names (needs RapidOCR)."""
    look_up: bool = True
    """When the sky is clear, tilt the camera up and look round for the sun, as players do: in a
    level view it seldom shows (never, in 150 saved rounds), since it usually stands higher.
    Stops once it finds it. Adds up to about 7 seconds to rounds with blue sky."""
    look_down: bool = False
    """After the level views, tilt the camera down and look round again, as players do to see the
    road's lines and the Google car, saving what it sees as down_*.jpg (about 6 seconds a round).
    Nothing uses these views yet: a yellow-line detector on 23 live rounds mistook cars, walls
    and grass for paint as often as it found lines, and road numbers written on the road are
    too distorted to read."""
    walk_below: float = 1500.0
    """When the model expects fewer points than this, walk on along the road and look round again
    before guessing, as players do when a place gives nothing away (0 = never). On 104 saved
    rounds it expected under 1,500 in about a third, which really scored 1,560 on average
    against 2,366 for the rest."""
    walk_steps: int = 5
    """Presses of the Up key when walking on: Street View moves about 10 metres each."""
    step_wait: float = 0.8
    record_answers: bool = True
    """Read the real location off each result screen and save it with the round."""
    dry_run: bool = False
    debug_dir: Path | None = Path("runs")


@dataclass
class Look:
    """Everything the bot saw in one round."""

    views: list[Image.Image] = field(default_factory=list)
    """What the model looks at."""
    scenes: list[Image.Image] = field(default_factory=list)
    """Each view with the road below it, where Street View writes road names, for reading."""
    headings: list[float | None] = field(default_factory=list)
    """Degrees clockwise from north that each view faces, when the compass was read."""
    down_views: list[Image.Image] = field(default_factory=list)
    """The road round the car, looking down, where its lines and names show best. Only read,
    not shown to the model, which learned from level photos."""
    down_headings: list[float | None] = field(default_factory=list)
    up_views: list[Image.Image] = field(default_factory=list)
    """The sky, looking up, where the sun shows. Only searched for the sun."""
    up_headings: list[float | None] = field(default_factory=list)
    up_pitches: list[float] = field(default_factory=list)
    """Degrees each up view is tilted above level, going by how far the view was dragged."""

    def extend(self, other: Look) -> None:
        """Add what was seen from another spot."""
        for name in (f.name for f in fields(self)):
            getattr(self, name).extend(getattr(other, name))


def scene_region(layout: Layout) -> Region:
    """The view plus the road below it, down to just above the game's buttons and adverts."""
    view = layout.view
    bottom = min(layout.guess_button.y, layout.continue_button.y) - 80
    return Region(view.left, view.top, view.width, max(bottom - view.top, view.height))


class OpenGuessrBot:
    def __init__(
        self,
        layout: Layout,
        predictor: Predictor,
        settings: BotSettings,
        screen,
        controls,
        continue_image: Image.Image | None = None,
    ):
        """``continue_image`` is calibration's snapshot of the Continue button. With it, the bot
        won't click Continue while something else covers the button."""
        self.layout = layout
        self.continue_image = continue_image
        self.predictor = predictor
        self.settings = settings
        self.screen = screen
        self.controls = controls
        self.scene = scene_region(layout)
        self.camera: Camera = default_camera(layout.view)
        self.camera_tries = 0
        self.compass = CompassReader(screen, compass_region(layout, screen.virtual_desktop()))
        self.guess_map = GuessMap(
            screen,
            controls,
            layout.map_region,
            layout.map_hover,
            zoom_levels=settings.zoom_levels,
            min_score=settings.min_map_score,
            expand_wait=settings.map_expand_wait,
            dry_run=settings.dry_run,
        )
        self.signs = None
        if settings.read_text:
            try:
                self.signs = SignReader()
            except ImportError:
                print("Not reading signs: install RapidOCR with `pip install rapidocr onnxruntime`")
        self.reader = None
        if settings.record_answers:
            self.reader = ResultReader(screen, controls, result_region(layout))

    def play(self) -> None:
        s = self.settings
        run_dir = None
        if s.debug_dir is not None:
            run_dir = Path(s.debug_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
        rounds = 1 if s.dry_run else s.rounds
        if s.dry_run:
            print("Dry run: one round, no clicks. The mouse still hovers, drags and zooms.")
        print("Stop any time with F8, or by slamming the mouse into a screen corner.")
        try:
            for number in range(1, rounds + 1):
                self.play_round(number, run_dir)
        except StopRequested:
            print("Stopped by user.")
        except self.controls.failsafe:
            print("Stopped: the mouse hit a screen corner.")
        except ContinueBlocked as blocked:
            print(f"Stopped: {blocked}")

    def play_round(self, number: int, run_dir: Path | None = None) -> Placement:
        s = self.settings
        print(f"\nRound {number}")
        self.controls.sleep(s.round_load_wait)

        look = self.look_around()
        evidence, lines = self.notice(look)
        guess = self.predictor.predict(look.views, evidence)
        walked = not s.dry_run and guess.expected_score < s.walk_below
        if walked:
            print(f"  unsure (~{guess.expected_score:,.0f} points): walking on to look again")
            self._walk()
            look.extend(self.look_around())
            evidence, lines = self.notice(look)
            guess = self.predictor.predict(look.views, evidence)
        for note in evidence.notes:
            print(f"  clue: {note}")
        print(
            f"  guess {guess.lat:.3f}, {guess.lon:.3f}{_describe(guess)} "
            f"(model expects ~{guess.expected_score:,.0f} points)"
        )

        placement = self.guess_map.place(guess.lat, guess.lon)
        if placement.confirmed is False:
            print("  couldn't see the pin where it was clicked; guessing anyway")
        placed = (placement.lat, placement.lon)
        folder = None
        if run_dir is not None:
            folder = run_dir / f"round_{number:02d}"
            self._save_round(folder, look, lines, evidence, guess, placement, walked, self.camera)

        self.controls.sleep(0.4)
        self.controls.click(self.layout.guess_button)
        if not s.dry_run:
            self.controls.sleep(s.result_wait)
            if self.reader is not None:
                if self.reader.scale is None:  # the result screen draws the same pin
                    self.reader.scale = self.guess_map.scale
                reading = self.reader.read(*placed)
                self._report(reading, placed)
                if folder is not None:
                    self._save_reading(folder, reading)
            self._press_continue(folder)
        return placement

    def look_around(self) -> Look:
        """See every direction: north, east, south and west by the compass, if it can be found."""
        s = self.settings
        self.controls.move(self.layout.view.center)  # also collapses the minimap if it was open
        self.controls.sleep(s.view_settle_wait)
        self._wait_for_street_view()
        compass = None if s.dry_run else self.compass.read()
        if compass is None:
            return self._drag_around()
        look = Look()
        for turn in range(min(s.views, 4)):
            heading = self._face(compass, 90.0 * turn)
            self._capture(look, 90.0 * turn if heading is None else heading)
        if not self.camera.measured and self.camera_tries < CAMERA_TRIES:
            self._measure_camera()
        if s.look_down and len(look.views) == 4:
            views, headings, _ = self._look_tilted(compass, -LOOK_DOWN_PITCH, self.scene)
            look.down_views += views
            look.down_headings += headings
        if s.look_up and self._sun_may_show(look):
            views, headings, pitch = self._look_tilted(
                compass, LOOK_UP_PITCH, self.layout.view, _shows_sun
            )
            look.up_views += views
            look.up_headings += headings
            look.up_pitches += [pitch] * len(views)
        return look

    def _sun_may_show(self, look: Look) -> bool:
        """Whether looking up could find the sun: some blue sky, and no sun in the level views."""
        views = [np.asarray(view.convert("RGB")) for view in look.views]
        if not views or max(clear_sky(rgb) for rgb in views) < CLEAR_SKY:
            return False
        return all(find_sun(rgb, sky_share=0.6) is None for rgb in views)

    def _measure_camera(self) -> None:
        """Drag the view sideways and measure Street View's camera from how details moved,
        which says which way each pixel looks (see :mod:`.camera`)."""
        self.camera_tries += 1
        view = self.layout.view
        before = np.asarray(self.screen.grab(view).convert("L"))
        distance = round(view.width * CAMERA_DRAG)
        start = Point(view.left + (view.width + distance) // 2, view.top + view.height // 4)
        self.controls.drag(start, -distance, 0)
        self.controls.sleep(self.settings.view_settle_wait)
        after = np.asarray(self.screen.grab(view).convert("L"))
        fit = measure(before, after, view, self.camera)
        if fit is None:
            print("  couldn't measure Street View's camera from this view; will try again")
            return
        self.camera = fit[0]
        print(
            f"  measured Street View's camera: it faces ({self.camera.cx:.0f}, "
            f"{self.camera.cy:.0f}) on screen, {2 * self.camera.f:.0f} pixels per 90 degrees"
        )

    def _walk(self) -> None:
        """Walk on along the road: click the sky, which gives Street View the keyboard without
        moving (pressing the compass keeps it), then press Up."""
        view = self.layout.view
        self.controls.click(Point(view.center.x, view.top + round(view.height * 0.05)))
        for _ in range(self.settings.walk_steps):
            self.controls.press("up")
            self.controls.sleep(self.settings.step_wait)

    def _look_tilted(self, compass: Compass, pitch: float, region: Region, enough=None):
        """Face north, tilt the camera up by ``pitch`` degrees (down if negative) and look round
        north, east, south and west, since the compass's arrow keeps the tilt. Stops early once
        ``enough(view)`` says so. Pressing the compass afterwards levels the camera again.
        Returns the views of ``region``, their headings, and the tilt they were seen at."""
        views: list[Image.Image] = []
        headings: list[float | None] = []
        heading = self._face(compass, 0.0)
        tilted = self._tilt(pitch)
        for turn in range(4):
            if turn:
                heading = self._face(compass, 90.0 * turn, clockwise=True)
            views.append(self.screen.grab(region))
            headings.append(90.0 * turn if heading is None else heading)
            if enough is not None and enough(views[-1]):
                break
        self.controls.click(compass.center)
        self.controls.sleep(self.settings.turn_wait)
        return views, headings, tilted

    def _tilt(self, degrees: float) -> float:
        """Drag the picture straight down (to look up) or up (to look down) until the camera
        tilts about ``degrees``, going by :attr:`camera`. Drags stay inside the view: looking up
        they start in the sky, and looking down away from the middle, where Street View's arrows
        for moving along the road are. Returns how far it should have tilted."""
        view, camera = self.layout.view, self.camera
        margin = round(view.height * 0.08)
        top, bottom = view.top + margin, view.top + view.height - margin
        if degrees > 0:
            x = round(min(max(camera.cx, view.left + margin), view.left + view.width - margin))
        else:
            x = view.left + round(view.width * 0.15)
        tilted = 0.0
        for _ in range(TILT_DRAGS):
            left = degrees - tilted
            if abs(left) < TILT_TOLERANCE:
                break
            start = top if left > 0 else bottom
            end = round(min(max(camera.row_for_tilt(start, left), top), bottom))
            self.controls.drag(Point(x, start), 0, end - start)
            tilted += camera.tilt(start, end)
        self.controls.sleep(self.settings.view_settle_wait)
        return tilted

    def _face(self, compass: Compass, target: float, clockwise: bool | None = None) -> float | None:
        """Press the compass until the view faces ``target``, a quarter turn on from the last
        view: pressing the compass itself faces north and levels the camera, its arrow turns to
        the next quarter and keeps any tilt. ``clockwise`` uses the arrow even to face north.
        Returns the heading the compass shows, or None if it can't be read."""
        if clockwise is None:
            clockwise = target != 0
        button = compass.clockwise if clockwise else compass.center
        heading = None
        for _ in range(PRESSES_PER_TURN):
            self.controls.click(button)
            self.controls.sleep(self.settings.turn_wait)
            reading = self.compass.read()
            if reading is None:
                return None
            heading = reading.heading
            still_to_turn = (target - heading) % 360
            if min(still_to_turn, 360 - still_to_turn) <= HEADING_TOLERANCE:
                break
            if clockwise and still_to_turn > 180:  # already past it: the arrow would go further
                break
        return heading

    def notice(self, look: Look) -> tuple[Evidence, list[TextLine]]:
        """Read the signs every way, and the road's name from above, and look for the sun:
        clues the model can't see."""
        lines: list[TextLine] = []
        if self.signs is not None:
            lines = self.signs.read(look.scenes)
            lines += self.signs.read(self._ground_views(look), min_box_score=GROUND_BOX_SCORE)
        clues = text_clues(lines)
        evidence = Evidence(countries=clues.likelihood, notes=clues.notes)
        level = [
            (image, heading, 0.0) for image, heading in zip(look.views, look.headings, strict=True)
        ]
        up = list(zip(look.up_views, look.up_headings, look.up_pitches, strict=True))
        view = self.layout.view
        for image, heading, pitch in level + up:
            if heading is None:
                continue
            spot = find_sun(np.asarray(image.convert("RGB")), sky_share=1.0 if pitch else 0.6)
            if spot is not None:
                azimuth, height = self.camera.direction(
                    view.left + spot[0], view.top + spot[1], heading, pitch
                )
                evidence.latitude = latitude_likelihood(azimuth, height)
                evidence.notes.append(
                    f"sun to the {_compass_point(azimuth)} ({azimuth:.0f} deg), {height:.0f} deg up"
                )
                break
        return evidence, lines

    def _ground_views(self, look: Look) -> list[Image.Image]:
        """The road round the car from above in each level view, where Street View's road names
        come out straight for reading (see :func:`.camera.ground_view`). Only once the camera
        has been measured, since a guessed one bends them."""
        if not self.camera.measured:
            return []
        corner = Point(self.scene.left, self.scene.top)
        grounds = (ground_view(scene, corner, self.camera) for scene in look.scenes)
        return [ground for ground in grounds if ground is not None]

    def _drag_around(self) -> Look:
        """Without the compass: drag the panorama right-to-left between shots."""
        view, s = self.layout.view, self.settings
        look = Look()
        self._capture(look, None)
        distance = round(view.width * s.drag_fraction)
        start = view.center.offset(distance // 2, 0)
        for _ in range(s.views - 1):
            self.controls.drag(start, -distance)
            self.controls.sleep(s.view_settle_wait)
            self._capture(look, None)
        return look

    def _capture(self, look: Look, heading: float | None) -> None:
        scene = self.screen.grab(self.scene)
        look.scenes.append(scene)
        look.views.append(scene.crop((0, 0, self.layout.view.width, self.layout.view.height)))
        look.headings.append(heading)

    def _press_continue(self, folder: Path | None) -> None:
        """Click Continue, but only once it looks as it did at calibration, not covered."""
        if self.continue_image is None:
            self.controls.click(self.layout.continue_button)
            return
        region, waited = button_region(self.layout.continue_button), 0.0
        while True:
            patch = self.screen.grab(region)
            if similarity(patch, self.continue_image) >= MIN_SIMILARITY:
                self.controls.click(self.layout.continue_button)
                return
            if waited >= self.settings.continue_timeout:
                break
            if not waited:
                print("  something covers the Continue button, maybe an advert; waiting")
            self.controls.sleep(1.0)
            waited += 1.0
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            patch.save(folder / "continue_blocked.png")
        raise ContinueBlocked(
            "the Continue button stayed covered, maybe by an advert, so it wasn't clicked. "
            "Close whatever covers it and play again; if nothing does, re-run calibrate."
        )

    def _wait_for_street_view(self) -> None:
        """Wait while Street View is still a black screen."""
        waited = 0.0
        while _is_blank(self.screen.grab(self.layout.view)):
            if waited >= self.settings.view_load_timeout:
                print("  Street View still looks blank; guessing anyway")
                return
            self.controls.sleep(0.5)
            waited += 0.5

    @staticmethod
    def _report(reading: Reading, placed: tuple[float, float]) -> None:
        if reading.answer is None:
            print(f"  couldn't read the answer: {reading.problem}")
            return
        answer = reading.answer
        km = haversine_km(*placed, answer.lat, answer.lon)
        print(f"  answer {answer.lat:.3f}, {answer.lon:.3f}: our pin was {km:,.0f} km away")

    @staticmethod
    def _save_round(
        folder: Path,
        look: Look,
        lines: Sequence[TextLine],
        evidence: Evidence,
        guess: Guess,
        placement: Placement,
        walked: bool = False,
        camera: Camera | None = None,
    ) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for i, view in enumerate(look.views):
            view.save(folder / f"view_{i}.jpg", quality=90)
        for i, view in enumerate(look.down_views):
            view.save(folder / f"down_{i}.jpg", quality=90)
        for i, view in enumerate(look.up_views):
            view.save(folder / f"up_{i}.jpg", quality=90)
        marked = np.array(placement.image.convert("RGB"))
        x, y = (round(v) for v in placement.map_xy)
        cv2.drawMarker(marked, (x, y), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 28, 3)
        Image.fromarray(marked).save(folder / "map.png")
        info = {
            "guess": asdict(guess),
            "headings": look.headings,
            "down_headings": look.down_headings,
            "up_headings": look.up_headings,
            "up_pitches": look.up_pitches,
            "camera": None if camera is None else asdict(camera),
            "walked": walked,
            "text": [asdict(line) for line in lines],
            "clues": evidence.notes,
            "projection": asdict(placement.projection),
            "pin": asdict(placement.click),
            "placed": {"lat": placement.lat, "lon": placement.lon},
            "pin_seen": placement.confirmed,
        }
        _write_round(folder / "round.json", info)

    @staticmethod
    def _save_reading(folder: Path, reading: Reading) -> None:
        """Add the real location (or why it couldn't be read) to the saved round."""
        if reading.screenshot is not None:
            reading.screenshot.save(folder / "result.png")
        path = folder / "round.json"
        info = json.loads(path.read_text(encoding="utf-8"))
        info["answer"] = None if reading.answer is None else asdict(reading.answer)
        if reading.problem:
            info["answer_problem"] = reading.problem
        _write_round(path, info)


def _write_round(path: Path, info: dict) -> None:
    """Write a round's record through a temporary file, so that the answer being added to it
    can't leave half a file behind if the machine goes down mid-write."""
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _describe(guess: Guess) -> str:
    """Where the model thinks it is, by country, and which side cars drive on there."""
    if not guess.countries:
        return ""
    places = ", ".join(f"{code} {share:.0%}" for code, share in guess.countries)
    side = "left" if guess.drives_left >= 0.5 else "right"
    sure = max(guess.drives_left, 1 - guess.drives_left)
    return f" ({places}; driving on the {side} {sure:.0%})"


def _shows_sun(view: Image.Image) -> bool:
    return find_sun(np.asarray(view.convert("RGB"))) is not None


def _compass_point(degrees: float) -> str:
    points = ("north", "north-east", "east", "south-east", "south", "south-west", "west")
    return (*points, "north-west")[round(degrees / 45) % 8]


def _is_blank(image: Image.Image) -> bool:
    """A view with almost no contrast: Street View hasn't drawn the panorama yet."""
    return float(np.asarray(image.convert("L")).std()) < 8.0
