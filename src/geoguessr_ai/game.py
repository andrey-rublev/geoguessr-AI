"""The bot: look around, ask the model where we are, drop the pin, read the answer, next round."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .buttons import MIN_SIMILARITY, button_region, similarity
from .compass import Compass, CompassReader, compass_region
from .config import Layout, Region
from .controls import StopRequested
from .geo import haversine_km
from .knowledge.evidence import Evidence
from .knowledge.sun import find_sun, latitude_likelihood, sun_azimuth
from .knowledge.text import TextLine, text_clues
from .minimap import GuessMap, Placement
from .model.predictor import Guess
from .ocr import SignReader
from .result import Reading, ResultReader, result_region

PRESSES_PER_TURN = 3
"""Street View sometimes ignores a press on the compass; press again up to this many times."""
HEADING_TOLERANCE = 30.0
"""How far from north, east, south or west a view may face and still count as that way."""
LOOK_DOWN_DRAGS = 2
"""Drags that tilt the camera down to look at the road (see ``BotSettings.look_down_drag``)."""


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
    look_down: bool = True
    """After the level views, tilt the camera down and look round again, as players do to see
    the road's lines and the names Street View writes on it. Adds about 6 seconds a round."""
    look_down_drag: float = 0.6
    """How far each drag that tilts the camera moves, as a fraction of the view's height. Street
    View tilts about 74 degrees per browser-window height, so where the view is half the window,
    two of these tilt it about 45 degrees down."""
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
        except ContinueBlocked as blocked:
            print(f"Stopped: {blocked}")

    def play_round(self, number: int, run_dir: Path | None = None) -> Placement:
        s = self.settings
        print(f"\nRound {number}")
        self.controls.sleep(s.round_load_wait)

        look = self.look_around()
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
            self._save_round(folder, look, lines, evidence, guess, placement)

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
        if s.look_down and len(look.views) == 4:
            self._look_down(compass, look)
        return look

    def _look_down(self, compass: Compass, look: Look) -> None:
        """Tilt the camera down and look round again, from west (where the level views ended)
        through north, east and south. Pressing the compass afterwards levels the camera."""
        s, view = self.settings, self.layout.view
        distance = round(view.height * s.look_down_drag)
        for _ in range(LOOK_DOWN_DRAGS):
            self.controls.drag(view.center.offset(0, distance // 2), 0, -distance)
        self.controls.sleep(s.view_settle_wait)
        for turn in range(4):
            target = (270.0 + 90.0 * turn) % 360
            if turn:
                heading = self._face(compass, target, clockwise=True)
            else:
                reading = self.compass.read()
                heading = None if reading is None else reading.heading
            look.down_views.append(self.screen.grab(self.scene))
            look.down_headings.append(target if heading is None else heading)
        self.controls.click(compass.center)
        self.controls.sleep(s.turn_wait)

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
        """Read the signs every way and look for the sun: clues the model can't see."""
        scenes = [*look.scenes, *look.down_views]
        lines: list[TextLine] = [] if self.signs is None else self.signs.read(scenes)
        clues = text_clues(lines)
        evidence = Evidence(countries=clues.likelihood, notes=clues.notes)
        for view, heading in zip(look.views, look.headings, strict=True):
            spot = None if heading is None else find_sun(np.asarray(view.convert("RGB")))
            if spot is not None:
                azimuth = sun_azimuth(spot[0], view.width, heading)
                evidence.latitude = latitude_likelihood(azimuth)
                evidence.notes.append(f"sun to the {_compass_point(azimuth)} ({azimuth:.0f} deg)")
                break
        return evidence, lines

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
    ) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for i, view in enumerate(look.views):
            view.save(folder / f"view_{i}.jpg", quality=90)
        for i, view in enumerate(look.down_views):
            view.save(folder / f"down_{i}.jpg", quality=90)
        marked = np.array(placement.image.convert("RGB"))
        x, y = (round(v) for v in placement.map_xy)
        cv2.drawMarker(marked, (x, y), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 28, 3)
        Image.fromarray(marked).save(folder / "map.png")
        info = {
            "guess": asdict(guess),
            "headings": look.headings,
            "down_headings": look.down_headings,
            "text": [asdict(line) for line in lines],
            "clues": evidence.notes,
            "projection": asdict(placement.projection),
            "pin": asdict(placement.click),
            "placed": {"lat": placement.lat, "lon": placement.lon},
            "pin_seen": placement.confirmed,
        }
        (folder / "round.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

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
        path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def _describe(guess: Guess) -> str:
    """Where the model thinks it is, by country, and which side cars drive on there."""
    if not guess.countries:
        return ""
    places = ", ".join(f"{code} {share:.0%}" for code, share in guess.countries)
    side = "left" if guess.drives_left >= 0.5 else "right"
    sure = max(guess.drives_left, 1 - guess.drives_left)
    return f" ({places}; driving on the {side} {sure:.0%})"


def _compass_point(degrees: float) -> str:
    points = ("north", "north-east", "east", "south-east", "south", "south-west", "west")
    return (*points, "north-west")[round(degrees / 45) % 8]


def _is_blank(image: Image.Image) -> bool:
    """A view with almost no contrast: Street View hasn't drawn the panorama yet."""
    return float(np.asarray(image.convert("L")).std()) < 8.0
