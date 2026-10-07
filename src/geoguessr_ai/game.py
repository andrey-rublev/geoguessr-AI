"""The bot: look around, ask the model where we are, drop the pin, read the answer, next round.

In a multiplayer room it plays each round the host starts instead, within the round's timer,
and waits on the result screen for the host to go on rather than pressing Continue.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .buttons import MIN_SIMILARITY, button_region, similarity
from .camera import Camera, default_camera, ground_view, measure
from .clock import GuessWatch, RoundClock, notice_region, timer_region
from .compass import Compass, CompassReader, compass_region
from .config import Layout, Point, Region
from .controls import StopRequested
from .files import write_safely
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
SHARPNESS_WIDTH, BLURRED = 968, 15.0
"""A view less sharp than this (see :func:`sharpness`) is still being drawn. Of 12,944 saved
views, the plainest real ones (hazy desert) measured 21 and up, and all 12 captured before
their tiles loaded under 12, like half of a Beijing round's, which sent the guess to Buenos
Aires. Black ones, even with "No Street View available" written on them, count as blank."""


STEP_SECONDS = {"view": 2.5, "read": 6.0, "walk": 22.0, "look_up": 8.0, "camera": 2.0}
"""In a multiplayer round, how long each optional step is planned to take until it has been
timed: one more view, reading the signs, walking on to look again, looking up for the sun and
measuring Street View's camera."""
GUESS_SECONDS = 9.0
"""How long placing the pin and pressing Guess is planned to take until it has been timed."""
SPARE_SECONDS = 4.0
"""Time kept in hand beyond the steps planned: the timer is read in whole seconds, and may be
cut short between readings."""
TIMER_DROP = 3.0
"""Seconds the timer must have lost beyond the time gone by to say it was cut short."""
LOOKING = {"view", "walk", "look_up", "camera"}
"""Steps that only look for more to go on, which stop once another player has guessed."""


class ContinueBlocked(Exception):
    """Something, like an advert, kept covering the Continue button."""


class RoundOver(Exception):
    """A multiplayer round ended before the bot guessed: its time ran out."""


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
    view_draw_timeout: float = 3.0
    """How long to wait, after each turn, for a view Street View is still drawing: black, or
    blurred until its tiles load. A view it never finishes isn't shown to the model."""
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
    walk_below: float = 1750.0
    """When the model expects fewer points than this, walk on along the road and look round again
    before guessing, as players do when a place gives nothing away (0 = never). On 128 held-out
    rounds that walked, the second look was worth +221 points (give or take 88): +232 where the
    model had expected 1,250 to 1,750 before walking, but +10 above 1,750."""
    walk_again_below: float = 1000.0
    """Still expecting fewer points than this after walking on, walk on once more. The least sure
    rounds gained the most from the first walk (+331 to +381 below 1,250)."""
    walk_steps: int = 5
    """Presses of the Up key when walking on: Street View moves about 10 metres each."""
    step_wait: float = 0.8
    record_answers: bool = True
    """Read the real location off each result screen and save it with the round."""
    party: bool = False
    """Play in a multiplayer room: each round the host starts, against its timer, waiting on
    the result screen for the host to go on rather than pressing Continue. ``rounds`` is then
    how many to play before stopping (0 = until stopped)."""
    round_time: float = 0.0
    """Multiplayer: seconds a round lasts, assumed when its timer can't be read (0 = no limit)."""
    party_wait: float = 0.5
    """Multiplayer: how often to look whether a round has begun or ended."""
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
    drawn: list[bool] = field(default_factory=list)
    """Whether Street View had finished drawing each view, rather than leaving it black or
    blurred."""
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

    def seen(self) -> list[Image.Image]:
        """The views to show the model: those Street View finished drawing, if any were."""
        drawn = [view for view, done in zip(self.views, self.drawn, strict=True) if done]
        return drawn or list(self.views)


def scene_region(layout: Layout) -> Region:
    """The view plus the road below it, down to just above the game's buttons and adverts."""
    view = layout.view
    buttons = [b.y for b in (layout.guess_button, layout.continue_button) if b is not None]
    bottom = min(buttons) - 80
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
        self.clock: RoundClock | None = None
        """In a multiplayer room, the round's timer."""
        self.watch: GuessWatch | None = None
        """In a multiplayer room, what watches for another player's guess."""
        self.others_guessed = False
        """Whether another player has guessed this round, as far as the bot has noticed."""
        self.took = dict(STEP_SECONDS, guess=GUESS_SECONDS)
        """How long each step of a multiplayer round has taken, lately."""
        if settings.party:
            self.clock = self._round_clock()
            self.watch = self._guess_watch()

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
            if s.party:
                self._play_party(run_dir)
            else:
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
        self.others_guessed = False
        if self.watch is not None:
            self.watch.begin_round()
        try:
            return self._play_round(number, run_dir)
        finally:
            if self.watch is not None:
                self.watch.end_round()

    def _play_round(self, number: int, run_dir: Path | None) -> Placement:
        s = self.settings
        self.controls.sleep(s.round_load_wait)
        self._start_clock()

        look = self.look_around()
        self._read_clock()  # it may have been cut short while looking
        read = self._time_for("read")
        if not read:
            print("  short of time: not reading the signs")
        with self._timing("read" if read else None):
            evidence, lines = self.notice(look, read=read)
        guess = self.predictor.predict(look.seen(), evidence)
        sure = [guess.expected_score]  # after each look, saved to judge walking by later
        for below in () if s.dry_run else (s.walk_below, s.walk_again_below):
            if guess.expected_score >= below:
                break
            self._read_clock()
            if not self._time_for("walk"):
                print(f"  unsure (~{guess.expected_score:,.0f} points), but short of time to walk")
                break
            again = " again" if len(sure) > 1 else ""
            print(f"  unsure (~{guess.expected_score:,.0f} points): walking on to look{again}")
            with self._timing("walk"):
                self._walk()
                look.extend(self.look_around())
                evidence, lines = self.notice(look, read=self._time_for("read"))
                guess = self.predictor.predict(look.seen(), evidence)
            sure.append(guess.expected_score)
        for note in evidence.notes:
            print(f"  clue: {note}")
        print(
            f"  guess {guess.lat:.3f}, {guess.lon:.3f}{_describe(guess)} "
            f"(model expects ~{guess.expected_score:,.0f} points)"
        )

        self._check_round()
        with self._timing("guess"):
            placement = self.guess_map.place(guess.lat, guess.lon)
            if placement.confirmed is False:
                print("  couldn't see the pin where it was clicked; guessing anyway")
            placed = (placement.lat, placement.lon)
            folder = None
            if run_dir is not None:
                folder = run_dir / f"round_{number:02d}"
                self._save_round(folder, look, lines, evidence, guess, placement, sure, self.camera)

            self.controls.sleep(0.4)
            self._check_round()  # Guess sits near Continue, which a finished round shows
            self.controls.click(self.layout.guess_button)
        if self.watch is not None:
            self.watch.end_round()  # nothing to hurry for now
        if not s.dry_run:
            if s.party:
                self.controls.move(self.layout.view.center)  # closes the map, to see it end
                if self.clock is not None and math.isfinite(spare := self.clock.left()):
                    print(f"  locked in with {spare:.0f} s to spare; waiting for the round to end")
                self._wait_for(lambda: not self._round_on())
            self.controls.sleep(s.result_wait)
            if self.reader is not None:
                if self.reader.scale is None:  # the result screen draws the same pin
                    self.reader.scale = self.guess_map.scale
                if s.party:  # every player's pin shows, and the host may go on at any time
                    reading = self.reader.read(*placed, crowded=True, abort=self._round_on)
                else:
                    reading = self.reader.read(*placed)
                self._report(reading, placed)
                if folder is not None:
                    self._save_reading(folder, reading)
            if not s.party:
                self._press_continue(folder)
        return placement

    def _play_party(self, run_dir: Path | None) -> None:
        """Play each round the host of a multiplayer room starts, until ``rounds`` have been
        played or the bot is stopped, waiting through result and standings screens."""
        s = self.settings
        print("Multiplayer: playing each round the host starts. It never presses Continue.")
        if self.watch is not None and hasattr(self.screen, "open_another"):
            self.watch.watch(self.screen.open_another)
        played = 0
        try:
            while not s.rounds or played < s.rounds:
                self._wait_for(self._round_on, "\nWaiting for the host to start a round...")
                played += 1
                try:
                    self.play_round(played, run_dir)
                except RoundOver:
                    print("  the round ended before the bot could guess")
                    self._wait_for(lambda: not self._round_on())
                if s.dry_run:
                    break
        finally:
            if self.watch is not None:
                self.watch.stop()

    def _round_on(self) -> bool:
        """Whether a multiplayer round is on: Street View's compass shows only then, not on the
        result, loading or standings screens."""
        return self.compass.read() is not None

    def _wait_for(self, condition: Callable[[], bool], message: str = "") -> None:
        """Wait until ``condition()`` holds twice running, since screens flicker as they change,
        saying ``message`` once if it doesn't at first."""
        running, said = 0, False
        while running < 2:
            if condition():
                running += 1
            else:
                running = 0
                if message and not said:
                    print(message)
                    said = True
            if running < 2:
                self.controls.sleep(self.settings.party_wait)

    def _check_round(self) -> None:
        """In a multiplayer room, give up on the round if it has ended, as when its time ran
        out, rather than click on whatever the result screen shows there."""
        if not self.settings.party or self.settings.dry_run:
            return
        for _ in range(3):
            if self._round_on():
                return
            self.controls.sleep(0.3)
        raise RoundOver

    def _round_clock(self) -> RoundClock:
        """The multiplayer round's timer, read with the sign reader's recogniser, if the layout
        says where it is and RapidOCR is there to read it."""
        read_text = None
        if self.layout.timer is None:
            print("No timer in the layout: run `geoguessr-ai calibrate --party` to time rounds")
        elif self.signs is not None:
            read_text = self.signs.read_line
        else:
            try:
                read_text = SignReader(models=["latin"]).read_line
            except ImportError:
                print("Can't read the round's timer: install RapidOCR (`pip install rapidocr`)")
        timer = self.layout.timer
        return RoundClock(
            self.screen,
            None if timer is None else timer_region(timer),
            read_text,
            self.controls.now,
            show=self._open_map,
            hide=self._close_map,
        )

    def _guess_watch(self) -> GuessWatch | None:
        """What watches the middle of Street View for the game saying another player has
        guessed, with a sign reader of its own on one thread, so as to keep out of the way."""
        try:
            reader = SignReader(models=["latin"], threads=1)
        except ImportError:
            print("Can't watch for other players' guesses: install RapidOCR")
            return None
        return GuessWatch(
            notice_region(self.layout.view),
            lambda image: reader.read_line(image, whole=False),
            self.controls.now,
        )

    def _notice_others(self) -> bool:
        """Whether another player has guessed this round: the first time it's noticed, cut the
        clock to what the game said is left and say so. Looks itself when nothing watches in
        the background."""
        watch = self.watch
        if watch is None or self.settings.dry_run:
            return False
        if not watch.watching and watch.guessed_at is None:
            watch.look(self.screen)
        if watch.guessed_at is None:
            return False
        if not self.others_guessed:
            self.others_guessed = True
            left = watch.guessed_at + watch.cut_to - self.controls.now()
            if self.clock is not None:
                self.clock.cut(watch.guessed_at + watch.cut_to)
                left = self.clock.left()
            print(f"  another player guessed, leaving {left:.0f} s: guessing now")
        return True

    def _open_map(self) -> None:
        self.controls.move(self.layout.map_hover)
        self.controls.move(self.layout.map_region.center)  # stay over it so it doesn't collapse
        self.controls.sleep(self.settings.map_expand_wait)

    def _close_map(self) -> None:
        self.controls.move(self.layout.view.center)
        self.controls.sleep(0.3)

    def _start_clock(self) -> None:
        """At the start of a multiplayer round, read how long it has."""
        if self.clock is None:
            return
        self.clock.forget()
        left = self._read_clock()
        if left is None and self.settings.round_time:
            left = self.settings.round_time - self.settings.round_load_wait
            self.clock.deadline = self.controls.now() + left
            print(f"  couldn't read the timer: assuming {left:.0f} s are left")
        elif left is not None:
            print(f"  {left:.0f} s on the clock")

    def _read_clock(self) -> float | None:
        """Read the multiplayer round's timer, saying if it was cut short since the last time,
        as when another player guesses in a duel."""
        if self.clock is None or not self.clock.readable:
            return None
        expected = self.clock.left()
        left = self.clock.read()
        if left is not None and math.isfinite(expected) and left < expected - TIMER_DROP:
            print(f"  the timer was cut to {left:.0f} s")
        return left

    def _time_for(self, *steps: str) -> bool:
        """Whether a multiplayer round's timer leaves time for these steps and then to guess.
        Once another player has guessed, there's no more looking round, only thinking."""
        if self.clock is None:
            return True
        if self._notice_others() and LOOKING & set(steps):
            return False
        need = sum(self.took[step] for step in steps) + self.took["guess"] + SPARE_SECONDS
        return self.clock.left() >= need

    @contextmanager
    def _timing(self, step: str | None) -> Iterator[None]:
        """Time a step of a multiplayer round, to plan the next ones by."""
        if self.clock is None or step is None:
            yield
            return
        start = self.controls.now()
        yield
        self.took[step] = (self.took[step] + self.controls.now() - start) / 2

    def look_around(self) -> Look:
        """See every direction: north, east, south and west by the compass, if it can be found."""
        s = self.settings
        self.controls.move(self.layout.view.center)  # also collapses the minimap if it was open
        self.controls.sleep(s.view_settle_wait)
        self._wait_for_street_view()
        compass = None if s.dry_run else self.compass.read()
        if compass is None:
            if s.party and not s.dry_run:  # no compass, no round: it must have ended
                self._check_round()
                compass = self.compass.read()
            if compass is None:
                return self._drag_around()
        look = Look()
        for turn in range(min(s.views, 4)):
            if turn and not self._time_for("view"):
                print(f"  short of time: looked {turn} way{'s' if turn > 1 else ''} round")
                break
            with self._timing("view"):
                heading = self._face(compass, 90.0 * turn)
                self._capture(look, 90.0 * turn if heading is None else heading)
        if not self.camera.measured and self.camera_tries < CAMERA_TRIES:
            if self._time_for("camera"):
                with self._timing("camera"):
                    self._measure_camera()
        if s.look_down and len(look.views) == 4:
            views, headings, _ = self._look_tilted(compass, -LOOK_DOWN_PITCH, self.scene)
            look.down_views += views
            look.down_headings += headings
        if s.look_up and self._time_for("look_up", "read") and self._sun_may_show(look):
            with self._timing("look_up"):
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
        before = np.asarray(self._grab(view).convert("L"))
        distance = round(view.width * CAMERA_DRAG)
        start = Point(view.left + (view.width + distance) // 2, view.top + view.height // 4)
        self.controls.drag(start, -distance, 0)
        self.controls.sleep(self.settings.view_settle_wait)
        after = np.asarray(self._grab(view).convert("L"))
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
            if self.clock is not None and self._notice_others():
                break
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
            if turn and self.clock is not None and self._notice_others():
                break
            if turn:
                heading = self._face(compass, 90.0 * turn, clockwise=True)
            views.append(self._grab(region))
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

    def notice(self, look: Look, read: bool = True) -> tuple[Evidence, list[TextLine]]:
        """Read the signs every way, and the road's name from above, and look for the sun:
        clues the model can't see. Without ``read``, only look for the sun, which is quick."""
        lines: list[TextLine] = []
        if self.signs is not None and read:
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
        """Save what the view shows, once Street View has drawn it: after a turn it can take a
        few seconds to replace its blur with the panorama's tiles."""
        waited = 0.0
        while True:
            scene = self._grab(self.scene)
            view = scene.crop((0, 0, self.layout.view.width, self.layout.view.height))
            drawn = _is_drawn(view)
            if drawn or waited >= self.settings.view_draw_timeout:
                break
            self.controls.sleep(0.5)
            waited += 0.5
        look.scenes.append(scene)
        look.views.append(view)
        look.headings.append(heading)
        look.drawn.append(drawn)

    def _grab(self, region: Region) -> Image.Image:
        """A screenshot of ``region`` with whatever the game draws over Street View, like a
        multiplayer room's chat box, painted over in the picture's middling colour: the chat
        is no sign to read, and the model never saw one in a photo."""
        image = self.screen.grab(region)
        overlaps = []
        for area in self.layout.covered:
            left, top = max(area.left, region.left) - region.left, max(area.top, region.top)
            right = min(area.left + area.width, region.left + region.width) - region.left
            bottom = min(area.top + area.height, region.top + region.height)
            if right > left and bottom > top:
                overlaps.append((left, top - region.top, right - 1, bottom - region.top - 1))
        if overlaps:
            fill = tuple(int(v) for v in np.median(np.asarray(image).reshape(-1, 3), axis=0))
            draw = ImageDraw.Draw(image)
            for box in overlaps:
                draw.rectangle(box, fill=fill)
        return image

    def _press_continue(self, folder: Path | None) -> None:
        """Click Continue, but only once it looks as it did at calibration, not covered."""
        if self.layout.continue_button is None:
            raise ContinueBlocked("this layout has no Continue button: it is for multiplayer")
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
        while _is_blank(self._grab(self.layout.view)):
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
        sure: Sequence[float] = (),
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
            "undrawn_views": [i for i, drawn in enumerate(look.drawn) if not drawn],
            "down_headings": look.down_headings,
            "up_headings": look.up_headings,
            "up_pitches": look.up_pitches,
            "camera": None if camera is None else asdict(camera),
            "walked": len(sure) > 1,
            "walks": max(len(sure) - 1, 0),
            "expected_each_look": list(sure),
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
    """Write a round's record so that the answer being added to it can't leave half a file
    behind if the machine goes down mid-write."""
    text = json.dumps(info, indent=2, ensure_ascii=False)
    write_safely(path, lambda file: file.write(text.encode("utf-8")))


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


def sharpness(image: Image.Image) -> float:
    """Variance of the Laplacian with the view shrunk to :data:`SHARPNESS_WIDTH` pixels wide."""
    grey = np.asarray(image.convert("L"))
    height = max(1, round(grey.shape[0] * SHARPNESS_WIDTH / grey.shape[1]))
    small = cv2.resize(grey, (SHARPNESS_WIDTH, height), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def _is_drawn(view: Image.Image) -> bool:
    return not _is_blank(view) and sharpness(view) >= BLURRED
