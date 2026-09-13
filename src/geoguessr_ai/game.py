"""The bot: look around, ask the model where we are, drop the pin, read the answer, next round."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .config import Layout, Point
from .controls import StopRequested
from .geo import haversine_km
from .mapcal import MapNotFound, MapProjection, locate_world
from .model.predictor import Guess
from .result import Reading, ResultReader, result_region


class Predictor(Protocol):
    def predict(self, images: Sequence[Image.Image]) -> Guess: ...


@dataclass
class BotSettings:
    rounds: int = 5
    views: int = 4
    """Screenshots per round; the camera is rotated between them."""
    drag_fraction: float = 0.6
    """How far to drag per rotation, as a fraction of the view width."""
    round_load_wait: float = 3.0
    view_settle_wait: float = 0.8
    map_expand_wait: float = 1.0
    result_wait: float = 3.5
    min_map_score: float = 0.6
    record_answers: bool = True
    """Read the real location off each result screen and save it with the round."""
    dry_run: bool = False
    debug_dir: Path | None = Path("runs")


class OpenGuessrBot:
    def __init__(
        self, layout: Layout, predictor: Predictor, settings: BotSettings, screen, controls
    ):
        self.layout = layout
        self.predictor = predictor
        self.settings = settings
        self.screen = screen
        self.controls = controls
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
            print("Dry run: one round, no clicks. The mouse still hovers and drags the view.")
        print("Stop any time with F8, or by slamming the mouse into a screen corner.")
        try:
            for number in range(1, rounds + 1):
                self.play_round(number, run_dir)
        except StopRequested:
            print("Stopped by user.")

    def play_round(self, number: int, run_dir: Path | None = None) -> Point:
        s = self.settings
        print(f"\nRound {number}")
        self.controls.sleep(s.round_load_wait)

        views = self.look_around()
        guess = self.predictor.predict(views)
        print(
            f"  guess {guess.lat:.3f}, {guess.lon:.3f} "
            f"(model expects ~{guess.expected_score:,.0f} points)"
        )

        map_image, projection = self.read_map()
        pin = self.place_pin(projection, guess.lat, guess.lon)
        region = self.layout.map_region
        placed = projection.to_latlon(pin.x - region.left, pin.y - region.top)
        folder = None
        if run_dir is not None:
            folder = run_dir / f"round_{number:02d}"
            self._save_round(folder, views, map_image, projection, guess, pin, placed)

        self.controls.sleep(0.4)
        self.controls.click(self.layout.guess_button)
        if not s.dry_run:
            self.controls.sleep(s.result_wait)
            if self.reader is not None:
                reading = self.reader.read(*placed)
                self._report(reading, placed)
                if folder is not None:
                    self._save_reading(folder, reading)
            self.controls.click(self.layout.continue_button)
        return pin

    def look_around(self) -> list[Image.Image]:
        """Capture several headings by dragging the panorama right-to-left between shots."""
        view = self.layout.view
        s = self.settings
        # Moving onto the panorama also collapses the minimap if it was open.
        self.controls.move(view.center)
        self.controls.sleep(s.view_settle_wait)
        views = [self.screen.grab(view)]
        distance = round(view.width * s.drag_fraction)
        start = Point(view.center.x + distance // 2, view.center.y)
        for _ in range(s.views - 1):
            self.controls.drag(start, -distance)
            self.controls.sleep(s.view_settle_wait)
            views.append(self.screen.grab(view))
        return views

    def read_map(self) -> tuple[Image.Image, MapProjection]:
        """Expand the minimap and work out its projection, zooming out once if needed."""
        region = self.layout.map_region
        self.controls.move(self.layout.map_hover)
        self.controls.move(region.center)  # stay over the expanded map so it doesn't collapse
        self.controls.sleep(self.settings.map_expand_wait)
        problem = ""
        for attempt in range(2):
            image = self.screen.grab(region)
            try:
                projection = locate_world(np.asarray(image))
                if projection.score >= self.settings.min_map_score:
                    return image, projection
                problem = f"match score {projection.score:.2f} is too low"
            except MapNotFound as exc:
                problem = str(exc)
            if attempt == 0:
                print(f"  couldn't read the map ({problem}); zooming out and retrying")
                self.controls.scroll(region.center, -6)
                self.controls.sleep(self.settings.map_expand_wait)
        raise MapNotFound(f"Couldn't read the map: {problem}. Check map_region in layout.json.")

    def place_pin(self, projection: MapProjection, lat: float, lon: float) -> Point:
        region = self.layout.map_region
        x, y = projection.to_pixel(lat, lon, region.width)
        cx = float(np.clip(x, 3, region.width - 4))
        cy = float(np.clip(y, 3, region.height - 4))
        if (cx, cy) != (x, y):
            print("  guess is outside the visible map; placing the pin at the nearest edge")
        pin = region.to_screen(cx, cy)
        self.controls.click(pin)
        return pin

    @staticmethod
    def _report(reading: Reading, placed: tuple[float, float]) -> None:
        if reading.answer is None:
            print(f"  couldn't read the answer: {reading.problem}")
            return
        answer = reading.answer
        km = haversine_km(*placed, answer.lat, answer.lon)
        print(f"  answer {answer.lat:.3f}, {answer.lon:.3f}: our pin was {km:,.0f} km away")

    def _save_round(
        self,
        folder: Path,
        views: Sequence[Image.Image],
        map_image: Image.Image,
        projection: MapProjection,
        guess: Guess,
        pin: Point,
        placed: tuple[float, float],
    ) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for i, view in enumerate(views):
            view.save(folder / f"view_{i}.jpg", quality=90)
        marked = np.array(map_image)
        local = (pin.x - self.layout.map_region.left, pin.y - self.layout.map_region.top)
        cv2.drawMarker(marked, local, (255, 0, 0), cv2.MARKER_TILTED_CROSS, 28, 3)
        Image.fromarray(marked).save(folder / "map.png")
        info = {
            "guess": asdict(guess),
            "projection": asdict(projection),
            "pin": asdict(pin),
            "placed": {"lat": placed[0], "lon": placed[1]},
        }
        (folder / "round.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

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
        path.write_text(json.dumps(info, indent=2), encoding="utf-8")
