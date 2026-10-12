"""Screen layout of the OpenGuessr UI, found on screen by ``geoguessr-ai calibrate`` (or pointed
out by hand, with ``--manual``).

All coordinates are physical screen pixels (the process is made DPI-aware before
capturing or moving the mouse, so these match what mss and pyautogui see).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_LAYOUT_PATH = Path("layout.json")
PARTY_LAYOUT_PATH = Path("layout-party.json")
"""Multiplayer rooms lay the game out differently, so they are calibrated apart."""


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def offset(self, dx: float, dy: float) -> Point:
        return Point(round(self.x + dx), round(self.y + dy))


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    @classmethod
    def from_corners(cls, a: Point, b: Point) -> Region:
        left, right = sorted((a.x, b.x))
        top, bottom = sorted((a.y, b.y))
        if right - left < 20 or bottom - top < 20:
            raise ValueError(f"Region between {a} and {b} is too small; pick opposite corners.")
        return cls(left, top, right - left, bottom - top)

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> Point:
        return Point(self.left + self.width // 2, self.top + self.height // 2)

    def contains(self, p: Point) -> bool:
        return (
            self.left <= p.x < self.left + self.width and self.top <= p.y < self.top + self.height
        )

    def to_screen(self, x: float, y: float) -> Point:
        """Convert coordinates relative to this region into an absolute screen point."""
        return Point(round(self.left + x), round(self.top + y))


@dataclass(frozen=True)
class Layout:
    view: Region
    """Street View area to work in: to drag, click and look up. Clear of the minimap and
    buttons."""
    map_hover: Point
    """A point on the collapsed minimap; hovering it expands the map."""
    map_region: Region
    """The expanded map (just the map tiles, not the bar underneath)."""
    guess_button: Point
    continue_button: Point | None
    """The "Continue" button on the round result screen. Multiplayer layouts have none: there
    the host presses it, for everyone."""
    timer: Point | None = None
    """Multiplayer: the round's timer, in the bar under the expanded map."""
    covered: tuple[Region, ...] = ()
    """Multiplayer: what the game draws over Street View, like the chat box, which is blanked out
    of everything the bot looks at, so it is neither shown to the model nor read as a sign."""
    page: Region | None = None
    """Where the game's page shows in the browser, when calibration found the game by itself."""
    party: bool = False
    """Whether this is a multiplayer room's layout."""
    minimap: Region | None = None
    """The collapsed map, when calibration found the game by itself. The model is then shown
    Street View down to its bottom and across to its right, with it covered over."""

    @property
    def picture(self) -> Region:
        """What the model is shown: the view, and on down to the bottom of the collapsed map
        and across to its right side, when that is known. The map comes down to two thirds
        of the window in a big one, but only halfway in a small one, where the view above
        it would be mostly sky."""
        if self.minimap is None:
            return self.view
        right = max(self.view.right, self.minimap.right)
        bottom = max(self.view.bottom, self.minimap.bottom)
        return Region(self.view.left, self.view.top, right - self.view.left, bottom - self.view.top)

    @property
    def hidden(self) -> tuple[Region, ...]:
        """What covers Street View in the picture, to be painted over: the chat box and the
        collapsed map."""
        return self.covered if self.minimap is None else (*self.covered, self.minimap)

    def save(self, path: Path = DEFAULT_LAYOUT_PATH) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_LAYOUT_PATH) -> Layout:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `geoguessr-ai calibrate` first.")
        raw = json.loads(path.read_text(encoding="utf-8"))
        point = raw.get("continue_button")
        timer = raw.get("timer")
        page = raw.get("page")
        minimap = raw.get("minimap")
        return cls(
            view=Region(**raw["view"]),
            map_hover=Point(**raw["map_hover"]),
            map_region=Region(**raw["map_region"]),
            guess_button=Point(**raw["guess_button"]),
            continue_button=None if point is None else Point(**point),
            timer=None if timer is None else Point(**timer),
            covered=tuple(Region(**region) for region in raw.get("covered", ())),
            page=None if page is None else Region(**page),
            party=raw.get("party", point is None),  # older multiplayer layouts had no Continue
            minimap=None if minimap is None else Region(**minimap),
        )
