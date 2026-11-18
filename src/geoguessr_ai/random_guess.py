"""Guesses anywhere on land, for collect mode: rounds saved to learn from, with no model to load
or think with. The answer is still read off each result screen, as far off as the pin may be: of
451 saved rounds guessed over 5,000 km away by November 2026, none was too rough to learn from."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from global_land_mask import globe
from PIL import Image

from .knowledge.evidence import Evidence
from .model.predictor import Guess

LATITUDES = (-56.0, 72.0)
"""Where the land with roads is, from Tierra del Fuego to the north of Norway."""


class RandomGuesser:
    """A predictor that picks a place on land at random, every part of the land as likely."""

    source = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def predict(
        self,
        images: Sequence[Image.Image],
        evidence: Evidence | None = None,
        quick: bool = False,
    ) -> Guess:
        low, high = (math.sin(math.radians(lat)) for lat in LATITUDES)
        while True:  # by area: as many points to a square km near the poles as the equator
            lat = math.degrees(math.asin(self.random.uniform(low, high)))
            lon = self.random.uniform(-180.0, 180.0)
            if globe.is_land(lat, lon):
                return Guess(lat, lon, 0.0, [])
