"""A detail expert: the small things players tell countries by, seen up close.

The model sees each view whole, shrunk to a few hundred pixels, where a bollard, a pole's
fittings, the colour of the road's lines, a kerb or a sign's shape are a few pixels each. This
looks at each view in close-up tiles instead, and a classifier learned from the bot's own rounds
says which countries such details belong to. Its say is a likelihood per country, which reweighs
the model's regions as the writing on signs does, at a weight tuned on rounds it never saw.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..knowledge.countries import country_codes

ROWS, COLS = 2, 4
"""Each view is cut into this many rows and columns of tiles: about twice as close as the model
looks, near enough to see a bollard's reflector and the colour of a road's lines."""
LOOKS = 4
"""Only the first look round (north, east, south, west) is cut up, which keeps it quick."""


def tiles(view: Image.Image, rows: int = ROWS, cols: int = COLS) -> list[Image.Image]:
    """``view`` cut into ``rows`` by ``cols`` close-up tiles, row by row."""
    w, h = view.size
    return [
        view.crop((c * w // cols, r * h // rows, (c + 1) * w // cols, (r + 1) * h // rows))
        for r in range(rows)
        for c in range(cols)
    ]


@dataclass
class DetailExpert:
    coef: np.ndarray
    """(classes, embedding dim) weights of the tile classifier."""
    intercept: np.ndarray
    classes: np.ndarray
    """The country index of each class (see :func:`.countries.country_codes`)."""
    log_prior: np.ndarray
    """Log share of training tiles from each country, divided out of what each tile says."""
    weight: float
    """How much its say counts against the model's."""
    backbone: str
    rows: int = ROWS
    cols: int = COLS

    def log_probs(self, embeddings: np.ndarray) -> np.ndarray:
        """(tiles, countries) log-probability of each country, by each tile. Of a country it
        never saw a tile from, it can't say, so a tile leaves it as likely as it was before."""
        logits = np.asarray(embeddings, dtype=np.float64) @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        out = np.tile(self.log_prior, (len(logits), 1))
        out[:, self.classes] = logp
        return out

    def likelihood(self, embeddings: np.ndarray, weight: float | None = None) -> np.ndarray:
        """How much the tiles favour each country (at most 1): the average over tiles of how much
        more likely each makes it than it was before, raised to :attr:`weight`."""
        ratio = (self.log_probs(embeddings) - self.log_prior).mean(axis=0)
        weight = self.weight if weight is None else weight
        return np.exp(np.clip(weight * (ratio - ratio.max()), -50.0, 0.0))

    def save(self, path: Path) -> None:
        np.savez(
            path,
            coef=self.coef,
            intercept=self.intercept,
            classes=self.classes,
            log_prior=self.log_prior,
            weight=self.weight,
            backbone=self.backbone,
            rows=self.rows,
            cols=self.cols,
        )

    @classmethod
    def load(cls, path: Path) -> DetailExpert:
        with np.load(path) as data:
            return cls(
                data["coef"],
                data["intercept"],
                data["classes"],
                data["log_prior"],
                float(data["weight"]),
                str(data["backbone"]),
                int(data["rows"]),
                int(data["cols"]),
            )


def fit(
    embeddings: np.ndarray, countries: Sequence[int], backbone: str, c: float = 1.0
) -> DetailExpert:
    """A detail expert learned from tile ``embeddings`` and the index of each one's country. Its
    weight is left at 1, to be tuned on held-out rounds."""
    from sklearn.linear_model import LogisticRegression

    countries = np.asarray(countries)
    classifier = LogisticRegression(C=c, max_iter=400).fit(embeddings, countries)
    coef, intercept = classifier.coef_, classifier.intercept_
    if len(coef) == 1:  # two countries: one set of weights, for the second against the first
        coef, intercept = np.vstack([np.zeros_like(coef), coef]), np.r_[0.0, intercept]
    counts = np.bincount(countries, minlength=len(country_codes())) + 1.0
    return DetailExpert(
        coef,
        intercept,
        classifier.classes_,
        np.log(counts / counts.sum()),
        1.0,
        backbone,
    )
