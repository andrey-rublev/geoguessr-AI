"""Run a trained model on screenshots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..knowledge.countries import country_codes
from ..knowledge.evidence import DEFAULT_COVERAGE_STRENGTH, Evidence, cell_countries, cell_weights
from ..knowledge.facts import per_country
from .backbone import ImageEncoder, square_crops
from .detail import LOOKS, DetailExpert, tiles
from .geocells import DEFAULT_GAME_PRIOR_STRENGTH, DEFAULT_PRIOR_STRENGTH
from .head import Checkpoint, Ensemble
from .neighbours import (
    VOTED_COVERAGE_STRENGTH,
    VOTED_GAME_PRIOR_STRENGTH,
    Neighbours,
    load_neighbours,
)


@dataclass(frozen=True)
class Guess:
    lat: float
    lon: float
    expected_score: float
    """The model's own estimate of the points this guess earns in OpenGuessr (0-5000), before
    its streak and supporter bonuses (see :meth:`.GeoCells.expected_points`)."""
    top_cells: list[tuple[float, float, float]]
    """The five likeliest geocells as (lat, lon, probability)."""
    countries: list[tuple[str, float]] = field(default_factory=list)
    """The three likeliest countries as (ISO code, probability)."""
    drives_left: float = 0.0
    """Probability that traffic keeps left there."""


@torch.inference_mode()
def guess_from_embeddings(
    model: Ensemble | Checkpoint,
    embeddings: torch.Tensor | np.ndarray,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    game_prior_strength: float = DEFAULT_GAME_PRIOR_STRENGTH,
    evidence: Evidence | None = None,
    coverage_strength: float = DEFAULT_COVERAGE_STRENGTH,
    votes: np.ndarray | None = None,
) -> Guess:
    """Guess one location from the embeddings of every crop of every view of a place.

    ``evidence`` (clues like the writing on signs) and Street View coverage, at
    ``coverage_strength`` (0 = ignore it), reweigh the model's geocells before choosing, and so
    do ``votes``, a weight for each cell from the look-alike photos (see
    :meth:`.Neighbours.vote`).
    """
    if isinstance(model, Checkpoint):
        model = Ensemble([model])
    probs = model.belief(model.log_probs(embeddings), prior_strength, game_prior_strength)
    cells = model.cells
    probs = probs * cell_weights(cells.centroids, evidence, coverage_strength)
    if votes is not None:
        probs = probs * votes
    probs /= probs.sum()

    lat, lon, expected = cells.best_guess(probs)
    top = probs.argsort()[::-1][:5]
    top_cells = [
        (float(cells.centroids[i, 0]), float(cells.centroids[i, 1]), float(probs[i])) for i in top
    ]
    by_country = probs @ cell_countries(cells.centroids)
    codes = country_codes()
    countries = [(codes[i], float(by_country[i])) for i in by_country.argsort()[::-1][:3]]
    drives_left = float(by_country @ per_country(lambda c: c.drives_left))
    return Guess(lat, lon, expected, top_cells, countries, drives_left)


class GeoPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        game_prior_strength: float | None = None,
        coverage_strength: float | None = None,
        embeddings: Path | None = None,
    ) -> None:
        """``embeddings``: the folder of embedded photos and rounds (one subfolder per image
        model), whose look-alikes vote for the cell and place each guess within it. Left out,
        ``game_prior_strength`` and ``coverage_strength`` are as tuned with them, or without
        (see :data:`.neighbours.VOTED_GAME_PRIOR_STRENGTH`)."""
        self.prior_strength = prior_strength
        saved = datetime.fromtimestamp(Path(checkpoint_path).stat().st_mtime)
        self.source = f"{Path(checkpoint_path).name}, saved {saved:%Y-%m-%d %H:%M}"
        """Which model this is, to tell rounds played by different models apart."""
        self.model = Ensemble.load(checkpoint_path)
        self.encoder = ImageEncoder(self.model.backbone, device)
        self.model.to(self.encoder.device)
        self.detail: DetailExpert | None = None
        """What close-ups of the views say, once ``train-details`` has learned it."""
        detail = detail_path(Path(checkpoint_path))
        if detail.exists():
            self.detail = DetailExpert.load(detail)
            if self.detail.backbone != self.model.backbone:
                self.detail = None  # learned from another encoder's embeddings
        self._details: tuple[tuple[int, ...], np.ndarray] | None = None
        """The last views' close-ups' say, kept: walking on, the same first views come again."""
        self.neighbours: Neighbours | None = None
        """Photos and rounds to choose the cell by, and the guess within it (see
        :mod:`.neighbours`)."""
        if embeddings is not None:
            folder = Path(embeddings) / self.model.backbone.replace("/", "__")
            self.neighbours = load_neighbours(folder, self.model.backbone, self.model.cells)
        voted = self.neighbours is not None
        if game_prior_strength is None:
            game_prior_strength = (
                VOTED_GAME_PRIOR_STRENGTH if voted else DEFAULT_GAME_PRIOR_STRENGTH
            )
        if coverage_strength is None:
            coverage_strength = VOTED_COVERAGE_STRENGTH if voted else DEFAULT_COVERAGE_STRENGTH
        self.game_prior_strength = game_prior_strength
        self.coverage_strength = coverage_strength

    def predict(self, images: Sequence[Image.Image], evidence: Evidence | None = None) -> Guess:
        """Guess one location from several views of the same place."""
        if not images:
            raise ValueError("predict() needs at least one image")
        crops = [crop for image in images for crop in square_crops(image)]
        if self.detail is not None:
            evidence = self._with_details(images[:LOOKS], evidence)
        embedded = self.encoder.encode(crops)
        alike = embedded.numpy().mean(axis=0)
        votes = None
        if self.neighbours is not None:
            votes = self.neighbours.vote(alike, len(self.model.cells))
        guess = guess_from_embeddings(
            self.model,
            embedded,
            self.prior_strength,
            self.game_prior_strength,
            evidence,
            self.coverage_strength,
            votes,
        )
        if self.neighbours is None:
            return guess
        cell = int(self.model.cells.assign(guess.lat, guess.lon))
        place = self.neighbours.near(alike, cell)
        return guess if place is None else replace(guess, lat=place[0], lon=place[1])

    def _with_details(self, views: Sequence[Image.Image], evidence: Evidence | None) -> Evidence:
        """``evidence``, and what close-ups of ``views`` say of each country."""
        detail = self.detail
        key = tuple(id(view) for view in views)
        if self._details is None or self._details[0] != key:
            close = [tile for view in views for tile in tiles(view, detail.rows, detail.cols)]
            self._details = key, detail.likelihood(self.encoder.encode(close).numpy())
        likelihood = self._details[1]
        evidence = evidence or Evidence()
        countries = likelihood if evidence.countries is None else evidence.countries * likelihood
        codes = country_codes()
        favoured = ", ".join(codes[i] for i in np.argsort(-likelihood)[:3])
        notes = [*evidence.notes, f"close-ups favour {favoured}"]
        return Evidence(countries, evidence.latitude, notes)


def detail_path(checkpoint: Path) -> Path:
    """Where a model's detail expert is kept: beside it."""
    return checkpoint.with_name(f"{checkpoint.stem}-detail.npz")
