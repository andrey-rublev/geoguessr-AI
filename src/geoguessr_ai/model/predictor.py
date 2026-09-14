"""Run a trained checkpoint on screenshots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..knowledge.countries import country_codes
from ..knowledge.evidence import DEFAULT_COVERAGE_STRENGTH, Evidence, cell_countries, cell_weights
from ..knowledge.facts import per_country
from .backbone import ImageEncoder, square_crops
from .geocells import DEFAULT_GAME_PRIOR_STRENGTH, DEFAULT_PRIOR_STRENGTH, debias
from .head import Checkpoint


@dataclass(frozen=True)
class Guess:
    lat: float
    lon: float
    expected_score: float
    """The model's own estimate of the points this guess earns (0-5000)."""
    top_cells: list[tuple[float, float, float]]
    """The five likeliest geocells as (lat, lon, probability)."""
    countries: list[tuple[str, float]] = field(default_factory=list)
    """The three likeliest countries as (ISO code, probability)."""
    drives_left: float = 0.0
    """Probability that traffic keeps left there."""


@torch.inference_mode()
def guess_from_embeddings(
    checkpoint: Checkpoint,
    embeddings: torch.Tensor | np.ndarray,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    game_prior_strength: float = DEFAULT_GAME_PRIOR_STRENGTH,
    evidence: Evidence | None = None,
    coverage_strength: float = DEFAULT_COVERAGE_STRENGTH,
) -> Guess:
    """Guess one location from the embeddings of every crop of every view of a place.

    ``evidence`` (clues like the writing on signs) and Street View coverage, at
    ``coverage_strength`` (0 = ignore it), reweigh the model's geocells before choosing.
    """
    head = checkpoint.head.eval()
    x = torch.as_tensor(embeddings, dtype=torch.float32).to(next(head.parameters()).device)
    # Each crop votes; summing log-probabilities rewards cells every view agrees on.
    log_probs = torch.log_softmax(head(x), dim=1).mean(dim=0)
    probs = debias(
        log_probs.cpu().numpy(),
        checkpoint.log_prior,
        prior_strength,
        checkpoint.game_log_prior,
        game_prior_strength,
    )
    cells = checkpoint.cells
    probs = probs * cell_weights(cells.centroids, evidence, coverage_strength)
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
        game_prior_strength: float = DEFAULT_GAME_PRIOR_STRENGTH,
        coverage_strength: float = DEFAULT_COVERAGE_STRENGTH,
    ) -> None:
        self.prior_strength = prior_strength
        self.game_prior_strength = game_prior_strength
        self.coverage_strength = coverage_strength
        self.checkpoint = Checkpoint.load(checkpoint_path)
        self.encoder = ImageEncoder(self.checkpoint.backbone, device)
        self.head = self.checkpoint.head.to(self.encoder.device).eval()

    def predict(self, images: Sequence[Image.Image], evidence: Evidence | None = None) -> Guess:
        """Guess one location from several views of the same place."""
        if not images:
            raise ValueError("predict() needs at least one image")
        crops = [crop for image in images for crop in square_crops(image)]
        return guess_from_embeddings(
            self.checkpoint,
            self.encoder.encode(crops),
            self.prior_strength,
            self.game_prior_strength,
            evidence,
            self.coverage_strength,
        )
