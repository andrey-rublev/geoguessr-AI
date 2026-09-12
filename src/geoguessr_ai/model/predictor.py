"""Run a trained checkpoint on screenshots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .backbone import ImageEncoder, square_crops
from .head import Checkpoint


@dataclass(frozen=True)
class Guess:
    lat: float
    lon: float
    expected_score: float
    """The model's own estimate of the points this guess earns (0-5000)."""
    top_cells: list[tuple[float, float, float]]
    """The five likeliest geocells as (lat, lon, probability)."""


class GeoPredictor:
    def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
        self.checkpoint = Checkpoint.load(checkpoint_path)
        self.encoder = ImageEncoder(self.checkpoint.backbone, device)
        self.head = self.checkpoint.head.to(self.encoder.device).eval()

    @torch.inference_mode()
    def predict(self, images: Sequence[Image.Image]) -> Guess:
        """Guess one location from several views of the same place."""
        if not images:
            raise ValueError("predict() needs at least one image")
        crops = [crop for image in images for crop in square_crops(image)]
        embeddings = self.encoder.encode(crops).to(self.encoder.device)
        # Each crop votes; summing log-probabilities rewards cells every view agrees on.
        log_probs = torch.log_softmax(self.head(embeddings), dim=1).mean(dim=0)
        probs = torch.softmax(log_probs, dim=0).cpu().numpy()

        cells = self.checkpoint.cells
        lat, lon, expected = cells.best_guess(probs)
        top = probs.argsort()[::-1][:5]
        top_cells = [
            (float(cells.centroids[i, 0]), float(cells.centroids[i, 1]), float(probs[i]))
            for i in top
        ]
        return Guess(lat, lon, expected, top_cells)
