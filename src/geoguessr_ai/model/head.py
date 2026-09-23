"""The trainable part of the model (an MLP from embeddings to geocell logits) and its checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..files import write_safely
from .geocells import GeoCells


class GeoHead(nn.Module):
    def __init__(self, embed_dim: int, n_cells: int, hidden: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_cells),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


@dataclass
class Checkpoint:
    head: GeoHead
    cells: GeoCells
    backbone: str
    metrics: dict[str, float] = field(default_factory=dict)
    log_prior: np.ndarray | None = None
    """Log share of training photos per cell, used to debias predictions."""
    game_log_prior: np.ndarray | None = None
    """Where the game sends players, per cell, estimated from played training rounds."""

    def save(self, path: Path) -> None:
        raw = {
            "state_dict": self.head.state_dict(),
            "embed_dim": self.head.embed_dim,
            "hidden": self.head.hidden,
            "centroids": torch.from_numpy(self.cells.centroids),
            "backbone": self.backbone,
            "metrics": self.metrics,
            "log_prior": _tensor(self.log_prior),
            "game_log_prior": _tensor(self.game_log_prior),
        }
        write_safely(Path(path), lambda file: torch.save(raw, file))

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
        cells = GeoCells(raw["centroids"].numpy())
        head = GeoHead(raw["embed_dim"], len(cells), hidden=raw["hidden"])
        head.load_state_dict(raw["state_dict"])
        # Both priors are absent from checkpoints saved before they existed.
        log_prior, game_log_prior = raw.get("log_prior"), raw.get("game_log_prior")
        return cls(
            head.eval(),
            cells,
            raw["backbone"],
            raw["metrics"],
            None if log_prior is None else log_prior.numpy(),
            None if game_log_prior is None else game_log_prior.numpy(),
        )


def _tensor(array: np.ndarray | None) -> torch.Tensor | None:
    return None if array is None else torch.from_numpy(array)
