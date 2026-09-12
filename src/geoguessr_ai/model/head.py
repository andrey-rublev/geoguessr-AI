"""The trainable part of the model (an MLP from embeddings to geocell logits) and its checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

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

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.head.state_dict(),
                "embed_dim": self.head.embed_dim,
                "hidden": self.head.hidden,
                "centroids": torch.from_numpy(self.cells.centroids),
                "backbone": self.backbone,
                "metrics": self.metrics,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
        cells = GeoCells(raw["centroids"].numpy())
        head = GeoHead(raw["embed_dim"], len(cells), hidden=raw["hidden"])
        head.load_state_dict(raw["state_dict"])
        return cls(head.eval(), cells, raw["backbone"], raw["metrics"])
