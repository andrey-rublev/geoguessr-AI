"""The trainable part of the model (an MLP from embeddings to geocell logits), its checkpoint,
and an ensemble of checkpoints that the bot guesses with."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..files import write_safely
from .geocells import GeoCells, debias


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
        write_safely(Path(path), lambda file: torch.save(self._raw(), file))

    def _raw(self) -> dict:
        return {
            "state_dict": self.head.state_dict(),
            "embed_dim": self.head.embed_dim,
            "hidden": self.head.hidden,
            "centroids": torch.from_numpy(self.cells.centroids),
            "backbone": self.backbone,
            "metrics": self.metrics,
            "log_prior": _tensor(self.log_prior),
            "game_log_prior": _tensor(self.game_log_prior),
        }

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
        if "members" in raw:
            raise ValueError(f"{path} holds several checkpoints: load it as an Ensemble")
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict) -> Checkpoint:
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


@dataclass
class Ensemble:
    """Checkpoints trained apart, whose beliefs are averaged on the cells of the first, the
    newest. They make different mistakes: on 883 held-out rounds the model of 24 September
    and the one retrained on 28 September scored 3,349 and 3,344 points a round alone, and
    3,414 (give or take 23) together."""

    members: list[Checkpoint]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("an ensemble needs at least one checkpoint")
        if len({m.backbone for m in self.members}) > 1:
            raise ValueError("every checkpoint in an ensemble must use the same backbone")
        # Each member's cells onto the nearest of the first member's: cells are refitted on
        # every training run, and only about a third come out in the same place.
        self._onto = [self.cells.assign(*m.cells.centroids.T) for m in self.members]

    @property
    def cells(self) -> GeoCells:
        return self.members[0].cells

    @property
    def backbone(self) -> str:
        return self.members[0].backbone

    def to(self, device) -> Ensemble:
        for member in self.members:
            member.head.to(device)
        return self

    def log_probs(self, embeddings: torch.Tensor | np.ndarray, one_place: bool = True):
        """Each member's log-probability per cell of its own: for one place, every crop voting
        (summing log-probabilities rewards cells every view agrees on), or with
        ``one_place=False`` for each embedding as a place of its own."""
        found = []
        with torch.inference_mode():
            for member in self.members:
                head = member.head.eval()
                x = torch.as_tensor(embeddings, dtype=torch.float32)
                log_probs = torch.log_softmax(head(x.to(next(head.parameters()).device)), dim=1)
                found.append((log_probs.mean(dim=0) if one_place else log_probs).cpu().numpy())
        return found

    def belief(
        self,
        log_probs: list[np.ndarray],
        prior_strength: float,
        game_prior_strength: float = 0.0,
    ) -> np.ndarray:
        """Probability per cell, from :meth:`log_probs`: each member's, debiased (see
        :func:`.geocells.debias`), on the first member's cells, averaged."""
        total = 0.0
        for member, onto, lp in zip(self.members, self._onto, log_probs, strict=True):
            probs = debias(
                lp, member.log_prior, prior_strength, member.game_log_prior, game_prior_strength
            )
            moved = np.zeros((len(self.cells), *probs.shape[:-1]))
            np.add.at(moved, onto, np.moveaxis(probs, -1, 0))
            total = total + np.moveaxis(moved, 0, -1)
        return total / len(self.members)

    def save(self, path: Path) -> None:
        raw = {"members": [member._raw() for member in self.members]}
        write_safely(Path(path), lambda file: torch.save(raw, file))

    @classmethod
    def load(cls, path: Path) -> Ensemble:
        """An ensemble file, or a single checkpoint as an ensemble of one."""
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls([Checkpoint._from_raw(r) for r in raw.get("members", [raw])])


def _tensor(array: np.ndarray | None) -> torch.Tensor | None:
    return None if array is None else torch.from_numpy(array)
