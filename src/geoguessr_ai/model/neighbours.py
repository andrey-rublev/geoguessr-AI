"""Places that look alike: once the model has chosen a geocell, the guess moves from the cell's
centre to where the photos and rounds that look most like the views were taken.

The references are every embedded photo and saved round, filed by the cell each lies in. For a
round, the views' embeddings are averaged, and the :data:`NEIGHBOURS` references in the chosen
cell that look most like them vote for where in it the answer is, the closer the look the more
(a softmax over their similarity at :data:`TEMPERATURE`). On 1,052 held-out rounds, with
StreetCLIP and 250,000 photos, that added about 6 OpenGuessr points a round (give or take 2),
on both halves of them; with plain CLIP it had added nothing.

They help choose the cell too: the :data:`VOTERS` references that look most like the views, from
anywhere, vote for the cells they lie in, and each cell's probability is weighed by its share of
the vote (see :meth:`Neighbours.vote`). On the same rounds that added about 14 points a round
more: settings chosen on half of them scored that on the other half, over 2,000 such splits, and
lost points in 11% of them. Every setting tried added 10 to 12 on average.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..geo import from_unit_vectors, to_unit_vectors
from .geocells import GeoCells
from .rounds import ROUNDS_FILE, RoundEmbeddings

NEIGHBOURS = 25
TEMPERATURE = 0.02
VOTERS = 200
VOTE_TEMPERATURE = 0.05
VOTE_FLOOR = 0.01
"""What a cell with no say in the vote keeps of it, so a vote never rules a cell out."""
VOTE_STRENGTH = 0.5


@dataclass
class Neighbours:
    embeddings: np.ndarray
    """(N, dim) L2-normalised embeddings of the references."""
    lat: np.ndarray
    lon: np.ndarray
    cell: np.ndarray
    """The geocell each reference lies in."""
    _order: np.ndarray = field(init=False, repr=False)
    _starts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._order = np.argsort(self.cell, kind="stable")
        self._starts = np.searchsorted(self.cell[self._order], np.arange(self.cell.max() + 2))

    def __len__(self) -> int:
        return len(self.cell)

    @classmethod
    def build(cls, embeddings, lat, lon, cells: GeoCells) -> Neighbours:
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        # Filed in chunks: every reference against every cell at once is gigabytes.
        found = [
            cells.assign(lat[i : i + 20_000], lon[i : i + 20_000])
            for i in range(0, len(lat), 20_000)
        ]
        return cls(np.asarray(embeddings), lat, lon, np.concatenate(found))

    def vote(self, query: np.ndarray, n_cells: int) -> np.ndarray:
        """A weight for each of ``n_cells`` cells: the share of the :data:`VOTERS` references most
        like ``query`` (an embedding) that lie in it, the closer the look the more, plus
        :data:`VOTE_FLOOR`, to the power :data:`VOTE_STRENGTH`."""
        query = np.asarray(query, np.float32) / np.linalg.norm(query)
        embeddings = torch.from_numpy(self.embeddings)
        sim = (embeddings @ torch.from_numpy(query).to(embeddings.dtype)).float().numpy()
        take = np.argpartition(-sim, min(VOTERS, len(sim)) - 1)[:VOTERS]
        weight = np.exp((sim[take] - sim[take].max()) / VOTE_TEMPERATURE)
        share = np.bincount(self.cell[take], weights=weight, minlength=n_cells)[:n_cells]
        return (share / share.sum() + VOTE_FLOOR) ** VOTE_STRENGTH

    def near(self, query: np.ndarray, cell: int) -> tuple[float, float] | None:
        """Where in ``cell`` the references most like ``query`` (an embedding) were taken; None
        if no reference lies in it."""
        if cell + 1 >= len(self._starts):
            return None
        refs = self._order[self._starts[cell] : self._starts[cell + 1]]
        if not len(refs):
            return None
        query = np.asarray(query, np.float32)
        sim = self.embeddings[refs].astype(np.float32) @ (query / np.linalg.norm(query))
        take = np.argsort(-sim)[:NEIGHBOURS]
        weight = np.exp((sim[take] - sim[take].max()) / TEMPERATURE)
        towards = (
            to_unit_vectors(self.lat[refs[take]], self.lon[refs[take]]) * weight[:, None]
        ).sum(0)
        lat, lon = from_unit_vectors(towards)
        return float(lat), float(lon)


def load_neighbours(folder: Path, backbone: str, cells: GeoCells) -> Neighbours | None:
    """The photos and rounds embedded in ``folder`` with ``backbone``, as references on ``cells``:
    each round once, its crops averaged. None when there are none."""
    from .train import resolve_embedding_files

    embeddings, lats, lons = [], [], []
    try:
        photos = resolve_embedding_files([folder])
    except FileNotFoundError:
        photos = []
    for path in photos:
        with np.load(path) as saved:
            if str(saved["backbone"]) == backbone:  # kept 16-bit: 250,000 photos are 400 MB
                embeddings.append(saved["embeddings"])
                lats.append(saved["lat"]), lons.append(saved["lon"])
    rounds_path = Path(folder) / ROUNDS_FILE
    if rounds_path.exists():
        rounds = RoundEmbeddings.load(rounds_path)
        if rounds.backbone == backbone and len(rounds):
            ids, first, which = np.unique(rounds.groups, return_index=True, return_inverse=True)
            mean = np.zeros((len(ids), rounds.embeddings.shape[1]), np.float32)
            np.add.at(mean, which, rounds.embeddings)
            embeddings.append(mean / np.linalg.norm(mean, axis=1, keepdims=True))
            lats.append(rounds.lat[first]), lons.append(rounds.lon[first])
    if not embeddings:
        return None
    return Neighbours.build(
        np.concatenate([e.astype(np.float16) for e in embeddings]),
        np.concatenate(lats),
        np.concatenate(lons),
        cells,
    )
