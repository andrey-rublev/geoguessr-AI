"""OpenGuessr rounds the bot has played, reused as training and test data.

``geoguessr-ai play`` saves every round under ``runs/<session>/round_NN/``: the views it
looked at and the real location, read off the result screen. Unlike the Mapillary photos in
OSV-5M, these are exactly the Google Street View images the bot faces in the game.

Rounds are split into train and test by a hash of their id, so a round never switches
sides as more games are played.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..files import write_safely
from .backbone import ImageEncoder, square_crops

ROUNDS_FILE = "openguessr-rounds.npz"
DEFAULT_TEST_FRACTION = 0.3


@dataclass(frozen=True)
class Round:
    id: str
    """``<session>/<round folder>``, e.g. ``20260913-161603/round_01``."""
    views: tuple[Path, ...]
    lat: float
    lon: float


def find_rounds(root: Path) -> list[Round]:
    """Every saved round under ``root`` whose real location was recorded, with the views Street
    View finished drawing: a black or blurred one teaches the model nothing about where it was.

    A round that can't be read is skipped rather than raising, so one unreadable file, as a
    power cut mid-write would leave behind, can't cost a whole session its learning.
    """
    rounds = []
    for info_path in sorted(Path(root).rglob("round.json")):
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as broken:
            print(f"Skipping {info_path}: {broken}")
            continue
        answer, undrawn = info.get("answer"), {f"view_{i}" for i in info.get("undrawn_views", ())}
        views = tuple(
            view for view in sorted(info_path.parent.glob("view_*.jpg")) if view.stem not in undrawn
        )
        if answer and views:
            folder = info_path.parent
            round_id = f"{folder.parent.name}/{folder.name}"
            rounds.append(Round(round_id, views, float(answer["lat"]), float(answer["lon"])))
    return rounds


def is_test_round(round_id: str, test_fraction: float = DEFAULT_TEST_FRACTION) -> bool:
    """Whether a round is held out for testing; depends only on its id."""
    digest = hashlib.sha256(round_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < test_fraction


@dataclass
class RoundEmbeddings:
    """One embedding per crop of every view, tagged with the round it came from."""

    embeddings: np.ndarray
    groups: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    backbone: str

    def __len__(self) -> int:
        return len(self.groups)

    @property
    def round_ids(self) -> list[str]:
        return list(dict.fromkeys(self.groups.tolist()))

    @classmethod
    def load(cls, path: Path) -> RoundEmbeddings:
        with np.load(path) as data:
            return cls(
                data["embeddings"].astype(np.float32),
                data["groups"].astype(str),
                data["lat"].astype(np.float64),
                data["lon"].astype(np.float64),
                str(data["backbone"]),
            )

    def save(self, path: Path) -> None:
        write_safely(
            Path(path),
            lambda file: np.savez(
                file,
                embeddings=self.embeddings.astype(np.float16),
                groups=self.groups,
                lat=self.lat.astype(np.float32),
                lon=self.lon.astype(np.float32),
                backbone=np.array(self.backbone),
            ),
        )

    def subset(self, mask: np.ndarray) -> RoundEmbeddings:
        return RoundEmbeddings(
            self.embeddings[mask], self.groups[mask], self.lat[mask], self.lon[mask], self.backbone
        )

    def split(
        self, test_fraction: float = DEFAULT_TEST_FRACTION
    ) -> tuple[RoundEmbeddings, RoundEmbeddings]:
        """``(train, test)`` rows, keeping every round whole."""
        is_test = np.array([is_test_round(g, test_fraction) for g in self.groups], dtype=bool)
        return self.subset(~is_test), self.subset(is_test)


def _crops(rounds: list[Round]) -> Iterator[tuple[Round, Image.Image]]:
    for rnd in rounds:
        for view in rnd.views:
            with Image.open(view) as image:
                for crop in square_crops(image.convert("RGB")):
                    yield rnd, crop


def embed_rounds(
    encoder: ImageEncoder, root: Path, out_path: Path, *, batch_size: int = 64
) -> RoundEmbeddings:
    """Bring ``out_path`` in line with the rounds under ``root``.

    Only rounds not embedded yet go through the encoder. Rounds deleted from ``root`` are
    dropped, and labels are re-read, so fixing a round's ``round.json`` takes effect.
    """
    rounds = {rnd.id: rnd for rnd in find_rounds(root)}
    if not rounds:
        raise FileNotFoundError(
            f"No rounds with a recorded answer under {root}. Play some with `geoguessr-ai play`."
        )
    out_path = Path(out_path)
    kept = None
    if out_path.exists():
        old = RoundEmbeddings.load(out_path)
        if old.backbone != encoder.name:
            raise ValueError(f"{out_path} was embedded with {old.backbone}, not {encoder.name}")
        kept = old.subset(np.isin(old.groups, list(rounds)))
        kept.lat = np.array([rounds[g].lat for g in kept.groups])
        kept.lon = np.array([rounds[g].lon for g in kept.groups])

    done = set() if kept is None else set(kept.groups.tolist())
    new = [rnd for round_id, rnd in rounds.items() if round_id not in done]
    parts = [] if kept is None else [kept]
    if new:
        chunks, groups, lats, lons = [], [], [], []
        crops = _crops(new)
        with tqdm(total=len(new), unit="round", desc="rounds") as bar:
            while batch := list(islice(crops, batch_size)):
                chunks.append(encoder.encode([crop for _, crop in batch]).numpy())
                for rnd, _ in batch:
                    groups.append(rnd.id)
                    lats.append(rnd.lat)
                    lons.append(rnd.lon)
                bar.n = len(set(groups))
                bar.refresh()
        parts.append(
            RoundEmbeddings(
                np.concatenate(chunks).astype(np.float32),
                np.array(groups),
                np.array(lats),
                np.array(lons),
                encoder.name,
            )
        )
    merged = RoundEmbeddings(
        np.concatenate([p.embeddings for p in parts]),
        np.concatenate([p.groups for p in parts]),
        np.concatenate([p.lat for p in parts]),
        np.concatenate([p.lon for p in parts]),
        encoder.name,
    )
    merged.save(out_path)
    return merged
