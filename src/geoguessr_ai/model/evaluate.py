"""Score a trained model on held-out photos, a labelled folder, or your own OpenGuessr rounds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from ..geo import geoguessr_score, haversine_km
from .backbone import pick_device
from .geocells import DEFAULT_GAME_PRIOR_STRENGTH, DEFAULT_PRIOR_STRENGTH
from .head import Checkpoint
from .predictor import Guess, guess_from_embeddings
from .rounds import DEFAULT_TEST_FRACTION, RoundEmbeddings
from .train import evaluate, load_embeddings, summarize


def evaluate_embeddings(
    checkpoint_path: Path,
    embedding_files: list[Path],
    device: str = "auto",
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> dict[str, float]:
    """Score a checkpoint on precomputed embeddings, e.g. the OSV-5M test split."""
    checkpoint = Checkpoint.load(checkpoint_path)
    x, lat, lon, backbone = load_embeddings(embedding_files)
    if backbone != checkpoint.backbone:
        raise ValueError(
            f"Embeddings come from {backbone} but the model was trained on {checkpoint.backbone}"
        )
    dev = pick_device(device)
    return {
        "places": len(x),
        **evaluate(
            checkpoint.head.to(dev),
            checkpoint.cells,
            x,
            lat,
            lon,
            dev,
            log_prior=checkpoint.log_prior,
            prior_strength=prior_strength,
        ),
    }


def _row(group: str, lat: float, lon: float, guess: Guess) -> dict:
    distance = haversine_km(guess.lat, guess.lon, lat, lon)
    return {
        "group": group,
        "lat": lat,
        "lon": lon,
        "guess_lat": guess.lat,
        "guess_lon": guess.lon,
        "distance_km": distance,
        "score": geoguessr_score(distance),
    }


def evaluate_folder(
    predictor, folder: Path, labels_csv: Path
) -> tuple[dict[str, float], list[dict]]:
    """Predict every place in ``labels_csv`` and compare with the truth.

    The CSV needs ``filename,latitude,longitude``. An optional ``group`` column bundles
    several views of the same place into one prediction, exactly like a game round.
    Returns summary metrics and one row per place.
    """
    folder = Path(folder)
    labels = pd.read_csv(labels_csv, dtype={"filename": str})
    if "group" not in labels.columns:
        labels["group"] = labels["filename"]

    rows = []
    for group, views in labels.groupby("group", sort=False):
        images = [Image.open(folder / name).convert("RGB") for name in views["filename"]]
        guess = predictor.predict(images)
        lat, lon = float(views["latitude"].iloc[0]), float(views["longitude"].iloc[0])
        rows.append(_row(str(group), lat, lon, guess))
    if not rows:
        raise ValueError(f"{labels_csv} has no rows")
    return summarize([row["distance_km"] for row in rows]), rows


def evaluate_rounds(
    checkpoint_path: Path,
    rounds_path: Path,
    device: str = "auto",
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    game_prior_strength: float = DEFAULT_GAME_PRIOR_STRENGTH,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> tuple[dict[str, float], list[dict]]:
    """Score a checkpoint on your held-out OpenGuessr rounds, views combined as in the game."""
    checkpoint = Checkpoint.load(checkpoint_path)
    _, test = RoundEmbeddings.load(rounds_path).split(test_fraction)
    if test.backbone != checkpoint.backbone:
        raise ValueError(
            f"{rounds_path} was embedded with {test.backbone}, "
            f"but the model was trained on {checkpoint.backbone}"
        )
    if not len(test):
        raise ValueError(f"No rounds in {rounds_path} are held out for testing yet; play more")
    checkpoint.head.to(pick_device(device))

    rows = []
    for round_id in test.round_ids:
        mask = test.groups == round_id
        guess = guess_from_embeddings(
            checkpoint, test.embeddings[mask], prior_strength, game_prior_strength
        )
        rows.append(_row(round_id, float(test.lat[mask][0]), float(test.lon[mask][0]), guess))
    return summarize([row["distance_km"] for row in rows]), rows
