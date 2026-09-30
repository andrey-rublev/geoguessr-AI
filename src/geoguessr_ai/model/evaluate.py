"""Score a trained model on held-out photos, a labelled folder, or your own OpenGuessr rounds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..geo import geoguessr_score, haversine_km
from ..knowledge.evidence import DEFAULT_COVERAGE_STRENGTH
from .backbone import pick_device
from .geocells import DEFAULT_GAME_PRIOR_STRENGTH, DEFAULT_PRIOR_STRENGTH
from .head import Ensemble
from .predictor import Guess, guess_from_embeddings
from .rounds import DEFAULT_TEST_FRACTION, RoundEmbeddings
from .train import load_embeddings, summarize


def evaluate_embeddings(
    checkpoint_path: Path,
    embedding_files: list[Path],
    device: str = "auto",
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> dict[str, float]:
    """Score a model on precomputed embeddings, e.g. the OSV-5M test split, each a place."""
    model = Ensemble.load(checkpoint_path)
    x, lat, lon, backbone = load_embeddings(embedding_files)
    if backbone != model.backbone:
        raise ValueError(
            f"Embeddings come from {backbone} but the model was trained on {model.backbone}"
        )
    model.to(pick_device(device))
    guesses = []
    for i in range(0, len(x), 1024):
        probs = model.belief(model.log_probs(x[i : i + 1024], one_place=False), prior_strength)
        guesses.extend(model.cells.best_guess(p)[:2] for p in probs)
    g = np.array(guesses)
    return {"places": len(x), **summarize(haversine_km(g[:, 0], g[:, 1], lat, lon))}


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
    coverage_strength: float = DEFAULT_COVERAGE_STRENGTH,
) -> tuple[dict[str, float], list[dict]]:
    """Score a model on your held-out OpenGuessr rounds, views combined as in the game."""
    model = Ensemble.load(checkpoint_path)
    _, test = RoundEmbeddings.load(rounds_path).split(test_fraction)
    if test.backbone != model.backbone:
        raise ValueError(
            f"{rounds_path} was embedded with {test.backbone}, "
            f"but the model was trained on {model.backbone}"
        )
    if not len(test):
        raise ValueError(f"No rounds in {rounds_path} are held out for testing yet; play more")
    model.to(pick_device(device))

    rows = []
    for round_id in test.round_ids:
        mask = test.groups == round_id
        guess = guess_from_embeddings(
            model,
            test.embeddings[mask],
            prior_strength,
            game_prior_strength,
            coverage_strength=coverage_strength,
        )
        rows.append(_row(round_id, float(test.lat[mask][0]), float(test.lon[mask][0]), guess))
    return summarize([row["distance_km"] for row in rows]), rows
