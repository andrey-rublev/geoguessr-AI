"""Score a trained model on a labelled folder of images."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from ..geo import geoguessr_score, haversine_km
from .train import summarize


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
        distance = haversine_km(guess.lat, guess.lon, lat, lon)
        rows.append(
            {
                "group": str(group),
                "lat": lat,
                "lon": lon,
                "guess_lat": guess.lat,
                "guess_lon": guess.lon,
                "distance_km": distance,
                "score": geoguessr_score(distance),
            }
        )
    if not rows:
        raise ValueError(f"{labels_csv} has no rows")
    return summarize([row["distance_km"] for row in rows]), rows
