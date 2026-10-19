"""Learn the detail expert (see :mod:`.detail`) from the bot's saved rounds.

Close-ups of every round's views are embedded (once: they're kept with the rounds' embeddings),
a classifier learns from the training rounds which countries they belong to, and how much its
say should count is tuned on the held-out rounds, against the model's own guesses there. It is
only kept if it scores surely more points there than the model does on its own.

On 1,800 rounds (1,200 to learn from, 600 held out) it didn't: at its best it scored 9 points a
round fewer than the model alone (give or take 14), at every weight and regularisation tried.
CLIP's features of a close-up seem to say little the model hasn't seen in the whole view; it may
take more rounds, or an encoder trained on close-ups.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from ..geo import haversine_km, openguessr_score
from ..knowledge.countries import country_index
from ..knowledge.evidence import Evidence
from .detail import LOOKS, DetailExpert, fit, tiles
from .head import Ensemble
from .predictor import detail_path, guess_from_embeddings
from .rounds import ROUNDS_FILE, Round, RoundEmbeddings, embed_rounds, is_test_round

TILES_FILE = "openguessr-tiles.npz"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
"""How much the expert's say may count, tried in turn."""


def tile_crops(rounds: list[Round]) -> Iterator[tuple[Round, Image.Image]]:
    """Close-up tiles of each round's first look round."""
    for rnd in rounds:
        for view in rnd.views:
            if int(view.stem.split("_")[-1]) >= LOOKS:
                continue
            with Image.open(view) as image:
                for tile in tiles(image.convert("RGB")):
                    yield rnd, tile


def tune(
    expert: DetailExpert, model: Ensemble, rounds: RoundEmbeddings, close: RoundEmbeddings
) -> dict[float, np.ndarray]:
    """OpenGuessr points on each round of ``close`` that ``rounds`` has too, at each weight."""
    points: dict[float, list[float]] = {w: [] for w in WEIGHTS}
    for round_id in dict.fromkeys(close.groups.tolist()):
        where = rounds.groups == round_id
        if not where.any():
            continue
        say = close.embeddings[close.groups == round_id]
        lat, lon = rounds.lat[where][0], rounds.lon[where][0]
        for weight in WEIGHTS:
            evidence = Evidence(countries=expert.likelihood(say, weight))
            guess = guess_from_embeddings(model, rounds.embeddings[where], evidence=evidence)
            points[weight].append(openguessr_score(haversine_km(guess.lat, guess.lon, lat, lon)))
    return {w: np.array(p) for w, p in points.items()}


def learn_details(
    model_path: Path,
    runs: Path,
    embeddings: Path,
    encoder,
    pace: Callable[[], None] | None = None,
    c: float = 1.0,
) -> DetailExpert | None:
    """Learn and keep a detail expert for the model at ``model_path``, if it adds points on the
    held-out rounds; ``embeddings`` is the folder of the rounds' embeddings."""
    close = embed_rounds(
        encoder, runs, embeddings / TILES_FILE, crops=tile_crops, pace=pace, save_every=100
    )
    countries = country_index(close.lat, close.lon)
    held = np.array([is_test_round(g) for g in close.groups])
    print(f"Learning from {int((~held).sum()):,} close-ups of the training rounds' views")
    expert = fit(close.embeddings[~held], countries[~held], encoder.name, c)
    model = Ensemble.load(model_path)
    rounds = RoundEmbeddings.load(embeddings / ROUNDS_FILE)
    points = tune(expert, model, rounds, close.subset(held))
    if not len(points[0.0]):
        print("No held-out rounds to judge it on: keeping none.")
        return None
    best = max(WEIGHTS, key=lambda w: points[w].mean())
    gain = points[best] - points[0.0]
    print(
        f"On {len(gain)} held-out rounds: {points[0.0].mean():,.0f} points without it, "
        f"{points[best].mean():,.0f} with it at weight {best:g} "
        f"({gain.mean():+.0f}, give or take {gain.std() / len(gain) ** 0.5:.0f})"
    )
    path = detail_path(model_path)
    if best == 0.0 or gain.mean() <= gain.std() / len(gain) ** 0.5:
        # Not even one standard error better, after picking the best weight on these rounds.
        print("It doesn't help surely enough: keeping none.")
        path.unlink(missing_ok=True)
        return None
    expert = fit(close.embeddings, countries, encoder.name, c)  # every round, now it's tuned
    expert.weight = best
    expert.save(path)
    print(f"Saved to {path}: the bot reads close-ups from now on.")
    return expert
