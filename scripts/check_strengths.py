"""Check how hard to lean on what the bot knows besides the photos, against your own rounds.

Three settings decide that, and this measures them instead of guessing:

* ``--prior-strength``: how much of the training set's own bias to divide out of the model.
* ``--game-prior-strength``: how much to favour where the game has really sent you.
* ``--coverage-strength``: how much to favour countries with more Street View.

Half the held-out rounds choose the settings and the other half say what they are worth, so
a number that only suits the rounds it was picked on shows up as one that doesn't carry over.

    geoguessr-ai embed --rounds          # once, after playing
    python scripts/check_strengths.py    # a few minutes

The best settings are the defaults in the code; rerun this after a lot more training, since
what suits the model can move as the model gets better.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from geoguessr_ai.geo import geoguessr_score, haversine_km
from geoguessr_ai.knowledge.evidence import (
    DEFAULT_COVERAGE_STRENGTH,
    Evidence,
    cell_weights,
)
from geoguessr_ai.knowledge.sun import latitude_likelihood
from geoguessr_ai.knowledge.text import TextLine, text_clues
from geoguessr_ai.model.geocells import (
    DEFAULT_GAME_PRIOR_STRENGTH,
    DEFAULT_PRIOR_STRENGTH,
    debias,
)
from geoguessr_ai.model.head import Checkpoint
from geoguessr_ai.model.rounds import ROUNDS_FILE, RoundEmbeddings, is_test_round

sys.path.insert(0, str(Path(__file__).parent))
from check_clues import SUN_NOTE, saved_rounds  # noqa: E402

DEFAULT_EMBEDDINGS = Path("data/embeddings/openai__clip-vit-base-patch32") / ROUNDS_FILE
NOW = (DEFAULT_PRIOR_STRENGTH, DEFAULT_GAME_PRIOR_STRENGTH, DEFAULT_COVERAGE_STRENGTH)
TRIED = ((0.3, 0.45, 0.6, 0.75, 1.0, 1.25), (0.0, 0.25, 0.5, 0.75, 1.0, 1.5), (0.25, 0.5, 1.0, 1.5))


def choosing(round_id: str) -> bool:
    """Whether a held-out round chooses the settings, rather than checking them."""
    return hashlib.md5(round_id.encode("utf-8")).digest()[0] < 128


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=Path, default=Path("runs"))
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--model", type=Path, default=Path("models/geoguessr.pt"))
    args = parser.parse_args()
    if not args.embeddings.exists():
        sys.exit(f"{args.embeddings} not found: run `geoguessr-ai embed --rounds` first")

    saved = saved_rounds(args.rounds)
    checkpoint = Checkpoint.load(args.model)
    cells, head = checkpoint.cells, checkpoint.head.eval()
    embedded = RoundEmbeddings.load(args.embeddings)

    played = []
    for round_id in embedded.round_ids:
        info = saved.get(round_id)
        if info is None or not is_test_round(round_id):
            continue
        where = embedded.groups == round_id
        crops = torch.as_tensor(embedded.embeddings[where], dtype=torch.float32)
        with torch.inference_mode():
            log_probs = torch.log_softmax(head(crops), dim=1).mean(dim=0).cpu().numpy()
        lines = [TextLine(t["text"], t["script"], t["score"]) for t in info.get("text", [])]
        clues = text_clues(lines) if lines else None
        evidence = Evidence(countries=None if clues is None else clues.likelihood)
        for note in info.get("clues", []):
            if found := SUN_NOTE.match(note):
                evidence.latitude = latitude_likelihood(
                    float(found.group(1)), float(found.group(2))
                )
        played.append(
            (
                round_id,
                log_probs,
                evidence,
                float(embedded.lat[where][0]),
                float(embedded.lon[where][0]),
            )
        )
    choose = [r for r in played if choosing(r[0])]
    check = [r for r in played if not choosing(r[0])]
    print(f"{len(choose)} held-out rounds to choose on, {len(check)} to check on")

    weights: dict[float, list[np.ndarray]] = {}

    def scores(group: list, prior: float, game: float, cover: float) -> np.ndarray:
        if cover not in weights:
            weights[cover] = {r[0]: cell_weights(cells.centroids, r[2], cover) for r in played}
        got = []
        for round_id, log_probs, _, lat, lon in group:
            belief = debias(log_probs, checkpoint.log_prior, prior, checkpoint.game_log_prior, game)
            odds = belief * weights[cover][round_id]
            guess = cells.best_guess(odds / odds.sum())
            got.append(float(geoguessr_score(haversine_km(guess[0], guess[1], lat, lon))))
        return np.array(got)

    tried = []
    for prior in TRIED[0]:
        for game in TRIED[1]:
            for cover in TRIED[2]:
                tried.append((scores(choose, prior, game, cover).mean(), (prior, game, cover)))
    tried.sort(key=lambda got: -got[0])
    now = scores(check, *NOW)
    print(
        f"  as it is now  prior {NOW[0]}, game {NOW[1]}, coverage {NOW[2]}: check {now.mean():.0f}"
    )
    for chose, (prior, game, cover) in tried[:8]:
        got = scores(check, prior, game, cover)
        moved = got - now
        print(
            f"  prior {prior:<5} game {game:<5} coverage {cover:<5}"
            f" chose {chose:5.0f}  check {got.mean():5.0f}"
            f"  ({moved.mean():+5.0f} +-{moved.std(ddof=1) / len(moved) ** 0.5:3.0f})"
        )
    print(json.dumps({"best_on_the_choosing_half": tried[0][1]}))


if __name__ == "__main__":
    main()
