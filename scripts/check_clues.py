"""Check the clues against your own played rounds: are they right, and what are they worth?

Replays every saved round without touching the screen: the model's belief from the round's
embeddings, then the same belief reweighed by the clues read from that round's saved text and
sun. Nothing is retrained, so changing a clue and rerunning this shows what the change did.

    geoguessr-ai embed --rounds          # once, after playing
    python scripts/check_clues.py        # every round the model has seen, and held-out ones

Rounds the model trained on are easy for it, so the held-out figures are the honest ones.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from geoguessr_ai.geo import geoguessr_score, haversine_km
from geoguessr_ai.knowledge.countries import country_at, country_codes
from geoguessr_ai.knowledge.evidence import Evidence, cell_weights
from geoguessr_ai.knowledge.sun import latitude_likelihood
from geoguessr_ai.knowledge.text import TextLine, text_clues
from geoguessr_ai.model.geocells import debias
from geoguessr_ai.model.head import Checkpoint
from geoguessr_ai.model.rounds import ROUNDS_FILE, RoundEmbeddings, is_test_round

SUN_NOTE = re.compile(r"sun to the [\w-]+ \((\d+) deg\), (-?\d+) deg up")
DEFAULT_EMBEDDINGS = Path("data/embeddings/openai__clip-vit-base-patch32") / ROUNDS_FILE


def saved_rounds(root: Path) -> dict[str, dict]:
    """Every played round's own record, by ``<session>/<round folder>``."""
    found = {}
    for path in sorted(root.rglob("round.json")):
        found[f"{path.parent.parent.name}/{path.parent.name}"] = json.loads(
            path.read_text(encoding="utf-8")
        )
    return found


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
    coverage_only = cell_weights(cells.centroids, None, 1.0)
    codes = country_codes()

    fits: collections.Counter[str] = collections.Counter()
    misses: list[tuple[str, str, str]] = []
    read = 0
    worth: dict[bool, list[tuple[float, float]]] = {True: [], False: []}
    gains: dict[str, list[float]] = collections.defaultdict(list)

    for round_id in embedded.round_ids:
        info = saved.get(round_id)
        if info is None:
            continue
        where = embedded.groups == round_id
        crops = torch.as_tensor(embedded.embeddings[where], dtype=torch.float32)
        with torch.inference_mode():
            log_probs = torch.log_softmax(head(crops), dim=1).mean(dim=0).cpu().numpy()
        belief = debias(log_probs, checkpoint.log_prior, 1.0, checkpoint.game_log_prior, 1.0)
        lat, lon = float(embedded.lat[where][0]), float(embedded.lon[where][0])

        def points(
            weights: np.ndarray, belief: np.ndarray = belief, lat: float = lat, lon: float = lon
        ) -> float:
            odds = belief * weights
            guess = cells.best_guess(odds / odds.sum())
            return float(geoguessr_score(haversine_km(guess[0], guess[1], lat, lon)))

        lines = [TextLine(t["text"], t["script"], t["score"]) for t in info.get("text", [])]
        clues = text_clues(lines) if lines else None
        evidence = Evidence(countries=None if clues is None else clues.likelihood)
        for note in info.get("clues", []):
            if found := SUN_NOTE.match(note):
                azimuth, height = float(found.group(1)), float(found.group(2))
                evidence.latitude = latitude_likelihood(azimuth, height)

        before = points(coverage_only)
        after = points(cell_weights(cells.centroids, evidence, 1.0))
        worth[is_test_round(round_id)].append((before, after))

        code = country_at(lat, lon)
        for note in [] if clues is None else clues.notes:
            kind = note.split(":")[0].split("(")[0].strip()
            gains[kind].append(after - before)
            if code in codes:
                read += 1
                if clues.likelihood[codes.index(code)] > 0.5:
                    fits[kind] += 1
                else:
                    misses.append((code, round_id, note))
        if evidence.latitude is not None:
            gains["(the sun)"].append(after - before)

    for held_out, label in ((False, "rounds the model trained on"), (True, "held-out rounds")):
        pairs = worth[held_out]
        if not pairs:
            continue
        before, after = (np.array(v) for v in zip(*pairs, strict=True))
        changed = after - before
        moved = np.abs(changed) > 1
        print(
            f"\n{len(pairs)} {label}: {before.mean():.0f} without clues, {after.mean():.0f} with"
            f"\n  clues are worth {changed.mean():+.0f} +- "
            f"{changed.std(ddof=1) / len(changed) ** 0.5:.0f} a round, changing "
            f"{int(moved.sum())} of them ({int((changed > 1).sum())} up, "
            f"{int((changed < -1).sum())} down)"
        )

    print("\nwhat each kind of clue was worth where it fired:")
    for kind, deltas in sorted(gains.items(), key=lambda kv: -sum(kv[1])):
        got = np.array(deltas)
        print(f"  {kind:42s} n={len(got):3d} {got.mean():+7.0f} each {got.sum():+9.0f} total")
    print(f"\nclues from text that fitted the real country: {sum(fits.values())} of {read}")
    for code, round_id, note in misses:
        print(f"  missed {code} in {round_id}: {note}")


if __name__ == "__main__":
    main()
