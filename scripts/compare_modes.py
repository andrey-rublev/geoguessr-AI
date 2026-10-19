"""Compare multiplayer and singleplayer rounds: do they score apart, and if so, why?

Every saved round with an answer is grouped by how it was played, which rounds have saved
since 18 October 2026 (mark multiplayer sessions from before then with --multiplayer), and by
the model that guessed, since a better model alone moves the scores. Two comparisons:

* As played: the points each kind of round scored, in OpenGuessr's own scoring before its streak
  bonus, what the model expected of the places (in rounds saved since 19 October 2026, when the
  estimate was put on the same scale), and, in multiplayer, how often the timer left steps out
  or another player guessed first.
* Replayed: the current model guessing again from each round's saved views alone, on rounds it
  never trained on. Places or pictures that are harder in multiplayer show up here; what the
  timer took away shows up only as played.

    geoguessr-ai embed --rounds                 # after playing, for the replay
    python scripts/compare_modes.py --multiplayer 20261017-220700
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from geoguessr_ai.geo import haversine_km, openguessr_score
from geoguessr_ai.model.head import Ensemble
from geoguessr_ai.model.predictor import guess_from_embeddings
from geoguessr_ai.model.rounds import ROUNDS_FILE, RoundEmbeddings, is_test_round

DEFAULT_EMBEDDINGS = Path("data/embeddings/openai__clip-vit-base-patch32") / ROUNDS_FILE
MODES = ("singleplayer", "multiplayer")


def played(root: Path, multiplayer: set[str], since: str) -> list[dict]:
    """Every saved round from session ``since`` on, with how it was played and what it scored."""
    rounds = []
    for path in sorted(root.rglob("round.json")):
        session = path.parent.parent.name
        if session < since:
            continue
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        answer, pin, km = info.get("answer"), info["placed"], None
        if answer:
            km = haversine_km(pin["lat"], pin["lon"], answer["lat"], answer["lon"])
        party = info.get("party", session in multiplayer)
        rounds.append(
            {
                "id": f"{session}/{path.parent.name}",
                "mode": MODES[party],
                "model": info.get("model") or "not saved (rounds before 18 October 2026)",
                "points": None if km is None else openguessr_score(km),
                "unreadable": "answer" in info and not answer,  # recording, but couldn't read it
                "expected": info["guess"]["expected_score"] if "expected_in" in info else None,
                "views": len(info["headings"]),
                "read": bool(info.get("text")),
                "walked": info.get("walks", 0) > 0,
                "short": bool(info["short_of_time"]) if "short_of_time" in info else None,
                "others": info.get("others_guessed", False) if "party" in info else None,
            }
        )
    return rounds


def mean_error(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    return mean, statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else float("nan")


def share(rounds: list[dict], key: str) -> str:
    """The share of ``rounds`` where ``key`` holds, or "?" where rounds didn't save it."""
    if any(r[key] is None for r in rounds):
        return "?"
    return f"{sum(r[key] for r in rounds) / len(rounds):.0%}"


def as_played(rounds: list[dict]) -> None:
    by_model = defaultdict(list)
    for r in rounds:
        by_model[r["model"]].append(r)
    for model, some in by_model.items():
        print(f"\nAs played, by {model}")
        print(
            f"  {'':13}{'rounds':>7}{'answered':>9}{'unreadable':>11}{'points':>16}{'expected':>9}"
            f"{'views':>6}{'read':>6}{'walked':>7}{'short of time':>14}{'others first':>13}"
        )
        scored = {}
        for mode in MODES:
            group = [r for r in some if r["mode"] == mode]
            if not group:
                continue
            points = [r["points"] for r in group if r["points"] is not None]
            scored[mode] = points
            shown = "-" if not points else "{:,.0f} ± {:,.0f}".format(*mean_error(points))
            unreadable = sum(r["unreadable"] for r in group)
            estimates = [r["expected"] for r in group if r["expected"] is not None]
            expected = f"{statistics.mean(estimates):,.0f}" if estimates else "-"
            views = statistics.mean(r["views"] for r in group)
            print(
                f"  {mode:13}{len(group):7}{len(points):9}{unreadable:11}{shown:>16}"
                f"{expected:>9}{views:6.1f}{share(group, 'read'):>6}{share(group, 'walked'):>7}"
                f"{share(group, 'short'):>14}{share(group, 'others'):>13}"
            )
        if any(r["unreadable"] for r in some):
            print(
                "  (unreadable: answers the bot couldn't read off the result screen, so those "
                "rounds have no points; see answer_problem in their round.json)"
            )
        if all(len(scored.get(mode, ())) > 1 for mode in MODES):
            gap(*(scored[mode] for mode in MODES))


def gap(single: list[float], multi: list[float]) -> None:
    """The multiplayer rounds' gap from the singleplayer ones, and whether it could be luck."""
    (a, ea), (b, eb) = mean_error(single), mean_error(multi)
    diff, error = b - a, (ea**2 + eb**2) ** 0.5
    spread = statistics.pstdev(single + multi)
    luck = "could well be luck" if abs(diff) < 2 * error else "more than luck"
    print(f"  multiplayer {diff:+,.0f} ± {error:,.0f} points a round: {luck}", end="")
    if abs(diff) >= 50:  # rounds of each to tell a gap this size from luck, 4 times in 5
        print(f" (to tell {abs(diff):,.0f} apart takes ~{16 * spread**2 / diff**2:,.0f} of each)")
    else:
        print()


def replayed(rounds: list[dict], model_path: Path, embeddings: Path) -> None:
    """Each mode's rounds the model never trained on, guessed again from their views alone."""
    if not embeddings.exists():
        print(f"\nNo replay: {embeddings} not found; run `geoguessr-ai embed --rounds` first.")
        return
    model = Ensemble.load(model_path)
    saved = datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
    data = RoundEmbeddings.load(embeddings)
    modes = {r["id"]: r["mode"] for r in rounds}
    points = defaultdict(list)
    for round_id in data.round_ids:
        mode = modes.get(round_id)
        if mode is None or not (is_test_round(round_id) or round_id > saved):
            continue
        where = data.groups == round_id
        guess = guess_from_embeddings(model, data.embeddings[where])
        km = haversine_km(guess.lat, guess.lon, data.lat[where][0], data.lon[where][0])
        points[mode].append(openguessr_score(km))
    print(f"\nReplayed by {model_path}, from the views alone, on rounds it never trained on")
    for mode in MODES:
        if points[mode]:
            print(
                f"  {mode:13}{len(points[mode]):7} rounds"
                + " {:9,.0f} ± {:,.0f}".format(*mean_error(points[mode]))
            )
    embedded = set(data.round_ids)
    missing = sum(1 for r in rounds if r["points"] is not None and r["id"] not in embedded)
    if missing:
        print(
            f"  ({missing} answered rounds left out: not embedded yet (`geoguessr-ai embed "
            "--rounds`), or none of their views finished drawing)"
        )
    if all(len(points[mode]) > 1 for mode in MODES):
        gap(points["singleplayer"], points["multiplayer"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=Path, default=Path("runs"))
    parser.add_argument("--model", type=Path, default=Path("models/geoguessr.pt"))
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--multiplayer",
        action="append",
        default=[],
        metavar="SESSION",
        help="a session played in a multiplayer room before rounds said so, like 20261017-220700",
    )
    parser.add_argument(
        "--since", default="", metavar="SESSION", help="leave out sessions before this one"
    )
    args = parser.parse_args()
    rounds = played(args.rounds, set(args.multiplayer), args.since)
    as_played(rounds)
    replayed(rounds, args.model, args.embeddings)


if __name__ == "__main__":
    main()
