"""Does CLIP's own idea of which country a photo shows, from prompts like "a Street View photo
taken in Kenya", improve the trained model's guesses? Scores the model with it at several
weights on OSV-5M test photos and on your saved rounds, from embeddings you already have.

    python scripts/try_zero_shot_countries.py [--model models/geoguessr.pt] [--places 2000]

A few minutes on a laptop CPU. A weight only helps if it beats weight 0 by more than about
twice its ± (the standard error of the difference).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from geoguessr_ai.geo import geoguessr_score, haversine_km
from geoguessr_ai.knowledge.countries import country_at, country_codes
from geoguessr_ai.knowledge.evidence import Evidence
from geoguessr_ai.model.head import Checkpoint
from geoguessr_ai.model.predictor import guess_from_embeddings
from geoguessr_ai.model.rounds import RoundEmbeddings
from geoguessr_ai.model.train import load_embeddings

EMBEDDINGS = Path("data/embeddings/openai__clip-vit-base-patch32")
WEIGHTS = (0.0, 0.25, 0.5, 1.0)
LOGIT_SCALE = 100.0  # CLIP's learned temperature
RENAMES = {
    "Congo - Kinshasa": "the Democratic Republic of the Congo",
    "Congo - Brazzaville": "the Republic of the Congo",
    "Palestinian Territories": "Palestine",
    "United States": "the United States",
    "United Kingdom": "the United Kingdom",
}

Place = tuple[np.ndarray, float, float]  # embeddings of every crop, latitude, longitude


def country_name(code: str) -> str:
    import langcodes

    if not code:  # the slot for places outside every country's borders
        return "the open sea"
    name = langcodes.Language.make(territory=code).territory_name()
    name = re.sub(r" \(.*\)| SAR China", "", name)  # Myanmar (Burma), Hong Kong SAR China
    name = name.replace(" & ", " and ").replace("St. ", "Saint ")
    return RENAMES.get(name, name)


@torch.inference_mode()
def country_prompts(backbone: str) -> np.ndarray:
    """CLIP text embeddings of one prompt per country, in :func:`country_codes` order."""
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = CLIPTextModelWithProjection.from_pretrained(backbone).eval()
    prompts = [f"A Street View photo taken in {country_name(c)}." for c in country_codes()]
    embeds = model(**tokenizer(prompts, padding=True, return_tensors="pt")).text_embeds
    return torch.nn.functional.normalize(embeds, dim=-1).numpy()


def country_probs(crops: np.ndarray, prompts: np.ndarray) -> np.ndarray:
    """How likely CLIP finds each country, every crop voting."""
    logits = LOGIT_SCALE * crops @ prompts.T
    log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    probs = np.exp(log_probs.mean(axis=0))
    return probs / probs.sum()


def score(checkpoint: Checkpoint, places: list[Place], prompts: np.ndarray):
    codes = country_codes()
    named, scores = 0, {weight: [] for weight in WEIGHTS}
    for crops, lat, lon in places:
        probs = country_probs(crops, prompts)
        named += codes[int(probs.argmax())] == country_at(lat, lon)
        for weight in WEIGHTS:
            evidence = Evidence(countries=(probs / probs.max()) ** weight)
            guess = guess_from_embeddings(checkpoint, crops, evidence=evidence)
            scores[weight].append(geoguessr_score(haversine_km(guess.lat, guess.lon, lat, lon)))
    return named / len(places), {weight: np.array(s) for weight, s in scores.items()}


def report(title: str, places: list[Place], checkpoint: Checkpoint, prompts: np.ndarray) -> None:
    named, scores = score(checkpoint, places, prompts)
    print(f"\n{title}: {len(places)} places. CLIP alone names the country for {named:.0%}.")
    for weight, s in scores.items():
        gain = s - scores[0.0]
        error = gain.std(ddof=1) / np.sqrt(len(gain)) if len(gain) > 1 else 0.0
        change = f"{gain.mean():+,.0f} ± {error:,.0f}"
        print(f"  weight {weight:<4}: mean score {s.mean():6,.0f}  ({change})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("models/geoguessr.pt"))
    parser.add_argument("--test", type=Path, default=EMBEDDINGS / "osv5m-test-04.npz")
    parser.add_argument("--rounds", type=Path, default=EMBEDDINGS / "openguessr-rounds.npz")
    parser.add_argument("--places", type=int, default=2000, help="test photos to score")
    args = parser.parse_args()

    checkpoint = Checkpoint.load(args.model)
    prompts = country_prompts(checkpoint.backbone)

    x, lat, lon, backbone = load_embeddings([args.test])
    if backbone != checkpoint.backbone:
        raise SystemExit(f"{args.test} is from {backbone}, the model from {checkpoint.backbone}")
    pick = np.random.default_rng(0).permutation(len(x))[: args.places]
    photos = [(np.asarray(x[i : i + 1], np.float32), float(lat[i]), float(lon[i])) for i in pick]
    report("OSV-5M test photos", photos, checkpoint, prompts)

    if args.rounds.exists():
        rounds = RoundEmbeddings.load(args.rounds)
        places = []
        for round_id in rounds.round_ids:
            mask = rounds.groups == round_id
            crops = np.asarray(rounds.embeddings[mask], np.float32)
            places.append((crops, float(rounds.lat[mask][0]), float(rounds.lon[mask][0])))
        report(
            "Your saved rounds (the model may have learned from some)", places, checkpoint, prompts
        )


if __name__ == "__main__":
    main()
