"""Pack the model and the photos it votes with for a release, for `geoguessr-ai fetch-model`:

    python scripts/pack_model.py geoguessr-model-1.zip
    gh release create model-1 geoguessr-model-1.zip --title "The model" --notes "..."

Then put what it prints in src/geoguessr_ai/fetch.py. Run from the repo. Only embeddings go in, no
pictures: what the model learned from, not what it saw. The rounds played go in as places, one to a
round, under another name than a player's own rounds' file, which training rewrites: there, each
round's crops take 15 times the room, and the rounds that a player hasn't got would be dropped.

The model goes in last, as what's unpacked last: `fetch-model` takes it being there for all done.
"""

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np

from geoguessr_ai.model.rounds import ROUNDS_FILE, RoundEmbeddings

MODEL = "models/geoguessr.pt"
EMBEDDINGS = Path("data/embeddings/geolocal__StreetCLIP")
PLAYED = "openguessr-played.npz"


def played_as_places() -> bytes:
    """Each round played once, its crops averaged, in the form of embedded photos."""
    rounds = RoundEmbeddings.load(EMBEDDINGS / ROUNDS_FILE)
    ids, first, which = np.unique(rounds.groups, return_index=True, return_inverse=True)
    mean = np.zeros((len(ids), rounds.embeddings.shape[1]), np.float32)
    np.add.at(mean, which, rounds.embeddings)
    file = io.BytesIO()
    np.savez(
        file,
        ids=ids,
        lat=rounds.lat[first].astype(np.float32),
        lon=rounds.lon[first].astype(np.float32),
        embeddings=(mean / np.linalg.norm(mean, axis=1, keepdims=True)).astype(np.float16),
        backbone=np.array(rounds.backbone),
    )
    return file.getvalue()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", type=Path)
args = parser.parse_args()

with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as pack:  # they hardly compress
    for path in sorted(EMBEDDINGS.glob("*.npz")):
        if path.name != ROUNDS_FILE:
            print(f"Adding {path.as_posix()}")
            pack.write(path, path.as_posix())
    played = played_as_places()
    print(f"Adding the rounds played, {len(played) / 1e6:.0f} MB")
    pack.writestr((EMBEDDINGS / PLAYED).as_posix(), played)
    print(f"Adding {MODEL}")
    pack.write(MODEL)

digest = hashlib.sha256()
with open(args.out, "rb") as file:
    while chunk := file.read(1 << 20):
        digest.update(chunk)
print(f'PACK_SHA256 = "{digest.hexdigest()}"')
print(f"PACK_BYTES = {args.out.stat().st_size}")
