"""Pack the model and the photos it votes with for a release, for `geoguessr-ai fetch-model`:

    python scripts/pack_model.py geoguessr-model-1.zip
    gh release create model-1 geoguessr-model-1.zip --title "The model" --notes "..."

Then put what it prints in src/geoguessr_ai/fetch.py. Run from the repo. Only embeddings go in, no
pictures: what the model learned from, not what it saw.
"""

import argparse
import hashlib
import zipfile
from pathlib import Path

FILES = ["models/geoguessr.pt"]
EMBEDDINGS = Path("data/embeddings/geolocal__StreetCLIP")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", type=Path)
args = parser.parse_args()

names = [*FILES, *(path.as_posix() for path in sorted(EMBEDDINGS.glob("*.npz")))]
with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as pack:  # they hardly compress
    for name in names:
        print(f"Adding {name}")
        pack.write(name)

digest = hashlib.sha256()
with open(args.out, "rb") as file:
    while chunk := file.read(1 << 20):
        digest.update(chunk)
print(f'PACK_SHA256 = "{digest.hexdigest()}"')
print(f"PACK_BYTES = {args.out.stat().st_size}")
