"""Build the list of towns that ships in ``geoguessr_ai/knowledge/towns.txt``: each town's name,
without accents and in lowercase, and where it is, for telling the scale of a result map by the
towns it names (see :mod:`geoguessr_ai.result`).

A town's place is the middle of its OpenStreetView-5M photos (CC-BY-SA 4.0). Only rerun it to
update the list:

    python scripts/build_towns.py [data/osv5m/train.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from geoguessr_ai.knowledge.places import place_key

SOURCE = Path("data/osv5m/train.csv")
OUT = Path(__file__).resolve().parents[1] / "src" / "geoguessr_ai" / "knowledge" / "towns.txt"
MIN_PHOTOS = 5
"""Fewer may be a stray photo, or a name for one farm."""
MAX_SPREAD_DEG = 0.25
"""A name whose photos spread further than this, in degrees, is a district, not a point."""
CHUNK = 500_000


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE
    columns = ["latitude", "longitude", "city", "unique_city"]
    totals = None
    for chunk in pd.read_csv(source, usecols=columns, chunksize=CHUNK, dtype={"city": str}):
        chunk = chunk.dropna(subset=["city", "unique_city"])
        chunk["lat2"], chunk["lon2"] = chunk.latitude**2, chunk.longitude**2
        chunk["n"] = 1
        summed = chunk.groupby(["unique_city", "city"], sort=False)[
            ["latitude", "longitude", "lat2", "lon2", "n"]
        ].sum()
        totals = summed if totals is None else totals.add(summed, fill_value=0)
        print(f"{int(totals.n.sum()):,} photos, {len(totals):,} towns", flush=True)
    lat, lon = totals.latitude / totals.n, totals.longitude / totals.n
    spread = (totals.lat2 / totals.n - lat**2).clip(0) + (totals.lon2 / totals.n - lon**2).clip(0)
    kept = (totals.n >= MIN_PHOTOS) & (spread**0.5 <= MAX_SPREAD_DEG)
    towns = sorted(
        {
            (place_key(city), round(float(la), 2), round(float(lo), 2))
            for (_, city), la, lo in zip(totals.index[kept], lat[kept], lon[kept], strict=True)
            if place_key(city)
        }
    )
    OUT.write_text(
        "".join(f"{name}\t{la}\t{lo}\n" for name, la, lo in towns), encoding="utf-8", newline="\n"
    )
    print(f"{len(towns):,} towns in {OUT}, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
