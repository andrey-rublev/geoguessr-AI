"""Build the list of place names that ships in ``geoguessr_ai/knowledge/places.txt``.

Takes every town and city of at least 15,000 people from GeoNames (CC BY 4.0) by its usual
and plain-letter names, plus its names in its country's own script where OCR can read that
script and it isn't Latin, like Москва or 서울. Only rerun it to update the list:

    python scripts/build_places.py [cities15000.zip]
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from geoguessr_ai.knowledge.facts import countries
from geoguessr_ai.knowledge.places import place_key
from geoguessr_ai.ocr import RECOGNISERS, script_of

SOURCE = "https://download.geonames.org/export/dump/cities15000.zip"
OUT = Path(__file__).resolve().parents[1] / "src" / "geoguessr_ai" / "knowledge" / "places.txt"
READABLE = set(RECOGNISERS.values()) | {"kana"}


def native(name: str, scripts: set[str]) -> bool:
    """Written only in one of the country's own scripts, other than Latin."""
    found = {script_of(c) for c in name} - {None}
    return bool(found) and found <= scripts


def main() -> None:
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1]).read_bytes()
    else:
        with urllib.request.urlopen(SOURCE) as response:
            raw = response.read()
    rows = zipfile.ZipFile(io.BytesIO(raw)).read("cities15000.txt").decode("utf-8")
    known = countries()
    names: dict[str, set[str]] = {}
    for row in rows.splitlines():
        fields = row.split("\t")
        name, plain, others, code = fields[1], fields[2], fields[3], fields[8]
        if code not in known:
            continue
        scripts = (set(known[code].scripts) & READABLE) - {"latin"}
        spellings = {name, plain} | {n for n in others.split(",") if scripts and native(n, scripts)}
        for spelling in spellings:
            if key := place_key(spelling):
                names.setdefault(key, set()).add(code)
    lines = [f"{name}\t{','.join(sorted(codes))}" for name, codes in sorted(names.items())]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines):,} names -> {OUT}")


if __name__ == "__main__":
    main()
