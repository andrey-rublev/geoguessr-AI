"""Read the signs in your saved rounds at several detection sizes, to see whether reading at a
larger size finds signs the default misses, and what it costs per round.

    python scripts/compare_ocr_sizes.py [runs] [--sizes 768 1024 1405] [--rounds 30]

A size is the width text is looked for at; views are never enlarged, so a size at or above a
saved view's width (1405 pixels on the screen this was written for) reads it at full size.
Takes a few minutes for every 10 rounds per size on a laptop CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from geoguessr_ai import ocr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="?", default=Path("runs"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[768, ocr.DETECT_WIDTH, 1405])
    parser.add_argument("--rounds", type=int, default=30, help="the most recent rounds to read")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    folders = sorted(args.runs.glob("*/round_*"))[-args.rounds :]
    rounds = [[Image.open(p).convert("RGB") for p in sorted(f.glob("view_*.jpg"))] for f in folders]
    reader = ocr.SignReader()
    for size in args.sizes:
        ocr.DETECT_WIDTH = size
        found, start = {}, time.perf_counter()
        for folder, views in zip(folders, rounds, strict=True):
            if lines := reader.read(views):
                found[f"{folder.parent.name}/{folder.name}"] = [line.text for line in lines]
        seconds = (time.perf_counter() - start) / max(len(folders), 1)
        print(f"\ndetect size {size}: text in {len(found)} of {len(folders)} rounds, ", end="")
        print(f"{seconds:.1f} s a round")
        for name, texts in found.items():
            print(f"  {name}: {texts}")


if __name__ == "__main__":
    main()
