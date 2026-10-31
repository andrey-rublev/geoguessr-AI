"""Draw the desktop app's icon: the mark a navigator plots a position with, a dot in a ring.

python scripts/build_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "src" / "geoguessr_ai" / "desktop.ico"
SEA, PLOT, INK = "#d5e6ec", "#a8127a", "#0f2c38"
SIDE = 1024
SIZES = [16, 24, 32, 48, 64, 128, 256]


def main() -> None:
    image = Image.new("RGBA", (SIDE, SIDE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, SIDE - 1, SIDE - 1), radius=SIDE // 5, fill=SEA)
    middle = SIDE // 2
    ring = SIDE * 5 // 16
    draw.ellipse(
        (middle - ring, middle - ring, middle + ring, middle + ring),
        outline=PLOT,
        width=SIDE * 7 // 64,
    )
    dot = SIDE // 10
    draw.ellipse((middle - dot, middle - dot, middle + dot, middle + dot), fill=INK)
    image.save(OUT, sizes=[(size, size) for size in SIZES])
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
