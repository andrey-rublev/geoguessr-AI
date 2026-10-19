import numpy as np

from geoguessr_ai.knowledge.text import TextLine
from geoguessr_ai.ocr import SignReader, _pad_width, _steep, pick_reading, script_of


def test_lines_are_padded_to_a_few_widths():
    crop = np.full((20, 130, 3), 40, np.uint8)
    padded = _pad_width(crop)
    assert padded.shape == (20, 160, 3) and (padded[:, :130] == crop).all()
    assert _pad_width(padded) is padded


def test_script_of_letters():
    assert [script_of(c) for c in "aЖ가ア中α5"] == [
        "latin",
        "cyrillic",
        "hangul",
        "kana",
        "han",
        "greek",
        None,
    ]


def test_keeps_the_most_confident_reading_in_its_own_script():
    line = pick_reading([("latin", "Soul Yeok", 0.41), ("hangul", "서울역", 0.97)], 0.8)
    assert line == TextLine("서울역", "hangul", 0.97)


def test_latin_text_read_by_another_script_is_still_latin():
    line = pick_reading([("latin", "Rua Augusta", 0.93), ("hangul", "Rua Augusta", 0.95)], 0.8)
    assert line == TextLine("Rua Augusta", "latin", 0.95)


def test_kana_tells_japanese_from_chinese():
    assert pick_reading([("han", "東京駅はこちら", 0.9)], 0.8).script == "kana"
    assert pick_reading([("han", "台北車站", 0.9)], 0.8).script == "han"


def test_drops_unsure_letterless_or_mixed_up_readings():
    assert pick_reading([("latin", "Rua", 0.5)], 0.8) is None
    assert pick_reading([("hangul", "5", 0.99)], 0.8) is None
    assert pick_reading([("latin", "+44 20 7946 0958", 0.9)], 0.8).text.startswith("+44")


def test_text_running_up_or_down_the_picture_is_steep():
    across = np.float32([[0, 0], [100, 10], [100, 40], [0, 30]])
    up = np.float32([[0, 100], [10, 0], [40, 0], [30, 100]])
    assert not _steep(across) and _steep(up) and _steep(up[[1, 2, 3, 0]])


BIG = np.array([[100, 100], [600, 100], [600, 200], [100, 200]], dtype=np.float32)
SMALL = np.array([[1500, 300], [1560, 300], [1560, 312], [1500, 312]], dtype=np.float32)


class FakeDetector:
    """A 2,000-wide picture with a big sign at the left and a small one at the right, which a
    copy shrunk to 1,024 shows too small to find: answers the shrunk copy, then each half."""

    def __init__(self):
        self.widths = []

    def __call__(self, bgr):
        w = bgr.shape[1]
        self.widths.append(w)
        boxes = {
            1: [BIG * 1024 / 2000],  # the shrunk copy
            2: [BIG],  # the left half, from 0, at full size: the big sign again
            3: [SMALL - np.array([900, 0], dtype=np.float32)],  # the right half, from 900
        }[len(self.widths)]
        return type("Found", (), {"boxes": boxes, "scores": [0.9] * len(boxes)})()


def test_small_writing_is_looked_for_again_in_each_half_at_full_size():
    reader = SignReader.__new__(SignReader)
    reader._detect, reader.close = FakeDetector(), True
    picture = np.zeros((800, 2000, 3), np.uint8)

    boxes = reader._find_lines(picture, 0.6)

    assert reader._detect.widths == [1024, 1100, 1100]  # shrunk, then each half at its own size
    centres = sorted(tuple(np.round(b.mean(axis=0))) for b in boxes)
    assert centres == [(350.0, 150.0), (1530.0, 306.0)]  # the big sign once, the small one too
    reader.close, reader._detect = False, FakeDetector()
    assert len(reader._find_lines(picture, 0.6)) == 1 and reader._detect.widths == [1024]
