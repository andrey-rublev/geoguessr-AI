import numpy as np

from geoguessr_ai.knowledge.text import TextLine
from geoguessr_ai.ocr import _pad_width, pick_reading, script_of


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
