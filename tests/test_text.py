import numpy as np

from geoguessr_ai.knowledge.countries import country_codes
from geoguessr_ai.knowledge.text import TextLine, text_clues


def ratio(clues, a, b):
    codes = country_codes()
    return clues.likelihood[codes.index(a)] / clues.likelihood[codes.index(b)]


def test_hangul_points_to_korea():
    clues = text_clues([TextLine("서울역", "hangul", 0.95)])
    assert ratio(clues, "KR", "JP") > 20 and "Hangul" in clues.notes[0]


def test_kana_means_japan_and_chinese_characters_alone_mean_chinese_speaking_places():
    assert ratio(text_clues([TextLine("とうきょう", "kana", 0.9)]), "JP", "CN") > 20
    han = text_clues([TextLine("台北車站", "han", 0.9)])
    assert ratio(han, "TW", "KR") > 20 and ratio(han, "TW", "JP") > 2


def test_portuguese_street_words_point_to_brazil_and_portugal():
    clues = text_clues(
        [TextLine("Rua São João", "latin", 0.9), TextLine("Farmácia Popular", "latin", 0.9)]
    )
    assert ratio(clues, "BR", "ES") >= 3 and ratio(clues, "PT", "BR") == 1


def test_shop_signs_in_swahili_and_spanish():
    kenya = text_clues([TextLine("LAST CUT KINYOZI", "latin", 0.9)])  # a barber in Nairobi
    assert ratio(kenya, "KE", "BD") >= 2 and kenya.notes == ["Swahili: kinyozi"]
    mexico = text_clues([TextLine("Vulcanizadora y Llantera", "latin", 0.9)])
    assert ratio(mexico, "MX", "BR") >= 3


def test_indonesian_road_label():
    clues = text_clues([TextLine("Jl. Lintas Selatan", "latin", 0.9)])
    assert ratio(clues, "ID", "TH") >= 2


def test_ukrainian_letters_tell_ukraine_from_russia():
    clues = text_clues([TextLine("вулиця Київська", "cyrillic", 0.9)])
    assert ratio(clues, "UA", "RU") > 2 and ratio(clues, "RU", "FR") > 5


def test_web_domains_and_phone_codes():
    brazil = text_clues([TextLine("www.lojas.com.br", "latin", 0.9)])
    assert ratio(brazil, "BR", "US") >= 10
    russia = text_clues([TextLine("тел. +7 (495) 123-45-67", "cyrillic", 0.9)])
    assert ratio(russia, "RU", "DE") >= 10 and ratio(russia, "KZ", "RU") == 1
    assert not text_clues([TextLine("St.No 5", "latin", 0.9)]).notes  # not a domain


def test_regional_brands():
    clues = text_clues([TextLine("PEMEX", "latin", 0.95)])
    assert ratio(clues, "MX", "US") >= 3 and clues.notes == ["brand pemex"]


def test_english_barely_counts_and_no_text_changes_nothing():
    english = text_clues([TextLine("Main Street Parking", "latin", 0.9)])
    assert 1 < ratio(english, "US", "FR") < 2
    nothing = text_clues([])
    assert np.all(nothing.likelihood == 1) and nothing.notes == []
