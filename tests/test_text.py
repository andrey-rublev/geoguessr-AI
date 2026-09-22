import numpy as np

from geoguessr_ai.knowledge.countries import country_codes
from geoguessr_ai.knowledge.text import TextLine, text_clues


def ratio(clues, a, b):
    codes = country_codes()
    return clues.likelihood[codes.index(a)] / clues.likelihood[codes.index(b)]


def test_hangul_points_to_korea():
    clues = text_clues([TextLine("서울역", "hangul", 0.95)])
    assert ratio(clues, "KR", "JP") > 20 and "Hangul" in clues.notes[0]


def test_kana_means_japan_and_chinese_characters_mean_places_that_write_them():
    assert ratio(text_clues([TextLine("とうきょう", "kana", 0.9)]), "JP", "CN") > 20
    han = text_clues([TextLine("海鮮燒鵝食家", "han", 0.9)])  # a shop, naming no town
    assert ratio(han, "TW", "KR") > 20 and ratio(han, "TW", "JP") == 1  # 岡崎酒店 is Japanese
    named = text_clues([TextLine("台北車站", "han", 0.9)])  # Taipei, so not Japan after all
    assert ratio(named, "TW", "JP") >= 3


def test_portuguese_street_words_point_to_brazil_and_portugal():
    clues = text_clues(
        [TextLine("Rua São João", "latin", 0.9), TextLine("Farmácia Popular", "latin", 0.9)]
    )
    assert ratio(clues, "BR", "ES") >= 3 and ratio(clues, "PT", "BR") == 1


def test_words_read_without_their_accents_still_count():
    capitals = text_clues([TextLine("PRACA DA SE", "latin", 0.9)])
    assert ratio(capitals, "BR", "ES") >= 2 and capitals.notes == ["Portuguese: praca"]
    pharmacy = text_clues([TextLine("FARMACIA", "latin", 0.9)])  # farmácia, or Spanish farmacia
    assert ratio(pharmacy, "BR", "FR") >= 2 and ratio(pharmacy, "BR", "MX") == 1


def test_words_cut_off_at_the_edge_of_a_sign_still_count():
    cut = text_clues([TextLine("DE DESCONT", "latin", 0.9)])  # desconto, past the frame
    assert ratio(cut, "BR", "MX") >= 2 and cut.notes == ["Portuguese: descont…"]
    start = text_clues([TextLine("ravessa Cinco", "latin", 0.9)])  # travessa, cut the other way
    assert ratio(start, "BR", "MX") >= 2 and start.notes == ["Portuguese: …ravessa"]
    assert not text_clues([TextLine("BIG DESCONT SALE", "latin", 0.9)]).notes  # a whole word
    assert not text_clues([TextLine("ANCIA", "latin", 0.9)]).notes


def test_a_word_inside_a_longer_one_read_nearby_is_the_same_sign_cut_off():
    twice = [TextLine("Calle Benito Juare", "latin", 0.94), TextLine("alle", "latin", 0.96)]
    assert text_clues(twice).notes == ["Spanish: calle"]  # not a Danish allé as well
    assert text_clues([TextLine("Solvej Allé 12", "latin", 0.9)]).notes == ["Danish: allé"]


def test_shop_signs_in_swahili_and_spanish():
    kenya = text_clues([TextLine("LAST CUT KINYOZI", "latin", 0.9)])  # a barber in Nairobi
    assert ratio(kenya, "KE", "BD") >= 2 and kenya.notes == ["Swahili: kinyozi"]
    mexico = text_clues([TextLine("Vulcanizadora y Llantera", "latin", 0.9)])
    assert ratio(mexico, "MX", "BR") >= 3


def test_indonesian_road_label():
    clues = text_clues([TextLine("Jl. Lintas Selatan", "latin", 0.9)])
    assert ratio(clues, "ID", "TH") >= 2


def test_road_words_written_onto_the_end_of_the_name():
    finnish = text_clues([TextLine("Piipsannevantie", "latin", 0.9)])
    assert ratio(finnish, "FI", "FR") >= 2 and finnish.notes == ["Finnish: …tie"]
    assert ratio(text_clues([TextLine("Eindstraat", "latin", 0.9)]), "NL", "FR") >= 2
    assert ratio(text_clues([TextLine("Ivalontie", "latin", 0.9)]), "FI", "SE") >= 2
    run_on = text_clues([TextLine("Bárðardalsvegurvest", "latin", 0.9)])  # words run together
    assert ratio(run_on, "IS", "AR") >= 4 and "Icelandic: …vegur" in run_on.notes


def test_a_short_road_ending_needs_a_name_before_it():
    for word in ("SORTIE", "PARTIE"):  # French, not a Finnish road ending in -tie
        notes = text_clues([TextLine(word, "latin", 0.9)]).notes
        assert not any("Finnish" in note for note in notes)


def test_a_road_word_before_a_preposition_isnt_a_web_address():
    for words in ("Ctra. de la Rabassa", "Chem. de Montigny", "CONSTRL . la Cierva"):
        assert not text_clues([TextLine(words, "latin", 0.9)]).notes, words
    spread = [TextLine("www.", "latin", 0.9), TextLine("lojas.com.br", "latin", 0.9)]
    assert text_clues(spread).notes == ["web domain .br"]  # an address split across lines


def test_state_letters_brazil_shares_with_the_united_states():
    shared = text_clues([TextLine("MS-465", "latin", 0.9)])  # Mississippi, or Mato Grosso do Sul
    assert shared.notes == ["state road: ms-465"] and ratio(shared, "US", "AR") >= 3
    assert ratio(shared, "BR", "US") == 1
    assert ratio(text_clues([TextLine("RSC-473", "latin", 0.9)]), "BR", "US") >= 3


def test_latin_misread_as_cyrillic_doesnt_count():
    misread = [TextLine("Francisco Bocanegra", "latin", 0.98), TextLine("со", "cyrillic", 0.91)]
    assert not any("Cyrillic" in note for note in text_clues(misread).notes)
    real = text_clues([TextLine("К СТОЛУ!", "cyrillic", 0.9)])  # Л couldn't be Latin
    assert ratio(real, "RU", "FR") > 5


def test_a_short_misreading_isnt_a_script_and_greek_yields_to_cyrillic():
    assert not text_clues([TextLine("по", "cyrillic", 0.99)]).notes  # Spanish no, read as Cyrillic
    assert not text_clues([TextLine("Μ ριό", "greek", 0.89)]).notes  # four letters of nothing
    both = [TextLine("ΣΥΠΕΡΜΑΡΚΕΤ", "greek", 0.99), TextLine("ул.Адмиральског", "cyrillic", 0.93)]
    assert text_clues(both).notes == ["Cyrillic: ул.Адмиральског"]  # Russian СУПЕРМАРКЕТ
    assert text_clues([TextLine("Ευδόξου", "greek", 0.98)]).notes == ["Greek: Ευδόξου"]


def test_ukrainian_letters_tell_ukraine_from_russia():
    clues = text_clues([TextLine("вулиця Київська", "cyrillic", 0.9)])
    assert ratio(clues, "UA", "RU") > 2 and ratio(clues, "RU", "FR") > 5


def test_web_domains_and_phone_codes():
    brazil = text_clues([TextLine("www.lojas.com.br", "latin", 0.9)])
    assert ratio(brazil, "BR", "US") >= 10
    assert text_clues([TextLine("autohaus-mueller.de", "latin", 0.9)]).notes == ["web domain .de"]
    russia = text_clues([TextLine("тел. +7 (495) 123-45-67", "cyrillic", 0.9)])
    assert ratio(russia, "RU", "DE") >= 10 and ratio(russia, "KZ", "RU") == 1
    assert not text_clues([TextLine("St.No 5", "latin", 0.9)]).notes  # not a domain


def test_local_phone_numbers_without_a_country_code():
    brazil = text_clues([TextLine("99983.2915", "latin", 0.9)])  # a Brazilian mobile
    assert ratio(brazil, "BR", "MX") >= 3 and brazil.notes == ["phone number 99983.2915"]
    assert ratio(text_clues([TextLine("Tel (415) 555-0132", "latin", 0.9)]), "US", "GB") >= 3
    assert ratio(text_clues([TextLine("01 42 68 53 00", "latin", 0.9)]), "FR", "ES") >= 3
    assert ratio(text_clues([TextLine("8 (495) 123-45-67", "latin", 0.9)]), "RU", "PL") >= 3
    both = text_clues([TextLine("+55 (11) 99983-2915", "latin", 0.9)])
    assert both.notes == ["phone number +55"]  # the same number counts once
    price = text_clues([TextLine("2024-05-12 R$ 1.299,90 KM 12", "latin", 0.9)])
    assert price.notes == ["price in reais: r$ 1"]  # no phone number in a date or a price


def test_prices_speeds_postcodes_and_road_numbers():
    for text, here, elsewhere in (
        ("Promoção R$ 9,99", "BR", "PT"),
        ("Pizza 25 zł", "PL", "CZ"),
        ("SPEED LIMIT 35 MPH", "US", "CA"),
        ("London SW1A 1AA", "GB", "US"),
        ("Toronto ON M5V 3L9", "CA", "US"),
        ("CEP 01310-100", "BR", "AR"),
        ("BR-116", "BR", "AR"),
        ("RSC-473", "BR", "AR"),
        ("I-95 North", "US", "CA"),
        ("DN1 Brasov", "RO", "BG"),
        ("Rs. 250", "IN", "BD"),
        ("Ksh 100", "KE", "NG"),
        ("Rp 15.000", "ID", "MY"),
        ("SH58", "IN", "BD"),
        ("Co Rd 158", "US", "CA"),
        ("Range Rd 20", "CA", "GB"),
        ("S Triple X Rd", "US", "GB"),
        ("14th St", "US", "GB"),
        ("Ulitsa Gagarina", "RU", "PL"),
        ("Motiram Marg", "NP", "BD"),
        ("Cra. 71c", "CO", "PE"),
        ("DW785", "PL", "CZ"),
        ("RP51", "AR", "CL"),
    ):
        clues = text_clues([TextLine(text, "latin", 0.9)])
        assert ratio(clues, here, elsewhere) >= 3, text
    assert text_clues([TextLine("1500 km", "latin", 0.9)]).notes == []
    two_signs = [TextLine("811.SH", "latin", 0.9), TextLine("247.Sk.", "latin", 0.98)]
    assert text_clues(two_signs).notes == ["Turkish: sk."]  # not the state highway SH 247
    rp51 = text_clues([TextLine("RP51", "latin", 0.9)]).notes  # a road, not a rupiah price
    assert not any("rupiah" in note for note in rp51)


def test_towns_on_signs_but_not_streets_and_shops_named_after_them():
    sign = text_clues([TextLine("Culiacán 210 km", "latin", 0.9)])
    assert ratio(sign, "MX", "US") >= 3 and sign.notes == ["town culiacan (MX)"]
    named = text_clues([TextLine("Farmácia Curitiba", "latin", 0.9)])
    assert not any(note.startswith("town") for note in named.notes)
    shortened = text_clues([TextLine("Tv. Pinheiro Chagas", "latin", 0.9)])  # travessa
    assert shortened.notes == ["Portuguese: tv."]


def test_regional_brands():
    clues = text_clues([TextLine("PEMEX", "latin", 0.95)])
    assert ratio(clues, "MX", "US") >= 3 and clues.notes == ["brand pemex"]


def test_brands_run_together_with_other_words():
    brazil = text_clues([TextLine("OBOTICARIO", "latin", 0.9)])  # O Boticário, read as one word
    assert ratio(brazil, "BR", "MX") >= 3 and brazil.notes == ["brand boticario"]
    assert ratio(text_clues([TextLine("M-PESA", "latin", 0.9)]), "KE", "BD") >= 3
    assert text_clues([TextLine("PETRONAS", "latin", 0.9)]).notes == ["brand petronas"]
    assert not text_clues([TextLine("COPTTER", "latin", 0.9)]).notes  # short names stand alone


def test_english_barely_counts_and_no_text_changes_nothing():
    english = text_clues([TextLine("Main Street Parking", "latin", 0.9)])
    assert 1 < ratio(english, "US", "FR") < 2
    nothing = text_clues([])
    assert np.all(nothing.likelihood == 1) and nothing.notes == []
