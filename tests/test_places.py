from geoguessr_ai.knowledge.places import place_key, places, places_in


def test_place_keys_drop_accents_case_and_punctuation():
    assert place_key("  Culiacán-ROSALES, Sin.") == "culiacan rosales sin"
    assert place_key("Αθήνα") == "αθηνα"


def test_finds_towns_in_any_script():
    assert places_in(["Mazatlán - Culiacán 210 km"]) == {"mazatlan": ("MX",), "culiacan": ("MX",)}
    assert places_in(["Москва 12 км"]) == {"москва": ("RU",)}
    assert places_in(["東京駅"]) == {"東京": ("JP",)}
    assert places_in(["서울역"]) == {"서울": ("KR",)}


def test_ignores_everyday_words_short_names_and_names_after_street_words():
    assert "victoria" not in places() and "lima" not in places()
    assert places_in(["Hotel Victoria", "Main Street Parking"]) == {}
    assert places_in(["JEEVAN TOYOTA", "NW 191st Terrace"]) == {}  # a make and a road word


def test_a_road_or_feature_word_after_a_name_makes_it_one():
    assert places_in(["Lucas Paddock Rd"]) == {}  # a road in Queensland, not Lucas in Brazil
    assert places_in(["Monroe Lake"]) == {} and places_in(["Monroe Lak"]) == {}  # cut off
    assert places_in(["Culiacán 210 km"]) == {"culiacan": ("MX",)}  # a sign pointing there
    assert places_in(["Rua Curitiba"], naming_words={"rua"}) == {}
    assert places_in(["Curitiba 12"], naming_words={"rua"}) == {"curitiba": ("BR",)}


def test_a_longer_name_found_hides_the_shorter_one_inside_it():
    found = places_in(["Buenos Aires"])
    assert "buenos aires" in found and "aires" not in found
