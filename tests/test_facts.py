from geoguessr_ai.knowledge.countries import country_codes, country_index
from geoguessr_ai.knowledge.facts import COVERAGE_WEIGHTS, countries, per_country


def test_every_country_on_the_map_has_facts():
    table = countries()
    assert set(country_codes()[1:]) <= set(table)
    assert all(country.coverage in COVERAGE_WEIGHTS for country in table.values())


def test_facts_players_rely_on():
    table = countries()
    assert table["JP"].drives_left and table["GB"].drives_left and not table["US"].drives_left
    assert table["US"].calling_codes == ("1",) and table["GB"].domains == ("uk",)
    assert table["CN"].coverage == "none" and table["KR"].coverage == "full"
    assert "hangul" in table["KR"].scripts and "cyrillic" in table["RU"].scripts
    assert "pt" in table["BR"].languages


def test_per_country_lines_up_with_the_map():
    left = per_country(lambda country: country.drives_left)
    assert left[country_index(35.68, 139.69)] == 1  # Tokyo
    assert left[country_index(48.85, 2.35)] == 0  # Paris
