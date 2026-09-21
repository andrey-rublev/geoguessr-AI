"""Towns and cities named on signs: the places a direction sign points to are usually nearby.

The names are GeoNames' towns and cities of at least 15,000 people (see
``scripts/build_places.py``), without accents and in lowercase, with each name's countries.
A name found in some text counts only if it is long enough not to be a chance match, isn't
also an everyday word, and doesn't belong to too many countries to say anything. Nor does one
with a word like "street" or "pharmacy" beside it: Rua São João, Farmácia São Paulo, Lucas
Paddock Rd and Monroe Lake are named after places, not signs to them. Which side that word
falls on depends on the language, so both are looked at.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Collection, Iterable
from importlib import resources

MIN_LETTERS = 5
"""Shorter names written in letters, like Lima or Bari, match too much by chance."""
MIN_CHARACTERS = 2
"""Names in Chinese characters, kana or Hangul, which aren't split into words, may be short."""
MAX_COUNTRIES = 4
"""A name shared by more countries than this, like San Jose, says too little."""
MAX_WORDS = 4
# English puts the road or feature word after the name, where no language's street words are
# looked for: Lucas Paddock Rd is a road in Queensland, not a sign to Lucas in Brazil.
FOLLOWING_ROAD_WORDS = set(
    """
    st street rd road ave avenue ln lane dr drive blvd boulevard hwy highway way ct court
    pl place ter terrace cir circle pkwy parkway trail crescent close row walk park paddock
    ranch farm estate garden gardens lake river creek stream valley beach bay falls hill hills
    mountain mount island point bridge crossing junction station airport school church
    hospital hotel store market mall
    """.split()
)
FOLLOWING_LETTERS = 3
"""Letters of a road word needed to count it when the frame cut it off, as in Monroe Lak."""
# Place names that are also everyday words on signs, in English or the languages whose street
# words the bot knows, people's names like Wilson or Lucas, road words like Terrace, or makes
# like Toyota: JEEVAN TOYOTA on an Indian sign used to point to Toyota in Japan.
EVERYDAY = set(
    """
    about adrian after alameda alexandria amazonas america anderson arena aurora avenida bahia
    banco bella bonito bridge brighton buena bueno burke campo canal canto carmen casino castle
    catalina central centro chandler chester church clara colon colonia concordia constitucion
    cristal cruces david del delta dixon dolores dorado eagle esperanza estrella fatima flores
    florida forest fortuna frontera garden general george granada grande green guadalupe hampton
    hayes heights hercules hills honda hotel imperial independence industrial isabel jackson
    jardim jardin jesus jupiter kingston lagos laguna lakewood libertad liberty lincoln linden
    lourdes lucas madison march marco maria marina market martin mercedes middleton milton mirador
    mission mobile monroe montana monte monterey mountain murphy mustang nazareth nelson newport
    nokia nueva nuevo olympia orange oriental palma palmas paradise paraiso park parker patria
    perry phoenix pilar plata plaza porto posadas pozos premier princeton progreso providence
    pueblo puerto ramon reading reforma regina remedios republica richmond rio rivera rosario
    rosas royal salem salud salvador santa santana santiago santo santos sao senhor sol split
    springfield springs station stuart sucre summit sunrise sunset superior tabernacle taylor
    terra terrace torres toyota trinidad triunfo union valle valley vega venus victoria villa
    vista vitoria walker washington water wellington westminster wilson
    """.split()
)
_NOT_LETTERS = re.compile(r"[\W\d_]+")
_UNSPACED = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3\u0e00-\u0e7f]")
"""Chinese characters, kana, Hangul and Thai, written without spaces between words."""


def place_key(text: str) -> str:
    """``Culiacán`` as ``culiacan``: lowercase, without accents, words split by single spaces."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(_NOT_LETTERS.split(unicodedata.normalize("NFC", plain))).strip()


@functools.lru_cache(maxsize=1)
def places() -> dict[str, tuple[str, ...]]:
    """Every place name worth looking for, as :func:`place_key`, with its countries."""
    text = (resources.files(__package__) / "places.txt").read_text(encoding="utf-8")
    found = {}
    for line in text.splitlines():
        name, codes = line.split("\t")
        countries = tuple(codes.split(","))
        if len(countries) <= MAX_COUNTRIES and _telling(name):
            found[name] = countries
    return found


def _telling(name: str) -> bool:
    if _UNSPACED.search(name):
        return len(name) >= MIN_CHARACTERS
    return sum(c.isalpha() for c in name) >= MIN_LETTERS and name not in EVERYDAY


@functools.lru_cache(maxsize=1)
def _unspaced() -> tuple[str, ...]:
    return tuple(name for name in places() if _UNSPACED.search(name))


def _road_follows(words: list[str], end: int) -> bool:
    """Whether the word after a name makes it a road or a feature, even cut off as Lak(e)."""
    if end >= len(words):
        return False
    following = words[end]
    if following in FOLLOWING_ROAD_WORDS:
        return True
    cut_off = end == len(words) - 1 and len(following) >= FOLLOWING_LETTERS
    return cut_off and any(word.startswith(following) for word in FOLLOWING_ROAD_WORDS)


def places_in(
    lines: Iterable[str], naming_words: Collection[str] = ()
) -> dict[str, tuple[str, ...]]:
    """Place names in lines of text, with their countries, except straight after one of
    ``naming_words`` (as :func:`place_key`) or straight before a road word. A name inside a
    longer one found, like Rio Grande in Rio Grande do Sul, doesn't count again."""
    known = places()
    found = {}
    for line in lines:
        words = place_key(line).split()
        for start in range(len(words)):
            if start and words[start - 1] in naming_words:
                continue
            for end in range(start + 1, min(start + MAX_WORDS, len(words)) + 1):
                name = " ".join(words[start:end])
                if name in known and not _road_follows(words, end):
                    found[name] = known[name]
        joined = "".join(words)
        found.update((name, known[name]) for name in _unspaced() if name in joined)
    return {
        name: countries
        for name, countries in found.items()
        if not any(other != name and name in other for other in found)
    }
