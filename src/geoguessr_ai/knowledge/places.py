"""Towns and cities named on signs: the places a direction sign points to are usually nearby.

The names are GeoNames' towns and cities of at least 15,000 people (see
``scripts/build_places.py``), without accents and in lowercase, with each name's countries.
A name found in some text counts only if it is long enough not to be a chance match, isn't
also an everyday word, and doesn't belong to too many countries to say anything. Nor does one
straight after a word like "street" or "pharmacy": Rua São João and Farmácia São Paulo are
named after places, not signs to them.
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
# Place names that are also everyday words on signs, in English or the languages whose street
# words the bot knows, or common first names.
EVERYDAY = set(
    """
    about after alameda alexandria amazonas america arena aurora avenida bahia banco bella bonito
    bridge brighton buena bueno campo canal canto carmen casino castle catalina central centro
    chester church clara colon colonia concordia constitucion cristal cruces del dolores dorado
    eagle esperanza estrella fatima florida fortuna frontera garden general grande granada green
    guadalupe hampton heights hills hotel independence industrial isabel jardim jardin jesus
    kingston lagos laguna lakewood libertad liberty lincoln linden lourdes madison maria marina
    market mercedes middleton milton mirador mission mobile monte montana monterey mountain
    nazareth newport nueva nuevo olympia orange oriental palma palmas paradise paraiso park
    patria pilar plata plaza porto posadas pozos premier princeton progreso providence pueblo
    puerto reading regina reforma remedios republica richmond rio rivera rosario rosas royal
    salem salud salvador santa santana santiago santo santos sao senhor sol springfield split
    station stuart sucre sunrise sunset tabernacle terra torres trinidad triunfo union valle
    valley vega venus victoria villa vista vitoria washington water wellington westminster
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


def places_in(
    lines: Iterable[str], naming_words: Collection[str] = ()
) -> dict[str, tuple[str, ...]]:
    """Place names in lines of text, with their countries, except straight after one of
    ``naming_words`` (as :func:`place_key`). A name inside a longer one found, like Rio Grande
    in Rio Grande do Sul, doesn't count again."""
    known = places()
    found = {}
    for line in lines:
        words = place_key(line).split()
        for start in range(len(words)):
            if start and words[start - 1] in naming_words:
                continue
            for end in range(start + 1, min(start + MAX_WORDS, len(words)) + 1):
                name = " ".join(words[start:end])
                if name in known:
                    found[name] = known[name]
        joined = "".join(words)
        found.update((name, known[name]) for name in _unspaced() if name in joined)
    return {
        name: countries
        for name, countries in found.items()
        if not any(other != name and name in other for other in found)
    }
