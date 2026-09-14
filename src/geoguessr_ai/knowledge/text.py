"""What the writing on screen gives away: scripts, languages, web domains and phone numbers.

Lines of text read off the screen become a likelihood for every country. Each kind of clue
counts once however many signs show it, and no clue rules a country out completely: text can
be misread, and brands, tourists and road signs cross borders.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .countries import country_codes
from .facts import Country, countries, per_country

SCRIPT_FLOOR = 0.03
"""Likelihood for a country whose signs don't use a script that was clearly read."""
LETTER_FLOOR = 0.3
"""For a country that doesn't use a telling letter that was read, like Ukrainian ї."""
LANGUAGE_FLOORS = (0.5, 0.3, 0.15)
"""For a country that doesn't speak a language, after one, two, or three or more signs of it."""
ENGLISH_FLOOR = 0.7
"""English is on signs everywhere, so it barely counts."""
DOMAIN_FLOOR = 0.1
PHONE_FLOOR = 0.1

SCRIPT_NAMES = {
    "han": "Chinese characters",
    "kana": "Japanese kana",
    "hangul": "Korean Hangul",
    "cyrillic": "Cyrillic",
    "greek": "Greek",
    "thai": "Thai",
    "arabic": "Arabic script",
    "devanagari": "Devanagari",
}

# Letters within a script that only some of the countries using it write.
LETTER_HINTS: dict[str, tuple[str, ...]] = {
    "єїґ": ("UA",),
    "ў": ("BY",),
    "ђћ": ("RS", "ME", "BA"),
    "љњџј": ("RS", "ME", "BA", "MK"),
    "ѓќѕ": ("MK",),
    "әғқұһ": ("KZ",),
    "ңөү": ("KZ", "KG", "MN"),
    "ыэ": ("RU", "BY", "KZ", "KG", "MN", "UZ", "TJ", "MD"),
    "پچژگ": ("IR", "AF", "PK"),
    "ٹڈڑںے": ("PK",),
}

# Latin-script languages: name, letters few other languages use, and words and phrases
# common on their streets and signs.
LANGUAGES: dict[str, tuple[str, str, str]] = {
    "pt": (
        "Portuguese",
        "ãõ",
        "rua,avenida,estrada,rodovia,travessa,praça,largo,alameda,saída,proibido,farmácia,"
        "padaria,loja,aluga-se,vende-se,oficina,correios,prefeitura,freguesia",
    ),
    "es": (
        "Spanish",
        "ñ",
        "calle,avenida,carretera,camino,paseo,calzada,carrera,jirón,pasaje,salida,prohibido,"
        "farmacia,panadería,tienda,ferretería,alquiler,gasolinera,municipalidad,ayuntamiento,"
        "colonia,ruta,oficina,se vende",
    ),
    "fr": (
        "French",
        "œ",
        "rue,avenue,boulevard,chemin,allée,impasse,sortie,interdit,pharmacie,boulangerie,"
        "mairie,arrêt,centre-ville,à vendre,à louer,toutes directions,autres directions",
    ),
    "it": (
        "Italian",
        "",
        "via,viale,piazza,corso,vicolo,uscita,vietato,farmacia,vendesi,affittasi,comune,"
        "località,tabacchi",
    ),
    "de": (
        "German",
        "ß",
        "straße,strasse,weg,gasse,platz,allee,ausfahrt,einfahrt,verboten,apotheke,bäckerei,"
        "gasthaus,gemeinde,zentrum,zu verkaufen",
    ),
    "nl": (
        "Dutch",
        "",
        "straat,weg,laan,plein,gracht,dijk,kade,steeg,uitrit,verboden,apotheek,gemeente,"
        "winkel,kerk,te koop,te huur",
    ),
    "pl": (
        "Polish",
        "łąęśźż",
        "ulica,aleja,plac,osiedle,wjazd,wyjazd,zakaz,apteka,sklep,sprzedam,gmina",
    ),
    "cs": ("Czech", "ěřů", "ulice,náměstí,třída,výjezd,zákaz,lékárna,prodej,obec"),
    "sk": ("Slovak", "ľĺŕ", "ulica,námestie,cesta,výjazd,zákaz,lekáreň,predaj,obec"),
    "hu": ("Hungarian", "őű", "utca,út,tér,körút,kijárat,tilos,gyógyszertár,eladó"),
    "ro": (
        "Romanian",
        "ășțşţ",
        "strada,bulevardul,calea,șoseaua,aleea,ieșire,interzis,farmacie,vând,magazin,primăria",
    ),
    "hr": ("Croatian", "ćđ", "ulica,cesta,trg,izlaz,zabranjeno,ljekarna,prodaja"),
    "sr": ("Serbian", "ćđ", "ulica,bulevar,trg,izlaz,apoteka,prodaja"),
    "sl": ("Slovene", "", "cesta,ulica,trg,izvoz,lekarna,prodaja"),
    "sq": ("Albanian", "ë", "rruga,bulevardi,sheshi,dalje,ndalohet,farmaci,shitet"),
    "tr": (
        "Turkish",
        "ğış",
        "sokak,sokağı,caddesi,cadde,mahallesi,bulvarı,yolu,çıkış,yasak,eczane,satılık,kiralık",
    ),
    "sv": ("Swedish", "å", "gatan,vägen,gata,väg,torget,gränd,utfart,förbjudet,apotek,säljes"),
    "no": ("Norwegian", "åæø", "gata,veien,vegen,plass,torget,utkjørsel,forbudt,apotek,selges"),
    "da": ("Danish", "åæø", "gade,vej,allé,plads,stræde,udkørsel,forbudt,apotek,sælges"),
    "fi": ("Finnish", "", "katu,kuja,polku,tori,liittymä,kielletty,apteekki,myydään"),
    "et": ("Estonian", "õ", "tänav,maantee,puiestee,väljak,väljasõit,keelatud,apteek,müüa"),
    "lv": ("Latvian", "āēīūģķļņ", "iela,prospekts,laukums,šoseja,aizliegts,aptieka,pārdod"),
    "lt": (
        "Lithuanian",
        "ąęėįų",
        "gatvė,prospektas,aikštė,plentas,draudžiama,vaistinė,parduodamas",
    ),
    "is": ("Icelandic", "þð", "gata,vegur,braut,stígur,apótek"),
    "fo": ("Faroese", "ðø", "gøta,vegur"),
    "mt": ("Maltese", "ħġċż", "triq,pjazza,sqaq"),
    "vi": (
        "Vietnamese",
        "ăđơưãõạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ",
        "đường,phố,ngõ,hẻm,quốc lộ,cà phê,nhà hàng",
    ),
    "id": (
        "Indonesian",
        "",
        "jalan,toko,warung,dijual,kantor,desa,kecamatan,kabupaten,masjid,sekolah,apotek,"
        "rumah makan",
    ),
    "ms": ("Malay", "", "jalan,lorong,lebuh,persiaran,kedai,dijual,masjid,sekolah,kampung"),
    "tl": ("Filipino", "", "barangay,kalye,tindahan,salamat,mabuhay,sari-sari"),
    "sw": ("Swahili", "", "barabara,mtaa,duka,hoteli,karibu,shule,kanisa,kituo"),
    "af": ("Afrikaans", "", "straat,weg,rylaan,winkel,kerk"),
    "ca": ("Catalan", "", "carrer,avinguda,plaça,passeig,camí,sortida"),
    "eu": ("Basque", "", "kalea,etorbidea,irteera"),
    "gl": ("Galician", "", "rúa,praza,saída"),
    "cy": ("Welsh", "ŵŷ", "ffordd,stryd,heol,araf"),
    "ga": ("Irish", "", "bóthar,sráid,géill slí"),
    "mi": ("Māori", "", "whare,marae,kia ora"),
    "en": (
        "English",
        "",
        "street,road,avenue,lane,drive,highway,boulevard,parking,pharmacy,chemist",
    ),
}
ABBREVIATIONS = {  # counted only when written with a dot, as on street signs
    "jl": "id",
    "jln": "ms",
    "brgy": "tl",
    "av": "pt,es",
    "str": "de,ro",
    "ul": "pl",
    "cd": "tr",
    "sk": "tr",
    "mah": "tr",
    "rr": "sq",
}
_AMBIGUOUS_DOMAINS = {"at", "be", "do", "go", "id", "in", "is", "it", "me", "my", "no", "so"}
_SECOND_LEVEL = {"com", "co", "org", "net", "gov", "gob", "gouv", "edu", "ac", "or", "ne", "go"}
_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
_DOMAIN = re.compile(
    r"(?<![\w.-])((?:https?://)?(?:www\.)?)((?:[a-z0-9-]+\.)+)([a-z]{2,3})(?![\w])"
)
_PHONE = re.compile(r"\+\s?(\d[\d\s().-]{7,})")


@dataclass(frozen=True)
class TextLine:
    text: str
    script: str
    """Writing system: latin, han, kana, hangul, cyrillic, greek, thai, arabic or devanagari."""
    score: float


@dataclass
class TextClues:
    likelihood: np.ndarray
    """How well each country fits the writing, indexed like :func:`.countries.country_codes`."""
    notes: list[str] = field(default_factory=list)


def text_clues(lines: Sequence[TextLine]) -> TextClues:
    """Combine every clue in the text into one likelihood per country."""
    clues = TextClues(np.ones(len(country_codes())))

    def weigh(matches: Callable[[Country], bool], floor: float, note: str) -> None:
        fits = per_country(matches) > 0
        clues.likelihood *= np.where(fits, 1.0, floor)
        clues.notes.append(note)

    scripts = {line.script for line in lines}
    for script in sorted(scripts - {"latin"}):
        example = next(line.text for line in lines if line.script == script)
        weigh(
            lambda c, s=script: s in c.scripts, SCRIPT_FLOOR, f"{SCRIPT_NAMES[script]}: {example}"
        )
        if script == "han" and "kana" not in scripts:  # Japanese signs would usually show kana
            clues.likelihood[country_codes().index("JP")] *= 0.3

    everything = " ".join(line.text for line in lines).lower()
    for letters, places in LETTER_HINTS.items():
        if seen := sorted(set(letters) & set(everything)):
            weigh(lambda c, p=places: c.code in p, LETTER_FLOOR, f"letters {''.join(seen)}")

    latin = " ".join(line.text for line in lines if line.script == "latin").lower()
    for languages, signs in sorted(_language_signs(latin).items(), key=lambda kv: sorted(kv[0])):
        floor = ENGLISH_FLOOR if languages == {"en"} else LANGUAGE_FLOORS[min(len(signs), 3) - 1]
        names = "/".join(LANGUAGES[code][0] for code in sorted(languages))
        weigh(
            lambda c, ls=languages: bool(ls & set(c.languages)),
            floor,
            f"{names}: {', '.join(sorted(signs)[:3])}",
        )

    compact = re.sub(r"\s*\.\s*", ".", everything)
    for tld in sorted(_domains(compact)):
        weigh(lambda c, t=tld: t in c.domains, DOMAIN_FLOOR, f"web domain .{tld}")
    for code in sorted(_calling_codes(everything)):
        weigh(lambda c, k=code: k in c.calling_codes, PHONE_FLOOR, f"phone number +{code}")
    return clues


@functools.lru_cache(maxsize=1)
def _indexes() -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Which languages use each telling letter, and which use each word or phrase."""
    letters: dict[str, set[str]] = {}
    words: dict[str, set[str]] = {}
    for code, (_, telling, common) in LANGUAGES.items():
        for letter in telling:
            letters.setdefault(letter, set()).add(code)
        for word in common.split(","):
            words.setdefault(word, set()).add(code)
    for abbreviation, codes in ABBREVIATIONS.items():
        words.setdefault(abbreviation + ".", set()).update(codes.split(","))
    return (
        {k: frozenset(v) for k, v in letters.items()},
        {k: frozenset(v) for k, v in words.items()},
    )


def _language_signs(text: str) -> dict[frozenset[str], set[str]]:
    """Telling letters, words and phrases in lowercase Latin text, grouped by who uses them."""
    letters, words = _indexes()
    found: dict[frozenset[str], set[str]] = {}
    tokens = set(_WORD.findall(text))
    for token in tokens:
        if token in words:
            found.setdefault(words[token], set()).add(token)
        if len(token) >= 3:  # a lone letter is too easily a misreading
            for letter in set(token) & letters.keys():
                found.setdefault(letters[letter], set()).add(letter)
    for phrase, languages in words.items():
        is_phrase, is_abbreviation = " " in phrase, phrase.endswith(".")
        if (is_phrase or is_abbreviation) and re.search(rf"(?<!\w){re.escape(phrase)}", text):
            if is_abbreviation or re.search(rf"{re.escape(phrase)}(?!\w)", text):
                found.setdefault(languages, set()).add(phrase)
    return found


def _domains(text: str) -> set[str]:
    """Country domains in web addresses, like .br in www.lojas.com.br."""
    tlds = {tld for c in countries().values() for tld in c.domains}
    found = set()
    for match in _DOMAIN.finditer(text):
        prefix, labels, tld = match.groups()
        names = labels.rstrip(".").split(".")
        if tld not in tlds:
            continue
        clearly_web = bool(prefix) or names[-1] in _SECOND_LEVEL
        if clearly_web or (tld not in _AMBIGUOUS_DOMAINS and len(names[-1]) >= 4):
            found.add(tld)
    return found


def _calling_codes(text: str) -> set[str]:
    """International calling codes in phone numbers, like 44 in +44 20 7946 0958."""
    codes = {code for c in countries().values() for code in c.calling_codes}
    found = set()
    for match in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", match.group(1))
        if len(digits) < 8:
            continue
        code = next((digits[:n] for n in (3, 2, 1) if digits[:n] in codes), None)
        if code:
            found.add(code)
    return found
