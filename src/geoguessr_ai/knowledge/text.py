"""What the writing on screen gives away: scripts, languages, web domains, phone numbers,
brands, prices, speed limits, postcodes, road numbers and the names of towns.

Lines of text read off the screen become a likelihood for every country. Each kind of clue
counts once however many signs show it, and no clue rules a country out completely: text can
be misread, and brands, tourists and road signs cross borders.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .countries import country_codes
from .facts import Country, countries, per_country
from .places import place_key, places_in

# How little weight a country keeps when a clue points away from it. These were timid: on the
# saved rounds a clue named the country the round was really in 69 times out of 70, and clues
# counted this much more sharply would have gained about 36 points a round (give or take 26)
# over all of them, and 120 (give or take 86) over the ones the model was never trained on.
SCRIPT_FLOOR = 0.03
"""Likelihood for a country whose signs don't use a script that was clearly read."""
LETTER_FLOOR = 0.15
"""For a country that doesn't use a telling letter that was read, like Ukrainian ї."""
LANGUAGE_FLOORS = (0.2, 0.1, 0.05)
"""For a country that doesn't speak a language, after one, two, or three or more signs of it."""
ENGLISH_FLOOR = 0.7
"""English is on signs everywhere, so it barely counts."""
CUT_OFF_LETTERS, CUT_OFF_MISSING = 5, 3
"""A word at the end of a line of text may be cut off by the frame: count it as a sign word
when it has at least this many letters and the sign word only this many more."""
DOMAIN_FLOOR = 0.1
PHONE_FLOOR = 0.1
LOCAL_PHONE_FLOOR = 0.15
"""For a phone number written the local way, which neighbours and misreadings can share."""
TELLING_FLOOR = 0.15
"""For a country whose shops don't price the way a price was written: currencies travel."""
OFFICIAL_FLOOR = 0.04
"""For a country whose roads and post aren't numbered the way a sign was, which only a
misreading should manage. The one clue on the saved rounds that missed the country it was
really in was a French street sign 10 km inside Germany, not a road number."""
PLACE_FLOOR, MAX_PLACES = 0.2, 3
"""For a country without a town of a name read, of which at most this many count (the longest):
a sign may point across a border, and a shop may be named after a faraway city."""

# Letters that look like Latin ones. A word in one of these scripts made of nothing else is a
# misreading: со read from Francisco once sent a Mexican round to North Macedonia.
LATIN_LOOKALIKES = {
    "cyrillic": set("АВЕЅІЈКМНОРСТУХаеіјорѕсух"),
    "greek": set("ΑΒΕΖΗΙΚΜΝΟΡΤΥΧαικνορτυχ"),
}

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
        "padaria,loja,aluga-se,vende-se,oficina,correios,prefeitura,freguesia,desconto,promoção,"
        "aluguel,lanchonete,borracharia,açougue,sorveteria,mercearia,drogaria",
    ),
    "es": (
        "Spanish",
        "ñ",
        "calle,avenida,carretera,camino,paseo,calzada,carrera,jirón,pasaje,salida,prohibido,"
        "farmacia,panadería,tienda,ferretería,alquiler,gasolinera,municipalidad,ayuntamiento,"
        "colonia,ruta,oficina,se vende,se renta,descuento,vulcanizadora,llantera,abarrotes,"
        "tortillería,carnicería,papelería,refaccionaria,licorería,cerrajería",
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
        "đường,phố,ngõ,hẻm,quốc lộ,cà phê,nhà hàng,nguyen,nguyễn,huynh,phuong,truong,hoang",
    ),
    "id": (
        "Indonesian",
        "",
        "jalan,raya,selatan,utara,timur,barat,toko,warung,dijual,kantor,desa,kecamatan,"
        "kabupaten,masjid,sekolah,apotek,rumah makan",
    ),
    "ms": (
        "Malay",
        "",
        "jalan,lorong,lebuh,persiaran,selatan,utara,timur,barat,kedai,dijual,masjid,sekolah,"
        "kampung",
    ),
    "tl": ("Filipino", "", "barangay,kalye,tindahan,salamat,mabuhay,sari-sari"),
    "sw": (
        "Swahili",
        "",
        "barabara,mtaa,duka,hoteli,karibu,shule,kanisa,kituo,kinyozi,dawa,nyama,choma,maziwa,"
        "mboga,chakula,huduma,pesa,soko,jumla,wakala",
    ),
    "af": ("Afrikaans", "", "straat,weg,rylaan,laan,dorp,fabriek,winkel,kerk,apteek,sentrum"),
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
# Road words written onto the end of the name, as Ivalontie, Eindstraat and Storgatan are,
# rather than beside it. Street View writes road names on the road itself, so these turn up
# whenever the bot looks down at one.
ROAD_ENDINGS = {
    "gatan": "sv",
    "vägen": "sv",
    "gränd": "sv",
    "veien": "no",
    "vegen": "no",
    "gata": "no,is",
    "vej": "da",
    "gade": "da",
    "vegur": "is",
    "braut": "is",
    "stígur": "is",
    "gøta": "fo",
    "tie": "fi",
    "katu": "fi",
    "polku": "fi",
    "tänav": "et",
    "maantee": "et",
    "puiestee": "et",
    "iela": "lv",
    "gatvė": "lt",
    "straat": "nl,af",
    "laan": "nl,af",
    "plein": "nl",
    "gracht": "nl",
    "straße": "de",
    "strasse": "de",
    "gasse": "de",
    "platz": "de",
    "allee": "de",
    "weg": "de,nl,af",
    "utca": "hu",
}
ROAD_ENDING_STEM = 4
"""Letters a name must have before its road ending, so that French SORTIE isn't Finnish -tie."""
ROAD_ENDING_INSIDE = 5
"""Endings this long also count with a word run on after them, as in Bárðardalsvegurvest."""
ABBREVIATIONS = {  # counted only when written with a dot, as on street signs
    "jl": "id",
    "ji": "id",  # how Street View's Jl. usually comes out when read
    "gg": "id",
    "jln": "ms",
    "brgy": "tl",
    "av": "pt,es,fr,ca",
    "tv": "pt",
    "rte": "fr",
    "c": "es",
    "cam": "es",
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
_NANP = tuple(code for code, country in countries().items() if "1" in country.calling_codes)
# Phone numbers without a country code, by how they are grouped where they are written.
LOCAL_PHONES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"(?:\(\d{2}\)\s?)?9\d{4}[-.\s]\d{4}", ("BR",)),  # (11) 99983-2915
    (r"(?:\([2-9]\d{2}\)\s?|[2-9]\d{2}[-.])[2-9]\d{2}[-.]\d{4}", _NANP),  # (415) 555-0132
    (r"0[1-9](?:[\s.]\d{2}){4}", ("FR",)),  # 01 42 68 53 00
    (r"0[17]\d{3}\s\d{3}\s?\d{3}|020\s[378]\d{3}\s\d{4}", ("GB", "IM", "JE", "GG")),
    (r"04\d{2}\s\d{3}\s\d{3}|\(0[2378]\)\s?\d{4}\s\d{4}", ("AU",)),  # 0412 345 678
    (r"07\d{2}\s\d{3}\s?\d{3}", ("KE", "UG", "TZ", "RW", "ZM", "MW")),  # 0722 123 456
    (r"0[789]\d{2}[\s-]\d{3}[\s-]\d{4}", ("PH", "NG")),  # 0917 123 4567
    (r"0[689]\d-\d{3}-\d{4}", ("TH",)),  # 081-234-5678
    (r"08\d{2}[-\s]\d{4}[-\s]\d{3,5}", ("ID",)),  # 0812-3456-7890
    (r"0[789]0-\d{4}-\d{4}|0120-\d{3}-\d{3}", ("JP",)),  # 090-1234-5678
    (r"010-\d{4}-\d{4}", ("KR",)),  # 010-1234-5678
    (r"\d{3}-\d{2}-\d{2}", ("RU", "UA", "BY", "KZ", "KG", "UZ", "TJ", "MD")),  # 123-45-67
    (r"[6-9]\d{4}\s\d{5}", ("IN",)),  # 98765 43210
    (r"0\d{2,4}\s?/\s?\d{3,8}", ("DE", "AT", "CH", "LI")),  # 0221 / 123456
    (r"[69]\d{2}\s\d{3}\s\d{3}", ("ES", "PT")),  # 612 345 678
)
_TOKEN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")
_EURO = ("AT", "BE", "CY", "DE", "EE", "ES", "FI", "FR", "GR", "HR", "IE", "IT", "LT", "LU", "LV")
_EURO += ("MT", "NL", "PT", "SI", "SK", "AD", "MC", "ME", "SM", "VA", "XK", "RE", "GP", "MQ")
_MPH = ("US", "GB", "IM", "JE", "GG", "PR", "GU", "AS", "MP", "VI", "LR", "BS", "BZ", "KY", "VG")
_MPH += ("AG", "DM", "GD", "KN", "LC", "VC", "TC", "AI", "FK")
_BRAZIL_STATES = "sp|mg|rs|sc|go|ba|pe|ce|pa|mt|ms|es|rj|al|se|pb|rn|pi|ma|to|ro|ac|am|rr|ap|df"
# Prices as some countries write them, in lowercase: what to look for, a short name for it,
# and where it is written so.
TELLING_TEXT: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (r"r\$\s?\d", "price in reais", ("BR",)),
    (r"\d\s?zł|\bpln\b", "price in złoty", ("PL",)),
    (r"\d\s?kč", "price in koruny", ("CZ",)),
    (r"\d\s?(?:lei|ron)\b", "price in lei", ("RO", "MD")),
    (r"\d\s?лв", "price in leva", ("BG",)),
    (r"₽|\d\s?руб", "price in roubles", ("RU", "BY")),
    (r"₴|\d\s?грн", "price in hryvnias", ("UA",)),
    (r"₸|\d\s?тг", "price in tenge", ("KZ",)),
    (r"₺|\d\s?tl\b", "price in lira", ("TR",)),
    (r"₹|\brs\.?\s?\d", "price in rupees", ("IN", "PK", "LK", "NP")),
    (r"৳|\btk\.?\s?\d", "price in taka", ("BD",)),
    (r"₱|\bphp\s?\d", "price in pesos", ("PH",)),
    (r"₩", "price in won", ("KR",)),
    (r"₫|\d\s?(?:vnđ|vnd)\b", "price in dong", ("VN",)),
    (r"฿|\d\s?บาท", "price in baht", ("TH",)),
    # Rupiah prices run to thousands, which keeps Argentina's RP 51 route out.
    (r"\brp\.?\s?(?:\d{1,3}(?:[.,]\d{3})+|\d{4,})", "price in rupiah", ("ID",)),
    (r"\brm\s?\d", "price in ringgit", ("MY",)),
    (r"\b(?:k|u|t)shs?\.?\s?\d|\b(?:kes|ugx|tzs)\s?\d", "price in shillings", ("KE", "UG", "TZ")),
    (r"₦", "price in naira", ("NG",)),
    (r"₵|\bghs\s?\d", "price in cedis", ("GH",)),
    (r"\bs/\.?\s?\d", "price in soles", ("PE",)),
    (r"\bchf\s?\d|\d\s?chf\b", "price in francs", ("CH", "LI")),
    (r"\d\s?kr\b|\bkr\.?\s?\d", "price in kronor", ("SE", "NO", "DK", "IS", "FO", "GL")),
    (r"€", "price in euros", _EURO),
    (r"£", "price in pounds", ("GB", "IM", "JE", "GG", "GI", "FK")),
)
# Speed limits, postcodes and road numbers, written the same way. These are what a country's
# own road authority and post office write, which a tourist or a shop's prices can't be.
OFFICIAL_TEXT: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (r"\bmph\b", "speed in mph", _MPH),
    (r"\b[a-z]{1,2}\d[a-z\d]?\s\d[a-z]{2}\b", "British postcode", ("GB", "IM", "JE", "GG")),
    (r"\b[a-z]\d[a-z]\s?\d[a-z]\d\b", "Canadian postcode", ("CA",)),
    (r"(?<![\d-])\d{5}-\d{3}(?![\d-])", "Brazilian postcode", ("BR",)),
    (r"(?<![\d-])\d{4}-\d{3}(?![\d-])", "Portuguese postcode", ("PT",)),
    (r"(?<![\d-])\d{2}-\d{3}(?![\d-])", "Polish postcode", ("PL",)),
    (r"〒", "Japanese postcode", ("JP",)),
    (rf"\b(?:br|{_BRAZIL_STATES})-\d{{3}}\b", "Brazilian road", ("BR",)),
    (r"\bi-\d{1,3}\b", "Interstate", ("US",)),
    (r"\bdn\s?\d{1,3}[a-z]?\b", "Romanian national road", ("RO",)),
    (r"\bss\s?\d{1,3}\b", "Italian state road", ("IT",)),
    (r"\bnh\s?\d{1,3}\b", "Indian national highway", ("IN",)),
    (r"\bsh\s?\d{1,3}\b", "state highway", ("IN", "NZ", "AL")),
    (r"\bco(?:unty)?\s?(?:rd|road|hwy|highway)\s?\d{1,4}", "county road", ("US",)),
    (r"\b(?:range|twp|township)\s?rd\b", "range road", ("CA", "US")),
    (r"\bfm[-\s]?\d{2,4}\b", "farm-to-market road", ("US",)),
    (r"\bdw\s?\d{3}\b", "Polish regional road", ("PL",)),
    (r"\brp\s?\d{1,3}\b", "Argentine provincial route", ("AR",)),
    (r"\bcra\.?\s?\d{1,3}[a-z]?", "carrera", ("CO",)),
    (r"\bmarg\b", "marg", ("IN", "NP")),
    # Street View labels roads in Russia and Central Asia in Latin letters as well.
    (
        r"\b(?:ulitsa|shosse|prospekt|pereulok|proyezd|naberezhnaya)\b",
        "Russian road word in Latin letters",
        ("RU", "BY", "KZ", "KG", "UZ", "TJ"),
    ),
    # American road names carry the compass point and number the streets: S Triple X Rd, 14th St.
    (
        r"\b(?:n|s|e|w|n[ew]|s[ew])\s(?:\w+\s){1,3}(?:rd|st|ave|dr|ln|blvd|hwy|way)\b"
        r"|\b(?:rd|st|ave|dr|ln|blvd|hwy|way)\s(?:n|s|e|w|n[ew]|s[ew])\b",
        "compass point in a road name",
        ("US", "CA"),
    ),
    (
        r"\b\d{1,3}(?:st|nd|rd|th)\s(?:st|ave|pl|ter|blvd|street|avenue|terrace)\b",
        "numbered street",
        ("US", "CA", "PH", "NG"),
    ),
)

BRAND_FLOOR = 0.15
BRAND_INSIDE_LETTERS = 6
"""Brand names this long also count run together with other words, as in OBOTICARIO."""
# Chains found in only one or a few countries: fuel stations, shops, banks and phone networks.
BRANDS: dict[str, tuple[str, ...]] = {
    "pemex": ("MX",),
    "oxxo": ("MX", "CO", "CL", "PE"),
    "telcel": ("MX",),
    "coppel": ("MX",),
    "soriana": ("MX",),
    "chedraui": ("MX",),
    "walmart": ("US", "MX", "CA"),
    "petrobras": ("BR",),
    "ipiranga": ("BR",),
    "boticario": ("BR",),
    "boticário": ("BR",),
    "drogasil": ("BR",),
    "bradesco": ("BR",),
    "sicredi": ("BR",),
    "sicoob": ("BR",),
    "magalu": ("BR",),
    "ypf": ("AR",),
    "axion": ("AR", "PY"),
    "copec": ("CL",),
    "ancap": ("UY",),
    "terpel": ("CO", "PA", "PE", "EC"),
    "bancolombia": ("CO",),
    "davivienda": ("CO",),
    "primax": ("PE", "EC"),
    "inkafarma": ("PE",),
    "walgreens": ("US",),
    "kroger": ("US",),
    "sheetz": ("US",),
    "wawa": ("US",),
    "petro-canada": ("CA",),
    "petrocanada": ("CA",),
    "pertamina": ("ID",),
    "indomaret": ("ID",),
    "alfamart": ("ID", "PH"),
    "alfamidi": ("ID",),
    "petronas": ("MY",),
    "ptt": ("TH", "TR"),
    "bangchak": ("TH",),
    "petron": ("PH", "MY"),
    "jollibee": ("PH",),
    "puregold": ("PH",),
    "cebuana": ("PH",),
    "cellcard": ("KH",),
    "jio": ("IN",),
    "lukoil": ("RU", "BG", "RO", "KZ"),
    "rosneft": ("RU",),
    "пятёрочка": ("RU",),
    "пятерочка": ("RU",),
    "магнит": ("RU",),
    "приватбанк": ("UA",),
    "укрпошта": ("UA",),
    "okko": ("UA",),
    "wog": ("UA",),
    "orlen": ("PL", "CZ", "LT"),
    "żabka": ("PL",),
    "zabka": ("PL",),
    "biedronka": ("PL",),
    "omv": ("AT", "RO", "BG", "HU", "SK", "SI", "RS"),
    "petrom": ("RO", "MD"),
    "dedeman": ("RO",),
    "opet": ("TR",),
    "a101": ("TR",),
    "migros": ("TR", "CH"),
    "repsol": ("ES", "PT", "PE"),
    "cepsa": ("ES", "PT"),
    "galp": ("PT", "ES"),
    "mercadona": ("ES",),
    "leclerc": ("FR", "ES", "PT", "PL", "SI"),
    "intermarché": ("FR", "BE", "PT", "PL"),
    "intermarche": ("FR", "BE", "PT", "PL"),
    "esselunga": ("IT",),
    "conad": ("IT",),
    "edeka": ("DE",),
    "rewe": ("DE",),
    "aral": ("DE", "LU"),
    "sparkasse": ("DE", "AT"),
    "heijn": ("NL", "BE"),
    "neste": ("FI",),
    "preem": ("SE",),
    "okq8": ("SE",),
    "tesco": ("GB", "IE", "CZ", "SK", "HU"),
    "asda": ("GB",),
    "sainsbury": ("GB",),
    "applegreen": ("IE", "GB", "US"),
    "woolworths": ("AU", "NZ", "ZA"),
    "ampol": ("AU",),
    "bunnings": ("AU", "NZ"),
    "engen": ("ZA", "BW", "NA", "LS", "SZ", "KE", "GH"),
    "sasol": ("ZA",),
    "shoprite": ("ZA", "BW", "NA", "LS", "SZ", "ZM", "GH", "MW"),
    "vodacom": ("ZA", "LS", "TZ", "MZ", "CD"),
    "mtn": ("NG", "GH", "UG", "ZA", "RW", "CM", "CI", "BJ", "ZM", "SZ", "LR", "GN", "SS"),
    "airtel": ("KE", "UG", "TZ", "RW", "NG", "ZM", "MW", "MG", "IN"),
    "glo": ("NG", "GH", "BJ"),
    "safaricom": ("KE",),
    "m-pesa": ("KE", "TZ"),
    "mpesa": ("KE", "TZ"),
    "naivas": ("KE",),
    "lawson": ("JP",),
    "eneos": ("JP",),
    "ministop": ("JP", "VN", "PH"),
    "gs25": ("KR",),
}


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

    scripts = {line.script for line in lines if _really_written_in(line.script, lines)}
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
    for brand in sorted(_brands(everything)):
        weigh(lambda c, b=brand: c.code in BRANDS[b], BRAND_FLOOR, f"brand {brand}")

    latin = [line.text.lower() for line in lines if line.script == "latin"]
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
    for number, places in _local_numbers(_PHONE.sub(" ", everything)):
        weigh(lambda c, p=places: c.code in p, LOCAL_PHONE_FLOOR, f"phone number {number}")
    # Within a line, since two signs side by side are not one: a Turkish 811.SH above a
    # 247.Sk. read as the state highway SH 247, which Turkey doesn't have.
    written = [line.text.lower() for line in lines]
    for table, floor in ((TELLING_TEXT, TELLING_FLOOR), (OFFICIAL_TEXT, OFFICIAL_FLOOR)):
        for pattern, name, places in table:
            found = (re.search(pattern, text) for text in written)
            if match := next((m for m in found if m), None):
                note = f"{name}: {match.group(0).strip()}"
                weigh(lambda c, p=places: c.code in p, floor, note)
    naming = {place_key(word) for word in _indexes()[1]}
    towns = sorted(
        places_in((line.text for line in lines), naming).items(), key=lambda kv: -len(kv[0])
    )
    for town, places in towns[:MAX_PLACES]:
        weigh(lambda c, p=places: c.code in p, PLACE_FLOOR, f"town {town} ({', '.join(places)})")
    return clues


@functools.lru_cache(maxsize=1)
def _indexes() -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Which languages use each telling letter, and which use each word or phrase. Words are
    also listed without their accents, which signs in capitals and misreadings often lose."""
    letters: dict[str, set[str]] = {}
    words: dict[str, set[str]] = {}
    for code, (_, telling, common) in LANGUAGES.items():
        for letter in telling:
            letters.setdefault(letter, set()).add(code)
        for word in common.split(","):
            words.setdefault(word, set()).add(code)
            if len(plain := _without_accents(word)) >= 4:
                words.setdefault(plain, set()).add(code)
    for abbreviation, codes in ABBREVIATIONS.items():
        words.setdefault(abbreviation + ".", set()).update(codes.split(","))
    return (
        {k: frozenset(v) for k, v in letters.items()},
        {k: frozenset(v) for k, v in words.items()},
    )


def _really_written_in(script: str, lines: Sequence[TextLine]) -> bool:
    """Whether text read in a script is really in it. Everything read in a script has to show
    at least one letter that couldn't be a misread Latin one, somewhere across the lines."""
    lookalikes = LATIN_LOOKALIKES.get(script)
    if lookalikes is None:
        return True
    letters = {c for line in lines if line.script == script for c in line.text if c.isalpha()}
    return bool(letters - lookalikes)


@functools.lru_cache(maxsize=1)
def _endings() -> dict[str, frozenset[str]]:
    """Which languages write each road ending, also without its accents, as capitals lose them."""
    found: dict[str, set[str]] = {}
    for ending, codes in ROAD_ENDINGS.items():
        for spelling in {ending, _without_accents(ending)}:
            found.setdefault(spelling, set()).update(codes.split(","))
    return {k: frozenset(v) for k, v in found.items()}


def _ends_road(token: str, ending: str) -> bool:
    """Whether a name is built on a road ending: enough name before it, and nothing after it
    unless the ending is long enough to be sure of even with a word run on."""
    if len(token) < len(ending) + ROAD_ENDING_STEM:
        return False
    if len(ending) >= ROAD_ENDING_INSIDE:
        return ending in token[ROAD_ENDING_STEM:]
    return token.endswith(ending)


def _without_accents(word: str) -> str:
    """``praça`` as ``praca``. Letters that aren't accented forms, like ß and đ, stay."""
    decomposed = unicodedata.normalize("NFD", word)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if not unicodedata.combining(c))
    )


def _language_signs(lines: Sequence[str]) -> dict[frozenset[str], set[str]]:
    """Telling letters, words and phrases in lowercase lines of Latin text, grouped by who uses
    them. Words cut off at either end of a line are marked with an ellipsis."""
    letters, words = _indexes()
    found: dict[frozenset[str], set[str]] = {}
    text = " ".join(lines)
    ends = set()
    for line in lines:
        if line_tokens := _WORD.findall(line):
            ends.update((line_tokens[0], line_tokens[-1]))
    for token in ends - words.keys():
        if len(token) < CUT_OFF_LETTERS:
            continue
        for word, languages in words.items():
            cut_from = word.startswith(token) and len(word) - len(token) <= CUT_OFF_MISSING
            if cut_from and _WORD.fullmatch(word):
                found.setdefault(languages, set()).add(token + "…")
    tokens = set(_WORD.findall(text))
    endings = _endings()
    for token in tokens:
        if token in words:
            found.setdefault(words[token], set()).add(token)
        if len(token) >= 3:  # a lone letter is too easily a misreading
            for letter in set(token) & letters.keys():
                found.setdefault(letters[letter], set()).add(letter)
        for ending, languages in endings.items():
            if _ends_road(token, ending):
                found.setdefault(languages, set()).add("…" + ending)
    for phrase, languages in words.items():
        is_phrase, is_abbreviation = " " in phrase, phrase.endswith(".")
        if (is_phrase or is_abbreviation) and re.search(rf"(?<!\w){re.escape(phrase)}", text):
            if is_abbreviation or re.search(rf"{re.escape(phrase)}(?!\w)", text):
                found.setdefault(languages, set()).add(phrase)
    return found


def _brands(text: str) -> set[str]:
    """Brands named in lowercase text, alone or, for long names, run together with other words.
    A brand inside another one found, like petron in petronas, doesn't count again."""
    tokens = set(_TOKEN.findall(text))
    found = {
        brand
        for brand in BRANDS
        if brand in tokens
        or (len(brand) >= BRAND_INSIDE_LETTERS and any(brand in token for token in tokens))
    }
    return {brand for brand in found if not any(o != brand and brand in o for o in found)}


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


def _local_numbers(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """The first phone number of each local way of writing them, and where it is written so."""
    found = []
    for pattern, places in LOCAL_PHONES:
        if match := re.search(rf"(?<![\d+])(?:{pattern})(?!\d)", text):
            found.append((match.group(0), places))
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
