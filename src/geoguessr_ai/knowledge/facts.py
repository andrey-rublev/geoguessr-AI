"""What GeoGuessr players know about each country: the side of the road cars drive on, how much
Google Street View it has, phone codes, web domains, and the writing on its signs.

Coverage is Google's official Street View as of 2025, roughly: ``full`` where cars have
driven most roads, ``some`` for a few roads, towns or trekker paths, ``none`` otherwise.
It is approximate; the game prior that ``learn`` estimates from your rounds refines it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .countries import country_codes

COVERAGE_WEIGHTS = {"full": 1.0, "some": 0.5, "none": 0.05}
"""How likely the game is to send you to a country, relative to one with full coverage.

Of the settings tried on 83 played rounds these did best (+49 points, give or take 89): harsher
ones lost as many rounds to a wrong call as they saved."""

# code, drives on (L/R), coverage, calling codes, web domains, scripts on signs, languages
_TABLE = """
AD R full 376 ad latin ca,es,fr
AE R full 971 ae arabic,latin ar,en
AF R none 93 af arabic fa,ps
AG L none 1 ag latin en
AI L none 1 ai latin en
AL R full 355 al latin sq
AM R some 374 am armenian hy
AO R none 244 ao latin pt
AQ R some - aq latin en
AR R full 54 ar latin es
AS R full 1 as latin en,sm
AT R full 43 at latin de
AU L full 61 au latin en
AW R some 297 aw latin nl,pap
AX R full 358 ax latin sv
AZ R none 994 az latin az
BA R some 387 ba latin,cyrillic bs,hr,sr
BB L none 1 bb latin en
BD L full 880 bd bengali,latin bn,en
BE R full 32 be latin nl,fr,de
BF R none 226 bf latin fr
BG R full 359 bg cyrillic,latin bg
BH R none 973 bh arabic,latin ar,en
BI R none 257 bi latin fr,rn
BJ R none 229 bj latin fr
BL R none 590 bl latin fr
BM L full 1 bm latin en
BN L some 673 bn latin,arabic ms,en
BO R full 591 bo latin es
BR R full 55 br latin pt
BS L some 1 bs latin en
BT L full 975 bt tibetan,latin dz,en
BW L full 267 bw latin en,tn
BY R none 375 by cyrillic be,ru
BZ R none 501 bz latin en,es
CA R full 1 ca latin en,fr
CD R none 243 cd latin fr
CF R none 236 cf latin fr
CG R none 242 cg latin fr
CH R full 41 ch latin de,fr,it,rm
CI R none 225 ci latin fr
CK L some 682 ck latin en,mi
CL R full 56 cl latin es
CM R none 237 cm latin fr,en
CN R none 86 cn han,latin zh
CO R full 57 co latin es
CR R some 506 cr latin es
CU R none 53 cu latin es
CV R none 238 cv latin pt
CW R full 599 cw latin nl,pap,en
CY L some 357 cy greek,latin el,tr,en
CZ R full 420 cz latin cs
DE R full 49 de latin de
DJ R none 253 dj latin,arabic fr,ar
DK R full 45 dk latin da
DM L none 1 dm latin en
DO R full 1 do latin es
DZ R none 213 dz arabic,latin ar,fr
EC R full 593 ec latin es
EE R full 372 ee latin et
EG R some 20 eg arabic,latin ar
EH R none 212 - arabic ar
ER R none 291 er ethiopic ti
ES R full 34 es latin es,ca,eu,gl
ET R none 251 et ethiopic am
FI R full 358 fi latin fi,sv
FJ L some 679 fj latin en
FK L some 500 fk latin en
FM R some 691 fm latin en
FO R full 298 fo latin fo
FR R full 33,262,590,594,596 fr,re,gp,mq,gf,yt latin fr
GA R none 241 ga latin fr
GB L full 44 uk latin en,cy
GD L none 1 gd latin en
GE R some 995 ge georgian ka
GG L some 44 gg latin en
GH R full 233 gh latin en
GI R full 350 gi latin en,es
GL R full 299 gl latin kl,da
GM R none 220 gm latin en
GN R none 224 gn latin fr
GQ R none 240 gq latin es,fr
GR R full 30 gr greek,latin el
GS L some 500 gs latin en
GT R full 502 gt latin es
GU R full 1 gu latin en
GW R none 245 gw latin pt
GY L none 592 gy latin en
HK L full 852 hk han,latin zh,en
HM L none - hm latin en
HN R none 504 hn latin es
HR R full 385 hr latin hr
HT R none 509 ht latin fr
HU R full 36 hu latin hu
ID L full 62 id latin id
IE L full 353 ie latin en,ga
IL R full 972 il hebrew,arabic,latin he,ar,en
IM L full 44 im latin en
IN L full 91 in devanagari,bengali,tamil,telugu,kannada,malayalam,gujarati,gurmukhi,latin hi,en
IO R none 246 io latin en
IQ R none 964 iq arabic ar,ku
IR R none 98 ir arabic fa
IS R full 354 is latin is
IT R full 39 it latin it
JE L full 44 je latin en
JM L some 1 jm latin en
JO R full 962 jo arabic,latin ar,en
JP L full 81 jp kana,han,latin ja
KE L full 254 ke latin en,sw
KG R full 996 kg cyrillic ky,ru
KH R full 855 kh khmer,latin km
KI L none 686 ki latin en
KM R none 269 km latin,arabic fr,ar
KN L none 1 kn latin en
KP R none 850 kp hangul ko
KR R full 82 kr hangul,latin ko
KW R none 965 kw arabic,latin ar
KY L some 1 ky latin en
KZ R full 7,997 kz cyrillic,latin kk,ru
LA R full 856 la lao,latin lo
LB R some 961 lb arabic,latin ar,fr
LC L none 1 lc latin en
LI R full 423 li latin de
LK L full 94 lk sinhala,tamil,latin si,ta,en
LR R none 231 lr latin en
LS L full 266 ls latin en,st
LT R full 370 lt latin lt
LU R full 352 lu latin lb,fr,de
LV R full 371 lv latin lv
LY R none 218 ly arabic ar
MA R none 212 ma arabic,latin ar,fr
MC R full 377 mc latin fr
MD R none 373 md latin,cyrillic ro,ru
ME R full 382 me latin,cyrillic sr
MF R some 590 mf latin fr
MG R some 261 mg latin mg,fr
MH R none 692 mh latin en
MK R full 389 mk cyrillic,latin mk,sq
ML R none 223 ml latin fr
MM R none 95 mm myanmar my
MN R full 976 mn cyrillic mn
MO L full 853 mo han,latin zh,pt
MP R full 1 mp latin en
MR R none 222 mr arabic ar
MS L none 1 ms latin en
MT L full 356 mt latin mt,en
MU L some 230 mu latin en,fr
MV L some 960 mv thaana,latin dv,en
MW L none 265 mw latin en
MX R full 52 mx latin es
MY L full 60 my latin,han,tamil ms,en
MZ L none 258 mz latin pt
NA L some 264 na latin en,af
NC R some 687 nc latin fr
NE R none 227 ne latin fr
NF L some 672 nf latin en
NG R full 234 ng latin en
NI R none 505 ni latin es
NL R full 31 nl latin nl
NO R full 47 no latin no
NP L some 977 np devanagari,latin ne
NR L none 674 nr latin en
NU L none 683 nu latin en
NZ L full 64 nz latin en,mi
OM R some 968 om arabic,latin ar,en
PA R full 507 pa latin es
PE R full 51 pe latin es
PF R some 689 pf latin fr
PG L none 675 pg latin en
PH R full 63 ph latin tl,en
PK L none 92 pk arabic,latin ur,en
PL R full 48 pl latin pl
PM R full 508 pm latin fr
PN L some 64 pn latin en
PR R full 1 pr latin es,en
PS R some 970 ps arabic,latin ar
PT R full 351 pt latin pt
PW R some 680 pw latin en
PY R some 595 py latin es
QA R full 974 qa arabic,latin ar,en
RO R full 40 ro latin ro
RS R full 381 rs cyrillic,latin sr
RU R full 7 ru cyrillic,latin ru
RW R full 250 rw latin rw,en,fr
SA R some 966 sa arabic,latin ar,en
SB L none 677 sb latin en
SC L none 248 sc latin en,fr
SD R none 249 sd arabic ar
SE R full 46 se latin sv
SG L full 65 sg latin,han,tamil en,ms,zh
SH L some 290 sh latin en
SI R full 386 si latin sl
SK R full 421 sk latin sk
SL R none 232 sl latin en
SM R full 378 sm latin it
SN R full 221 sn latin fr
SO R none 252 so latin,arabic so,ar
SR L none 597 sr latin nl
SS R none 211 ss latin en
ST R full 239 st latin pt
SV R none 503 sv latin es
SX R some 1 sx latin en,nl
SY R none 963 sy arabic ar
SZ L full 268 sz latin en
TC L some 1 tc latin en
TD R none 235 td latin,arabic fr,ar
TF R none 262 tf latin fr
TG R none 228 tg latin fr
TH L full 66 th thai,latin th
TJ R none 992 tj cyrillic tg,ru
TL L none 670 tl latin pt
TM R none 993 tm latin tk
TN R full 216 tn arabic,latin ar,fr
TO L some 676 to latin en
TR R full 90 tr latin tr
TT L none 1 tt latin en
TV L none 688 tv latin en
TW R full 886 tw han,latin zh
TZ L some 255 tz latin sw,en
UA R full 380 ua cyrillic,latin uk
UG L full 256 ug latin en,sw
UM R none 1 um latin en
US R full 1 us latin en,es
UY R full 598 uy latin es
UZ R none 998 uz latin,cyrillic uz,ru
VA R some 39,379 va latin it
VC L none 1 vc latin en
VE R none 58 ve latin es
VG L some 1 vg latin en
VI L full 1 vi latin en
VN R some 84 vn latin vi
VU R none 678 vu latin fr,en
WF R none 681 wf latin fr
WS L some 685 ws latin en
XK R none 383 - latin sq,sr
YE R none 967 ye arabic ar
ZA L full 27 za latin en,af
ZM L none 260 zm latin en
ZW L none 263 zw latin en
"""


@dataclass(frozen=True)
class Country:
    code: str
    drives_left: bool
    coverage: str
    calling_codes: tuple[str, ...]
    domains: tuple[str, ...]
    scripts: tuple[str, ...]
    languages: tuple[str, ...]


def _items(field: str) -> tuple[str, ...]:
    return () if field == "-" else tuple(field.split(","))


@functools.lru_cache(maxsize=1)
def countries() -> dict[str, Country]:
    """Facts for every country, by ISO 3166-1 alpha-2 code."""
    table = {}
    for line in _TABLE.strip().splitlines():
        code, drive, coverage, calling, domains, scripts, languages = line.split()
        table[code] = Country(
            code,
            drive == "L",
            coverage,
            _items(calling),
            _items(domains),
            _items(scripts),
            _items(languages),
        )
    return table


def per_country(value: Callable[[Country], float], default: float = 0.0) -> np.ndarray:
    """``value(country)`` for every index of :func:`.countries.country_codes`."""
    table = countries()
    return np.array(
        [value(table[code]) if code in table else default for code in country_codes()],
        dtype=np.float64,
    )
