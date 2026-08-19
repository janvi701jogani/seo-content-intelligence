"""
Search Intelligence module.

Self-contained: captures Google's understanding of user intent from four
sources, merges them into one Question Intelligence view, and cross-refs
against Reddit and competitors. Reuses modules.extractor (NOT modified).

    People Also Ask (Serper)  ┐
    Related Searches (Serper) │
    Autosuggest (Google)      ├─> Question Intelligence (cluster + type)
    FAQ (competitor pages)    ┘        │
                                       v
              Signal Matrix: Google x Reddit x Competitors

The signal matrix is the payoff: a question that Google surfaces, Reddit
users discuss, and competitors barely answer is a stronger content
opportunity than any single source shows on its own.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from modules.extractor import (
    compact_text,
    get_sentence_model,
    make_agglomerative,
)

SERPER_ENDPOINT = "https://google.serper.dev/search"
AUTOSUGGEST_ENDPOINT = "https://suggestqueries.google.com/complete/search"

# Streamlit selectbox returns the dict KEY ("India"); normalize to codes.
COUNTRY_CODES = {
    "afghanistan": "af",
    "albania": "al",
    "algeria": "dz",
    "american samoa": "as",
    "andorra": "ad",
    "angola": "ao",
    "anguilla": "ai",
    "antarctica": "aq",
    "antigua and barbuda": "ag",
    "argentina": "ar",
    "armenia": "am",
    "aruba": "aw",
    "australia": "au",
    "austria": "at",
    "azerbaijan": "az",
    "bahamas (the)": "bs",
    "bahrain": "bh",
    "bangladesh": "bd",
    "barbados": "bb",
    "belarus": "by",
    "belgium": "be",
    "belize": "bz",
    "benin": "bj",
    "bermuda": "bm",
    "bhutan": "bt",
    "bolivia (plurinational state of)": "bo",
    "bonaire, sint eustatius and saba": "bq",
    "bosnia and herzegovina": "ba",
    "botswana": "bw",
    "bouvet island": "bv",
    "brazil": "br",
    "british indian ocean territory (the)": "io",
    "brunei darussalam": "bn",
    "bulgaria": "bg",
    "burkina faso": "bf",
    "burundi": "bi",
    "cabo verde": "cv",
    "cambodia": "kh",
    "cameroon": "cm",
    "canada": "ca",
    "cayman islands (the)": "ky",
    "central african republic (the)": "cf",
    "chad": "td",
    "chile": "cl",
    "china": "cn",
    "christmas island": "cx",
    "cocos (keeling) islands (the)": "cc",
    "colombia": "co",
    "comoros (the)": "km",
    "congo (the democratic republic of the)": "cd",
    "congo (the)": "cg",
    "cook islands (the)": "ck",
    "costa rica": "cr",
    "croatia": "hr",
    "cuba": "cu",
    "curaçao": "cw",
    "cyprus": "cy",
    "czechia": "cz",
    "côte d'ivoire": "ci",
    "denmark": "dk",
    "djibouti": "dj",
    "dominica": "dm",
    "dominican republic (the)": "do",
    "ecuador": "ec",
    "egypt": "eg",
    "el salvador": "sv",
    "equatorial guinea": "gq",
    "eritrea": "er",
    "estonia": "ee",
    "eswatini": "sz",
    "ethiopia": "et",
    "falkland islands (the) [malvinas]": "fk",
    "faroe islands (the)": "fo",
    "fiji": "fj",
    "finland": "fi",
    "france": "fr",
    "french guiana": "gf",
    "french polynesia": "pf",
    "french southern territories (the)": "tf",
    "gabon": "ga",
    "gambia (the)": "gm",
    "georgia": "ge",
    "germany": "de",
    "ghana": "gh",
    "gibraltar": "gi",
    "greece": "gr",
    "greenland": "gl",
    "grenada": "gd",
    "guadeloupe": "gp",
    "guam": "gu",
    "guatemala": "gt",
    "guernsey": "gg",
    "guinea": "gn",
    "guinea-bissau": "gw",
    "guyana": "gy",
    "haiti": "ht",
    "heard island and mcdonald islands": "hm",
    "holy see (the)": "va",
    "honduras": "hn",
    "hong kong": "hk",
    "hungary": "hu",
    "iceland": "is",
    "india": "in",
    "indonesia": "id",
    "iran (islamic republic of)": "ir",
    "iraq": "iq",
    "ireland": "ie",
    "isle of man": "im",
    "israel": "il",
    "italy": "it",
    "jamaica": "jm",
    "japan": "jp",
    "jersey": "je",
    "jordan": "jo",
    "kazakhstan": "kz",
    "kenya": "ke",
    "kiribati": "ki",
    "korea (the democratic people's republic of)": "kp",
    "korea (the republic of)": "kr",
    "kuwait": "kw",
    "kyrgyzstan": "kg",
    "lao people's democratic republic (the)": "la",
    "latvia": "lv",
    "lebanon": "lb",
    "lesotho": "ls",
    "liberia": "lr",
    "libya": "ly",
    "liechtenstein": "li",
    "lithuania": "lt",
    "luxembourg": "lu",
    "macao": "mo",
    "madagascar": "mg",
    "malawi": "mw",
    "malaysia": "my",
    "maldives": "mv",
    "mali": "ml",
    "malta": "mt",
    "marshall islands (the)": "mh",
    "martinique": "mq",
    "mauritania": "mr",
    "mauritius": "mu",
    "mayotte": "yt",
    "mexico": "mx",
    "micronesia (federated states of)": "fm",
    "moldova (the republic of)": "md",
    "monaco": "mc",
    "mongolia": "mn",
    "montenegro": "me",
    "montserrat": "ms",
    "morocco": "ma",
    "mozambique": "mz",
    "myanmar": "mm",
    "namibia": "na",
    "nauru": "nr",
    "nepal": "np",
    "netherlands (the)": "nl",
    "new caledonia": "nc",
    "new zealand": "nz",
    "nicaragua": "ni",
    "niger (the)": "ne",
    "nigeria": "ng",
    "niue": "nu",
    "norfolk island": "nf",
    "north macedonia": "mk",
    "northern mariana islands (the)": "mp",
    "norway": "no",
    "oman": "om",
    "pakistan": "pk",
    "palau": "pw",
    "palestine, state of": "ps",
    "panama": "pa",
    "papua new guinea": "pg",
    "paraguay": "py",
    "peru": "pe",
    "philippines (the)": "ph",
    "pitcairn": "pn",
    "poland": "pl",
    "portugal": "pt",
    "puerto rico": "pr",
    "qatar": "qa",
    "romania": "ro",
    "russian federation (the)": "ru",
    "rwanda": "rw",
    "réunion": "re",
    "saint barthélemy": "bl",
    "saint helena, ascension and tristan da cunha": "sh",
    "saint kitts and nevis": "kn",
    "saint lucia": "lc",
    "saint martin (french part)": "mf",
    "saint pierre and miquelon": "pm",
    "saint vincent and the grenadines": "vc",
    "samoa": "ws",
    "san marino": "sm",
    "sao tome and principe": "st",
    "saudi arabia": "sa",
    "senegal": "sn",
    "serbia": "rs",
    "seychelles": "sc",
    "sierra leone": "sl",
    "singapore": "sg",
    "sint maarten (dutch part)": "sx",
    "slovakia": "sk",
    "slovenia": "si",
    "solomon islands": "sb",
    "somalia": "so",
    "south africa": "za",
    "south georgia and the south sandwich islands": "gs",
    "south sudan": "ss",
    "spain": "es",
    "sri lanka": "lk",
    "sudan (the)": "sd",
    "suriname": "sr",
    "svalbard and jan mayen": "sj",
    "sweden": "se",
    "switzerland": "ch",
    "syrian arab republic": "sy",
    "taiwan (province of china)": "tw",
    "tajikistan": "tj",
    "tanzania, united republic of": "tz",
    "thailand": "th",
    "timor-leste": "tl",
    "togo": "tg",
    "tokelau": "tk",
    "tonga": "to",
    "trinidad and tobago": "tt",
    "tunisia": "tn",
    "turkmenistan": "tm",
    "turks and caicos islands (the)": "tc",
    "tuvalu": "tv",
    "türkiye": "tr",
    "uganda": "ug",
    "ukraine": "ua",
    "united arab emirates (the)": "ae",
    "united kingdom of great britain and northern ireland (the)": "gb",
    "united states minor outlying islands (the)": "um",
    "united states of america (the)": "us",
    "uruguay": "uy",
    "uzbekistan": "uz",
    "vanuatu": "vu",
    "venezuela (bolivarian republic of)": "ve",
    "viet nam": "vn",
    "virgin islands (british)": "vg",
    "virgin islands (u.s.)": "vi",
    "wallis and futuna": "wf",
    "western sahara": "eh",
    "yemen": "ye",
    "zambia": "zm",
    "zimbabwe": "zw",
    "åland islands": "ax",
    # Friendly aliases (short/common names) alongside the official ISO names above.
    "united states": "us",
    "united kingdom": "gb",
    "uk": "gb",
    "south korea": "kr",
    "north korea": "kp",
    "russia": "ru",
    "vietnam": "vn",
    "laos": "la",
    "syria": "sy",
    "iran": "ir",
    "bolivia": "bo",
    "venezuela": "ve",
    "tanzania": "tz",
    "moldova": "md",
    "czech republic": "cz",
    "ivory coast": "ci",
    "cape verde": "cv",
    "swaziland": "sz",
    "brunei": "bn",
    "the bahamas": "bs",
    "the gambia": "gm",
}
LANGUAGE_CODES = {
    "afar": "aa",
    "abkhazian": "ab",
    "avestan": "ae",
    "afrikaans": "af",
    "akan": "ak",
    "amharic": "am",
    "aragonese": "an",
    "arabic": "ar",
    "arabic (u.a.e.)": "ar",
    "arabic (bahrain)": "ar",
    "arabic (algeria)": "ar",
    "arabic (egypt)": "ar",
    "arabic (iraq)": "ar",
    "arabic (jordan)": "ar",
    "arabic (kuwait)": "ar",
    "arabic (lebanon)": "ar",
    "arabic (libya)": "ar",
    "arabic (morocco)": "ar",
    "arabic (oman)": "ar",
    "arabic (qatar)": "ar",
    "arabic (saudi arabia)": "ar",
    "arabic (syria)": "ar",
    "arabic (tunisia)": "ar",
    "arabic (yemen)": "ar",
    "assamese": "as",
    "avaric": "av",
    "aymara": "ay",
    "azeri": "az",
    "bashkir": "ba",
    "belarusian": "be",
    "bulgarian": "bg",
    "bihari": "bh",
    "bislama": "bi",
    "bambara": "bm",
    "bengali": "bn",
    "tibetan": "bo",
    "breton": "br",
    "bosnian": "bs",
    "catalan": "ca",
    "chechen": "ce",
    "chamorro": "ch",
    "corsican": "co",
    "cree": "cr",
    "czech": "cs",
    "church slavonic": "cu",
    "chuvash": "cv",
    "welsh": "cy",
    "danish": "da",
    "german": "de",
    "german (austria)": "de",
    "german (switzerland)": "de",
    "german (germany)": "de",
    "german (liechtenstein)": "de",
    "german (luxembourg)": "de",
    "divehi": "dv",
    "bhutani": "dz",
    "ewe": "ee",
    "greek": "el",
    "english": "en",
    "english (australia)": "en",
    "english (belize)": "en",
    "english (canada)": "en",
    "english (caribbean)": "en",
    "english (united kingdom)": "en",
    "english (ireland)": "en",
    "english (jamaica)": "en",
    "english (new zealand)": "en",
    "english (philippines)": "en",
    "english (trinidad and tobago)": "en",
    "english (united states)": "en",
    "english (south africa)": "en",
    "english (zimbabwe)": "en",
    "esperanto": "eo",
    "spanish": "es",
    "spanish (argentina)": "es",
    "spanish (bolivia)": "es",
    "spanish (chile)": "es",
    "spanish (colombia)": "es",
    "spanish (costa rica)": "es",
    "spanish (dominican republic)": "es",
    "spanish (ecuador)": "es",
    "spanish (spain)": "es",
    "spanish (guatemala)": "es",
    "spanish (honduras)": "es",
    "spanish (mexico)": "es",
    "spanish (nicaragua)": "es",
    "spanish (panama)": "es",
    "spanish (peru)": "es",
    "spanish (puerto rico)": "es",
    "spanish (paraguay)": "es",
    "spanish (el salvador)": "es",
    "spanish (united states)": "es",
    "spanish (uruguay)": "es",
    "spanish (venezuela)": "es",
    "estonian": "et",
    "basque": "eu",
    "farsi": "fa",
    "fulah": "ff",
    "finnish": "fi",
    "fiji": "fj",
    "faroese": "fo",
    "french": "fr",
    "french (belgium)": "fr",
    "french (canada)": "fr",
    "french (switzerland)": "fr",
    "french (france)": "fr",
    "french (luxembourg)": "fr",
    "french (monaco)": "fr",
    "frisian": "fy",
    "irish": "ga",
    "gaelic": "gd",
    "galician": "gl",
    "guarani": "gn",
    "gujarati": "gu",
    "manx": "gv",
    "hausa": "ha",
    "hebrew": "he",
    "hindi": "hi",
    "hiri motu": "ho",
    "croatian": "hr",
    "croatian (bosnia and herzegovina)": "hr",
    "croatian (croatia)": "hr",
    "haitian": "ht",
    "hungarian": "hu",
    "armenian": "hy",
    "herero": "hz",
    "interlingua": "ia",
    "indonesian": "id",
    "interlingue": "ie",
    "igbo": "ig",
    "sichuan yi": "ii",
    "inupiak": "ik",
    "ido": "io",
    "icelandic": "is",
    "italian": "it",
    "italian (switzerland)": "it",
    "italian (italy)": "it",
    "inuktitut": "iu",
    "japanese": "ja",
    "yiddish": "yi",
    "javanese": "jv",
    "georgian": "ka",
    "kongo": "kg",
    "kikuyu": "ki",
    "kuanyama": "kj",
    "kazakh": "kk",
    "greenlandic": "kl",
    "cambodian": "km",
    "kannada": "kn",
    "korean": "ko",
    "konkani": "kok",
    "kanuri": "kr",
    "kashmiri": "ks",
    "kurdish": "ku",
    "komi": "kv",
    "cornish": "kw",
    "kirghiz": "ky",
    "kyrgyz": "kz",
    "latin": "la",
    "luxembourgish": "lb",
    "ganda": "lg",
    "limburgan": "li",
    "lingala": "ln",
    "laothian": "lo",
    "slovenian": "sl",
    "lithuanian": "lt",
    "luba-katanga": "lu",
    "latvian": "lv",
    "malagasy": "mg",
    "marshallese": "mh",
    "maori": "mi",
    "fyro macedonian": "mk",
    "malayalam": "ml",
    "mongolian": "mn",
    "moldavian": "mo",
    "marathi": "mr",
    "malay": "ms",
    "malay (brunei darussalam)": "ms",
    "malay (malaysia)": "ms",
    "maltese": "mt",
    "burmese": "my",
    "nauru": "na",
    "norwegian (bokmal)": "nb",
    "north ndebele": "nd",
    "nepali (india)": "ne",
    "ndonga": "ng",
    "dutch": "nl",
    "dutch (belgium)": "nl",
    "dutch (netherlands)": "nl",
    "norwegian (nynorsk)": "nn",
    "norwegian": "no",
    "south ndebele": "nr",
    "northern sotho": "ns",
    "navajo": "nv",
    "chichewa": "ny",
    "occitan": "oc",
    "ojibwa": "oj",
    "(afan)/oromoor/oriya": "om",
    "oriya": "or",
    "ossetian": "os",
    "punjabi": "pa",
    "pali": "pi",
    "polish": "pl",
    "pashto/pushto": "ps",
    "portuguese": "pt",
    "portuguese (brazil)": "pt",
    "portuguese (portugal)": "pt",
    "quechua": "qu",
    "quechua (bolivia)": "qu",
    "quechua (ecuador)": "qu",
    "quechua (peru)": "qu",
    "rhaeto-romanic": "rm",
    "kirundi": "rn",
    "romanian": "ro",
    "russian": "ru",
    "kinyarwanda": "rw",
    "sanskrit": "sa",
    "sorbian": "sb",
    "sardinian": "sc",
    "sindhi": "sd",
    "sami": "se",
    "sami (finland)": "se",
    "sami (norway)": "se",
    "sami (sweden)": "se",
    "sangro": "sg",
    "serbo-croatian": "sh",
    "singhalese": "si",
    "slovak": "sk",
    "samoan": "sm",
    "shona": "sn",
    "somali": "so",
    "albanian": "sq",
    "serbian": "sr",
    "serbian (bosnia and herzegovina)": "sr",
    "serbian (serbia and montenegro)": "sr",
    "siswati": "ss",
    "sesotho": "st",
    "sundanese": "su",
    "swedish": "sv",
    "swedish (finland)": "sv",
    "swedish (sweden)": "sv",
    "swahili": "sw",
    "sutu": "sx",
    "syriac": "syr",
    "tamil": "ta",
    "telugu": "te",
    "tajik": "tg",
    "thai": "th",
    "tigrinya": "ti",
    "turkmen": "tk",
    "tagalog": "tl",
    "tswana": "tn",
    "tonga": "to",
    "turkish": "tr",
    "tsonga": "ts",
    "tatar": "tt",
    "twi": "tw",
    "tahitian": "ty",
    "uighur": "ug",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "venda": "ve",
    "vietnamese": "vi",
    "volapuk": "vo",
    "walloon": "wa",
    "wolof": "wo",
    "xhosa": "xh",
    "yoruba": "yo",
    "zhuang": "za",
    "chinese": "zh",
    "chinese (china)": "zh",
    "chinese (hong kong sar)": "zh",
    "chinese (macau sar)": "zh",
    "chinese (singapore)": "zh",
    "chinese (taiwan)": "zh",
    "zulu": "zu",
}

QUESTION_CLUSTER_DISTANCE = 0.30
FUZZY_MATCH_THRESHOLD = 82
MAX_ROWS = 60
REQUEST_TIMEOUT = 20
REQUEST_RETRIES = 2

# Seed prefixes for autosuggest expansion. Structural (question words +
# generic modifiers), niche-agnostic.
AUTOSUGGEST_SEEDS = [
    "{kw}",
    "how {kw}",
    "what {kw}",
    "why {kw}",
    "is {kw}",
    "can {kw}",
    "best {kw}",
    "{kw} vs",
    "{kw} for",
    "{kw} without",
    "{kw} cost",
    "{kw} benefits",
]

# Generic question-type patterns (order matters: first match wins).
QUESTION_TYPE_RULES = [
    ("Comparison", re.compile(r"\bvs\b|\bversus\b|\bor\b|\bdifference\b|\bbetter\b|\bcompare\b", re.I)),
    ("Cost", re.compile(r"\bcost\b|\bprice\b|\bworth\b|\bcheap\b|\bexpensive\b|\bfees?\b", re.I)),
    ("Quantity", re.compile(r"\bhow much\b|\bhow many\b|\bhow often\b|\bdosage\b|\bdose\b", re.I)),
    ("Safety", re.compile(r"\bsafe\b|\bside effects?\b|\brisks?\b|\bdanger\w*\b|\bcause\b|\bharmful\b|\bbad for\b", re.I)),
    ("Process", re.compile(r"^how (to|do|can|should)\b", re.I)),
    ("Timing", re.compile(r"\bwhen\b|\bhow long\b", re.I)),
    ("Suitability", re.compile(r"\bfor (women|men|beginners?|teens?|kids?|seniors?)\b|\bcan i\b|\bshould i\b", re.I)),
    ("Definition", re.compile(r"^(what|which) (is|are|does|do)\b|\bmeaning\b|\bdefine\b", re.I)),
    ("Reason", re.compile(r"^why\b", re.I)),
]


@dataclass
class QuestionRecord:
    text: str
    sources: Set[str] = field(default_factory=set)
    competitor_hits: int = 0


def _normalize_code(value: str, mapping: Dict[str, str], default: str) -> str:
    cleaned = (value or "").strip().lower()
    if len(cleaned) == 2:
        return mapping.get(cleaned, cleaned)
    return mapping.get(cleaned, default)


def classify_question_type(question: str) -> str:
    text = compact_text(question)
    for label, pattern in QUESTION_TYPE_RULES:
        if pattern.search(text):
            return label
    return "Other"


def _looks_like_question(text: str) -> bool:
    text = compact_text(text)
    if text.endswith("?"):
        return True
    return bool(re.match(
        r"^(what|which|why|how|when|where|who|is|are|does|do|did|"
        r"should|can|could|would|will|has|have)\b",
        text,
        re.I,
    ))


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class SearchIntelligenceCollector:

    def __init__(self, serper_key: str):
        self.serper_key = serper_key

    def fetch_serp(
        self,
        keyword: str,
        country: str,
        language: str,
    ) -> Dict[str, Any]:
        """Returns raw peopleAlsoAsk and relatedSearches from Serper."""
        payload = {
            "q": keyword,
            "gl": _normalize_code(country, COUNTRY_CODES, "us"),
            "hl": _normalize_code(language, LANGUAGE_CODES, "en"),
        }
        for attempt in range(REQUEST_RETRIES + 1):
            response = requests.post(
                SERPER_ENDPOINT,
                headers={
                    "X-API-KEY": self.serper_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        return {}

    def fetch_autosuggest(
        self,
        keyword: str,
        country: str,
        language: str,
    ) -> List[str]:
        gl = _normalize_code(country, COUNTRY_CODES, "us")
        hl = _normalize_code(language, LANGUAGE_CODES, "en")
        suggestions: List[str] = []
        seen: Set[str] = set()
        for template in AUTOSUGGEST_SEEDS:
            seed = template.format(kw=keyword)
            for suggestion in self._autosuggest_one(seed, gl, hl):
                key = suggestion.lower()
                if key not in seen:
                    seen.add(key)
                    suggestions.append(suggestion)
        return suggestions

    @staticmethod
    def _autosuggest_one(query: str, gl: str, hl: str) -> List[str]:
        try:
            response = requests.get(
                AUTOSUGGEST_ENDPOINT,
                params={"client": "firefox", "q": query, "gl": gl, "hl": hl},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = json.loads(response.text)
            # Format: [query, [suggestion, suggestion, ...], ...]
            return [compact_text(s) for s in data[1] if compact_text(s)]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Parsing / extraction
# ---------------------------------------------------------------------------


def parse_paa(serp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for item in serp.get("peopleAlsoAsk", []) or []:
        question = compact_text(item.get("question", ""))
        if not question:
            continue
        rows.append(
            {
                "question": question,
                "type": classify_question_type(question),
                "snippet": compact_text(item.get("snippet", "")),
                "source_url": item.get("link", ""),
            }
        )
    return rows


def parse_related(serp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    seen: Set[str] = set()
    for item in serp.get("relatedSearches", []) or []:
        query = compact_text(item.get("query", "")) if isinstance(item, dict) else compact_text(item)
        if not query or query.lower() in seen:
            continue
        seen.add(query.lower())
        rows.append({"query": query})
    return rows


def _faq_from_jsonld(html: str) -> List[str]:
    questions: List[str] = []
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        for block in data if isinstance(data, list) else [data]:
            questions.extend(_walk_jsonld_for_questions(block))
    return questions


def _walk_jsonld_for_questions(node: Any) -> List[str]:
    found: List[str] = []
    if isinstance(node, dict):
        node_type = node.get("@type", "")
        types = node_type if isinstance(node_type, list) else [node_type]
        if "Question" in types and node.get("name"):
            found.append(compact_text(node["name"]))
        for value in node.values():
            found.extend(_walk_jsonld_for_questions(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_jsonld_for_questions(item))
    return found


def _faq_from_structure(competitor: Dict[str, Any]) -> List[str]:
    structure = competitor.get("structure", {}) or {}
    headings: List[str] = []
    for values in (structure.get("headings", {}) or {}).values():
        headings.extend(values)
    headings.extend(structure.get("logical_headings", []) or [])
    return [
        compact_text(h)
        for h in headings
        if compact_text(h).endswith("?")
    ]


def _candidate_html(competitor: Dict[str, Any]) -> str:
    for key in ("html", "raw_html", "content_html", "body_html", "scraped_html"):
        value = competitor.get(key)
        if isinstance(value, str) and "<" in value and ">" in value:
            return value
    return ""


def extract_faqs(competitors: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    FAQ questions from competitor pages (JSON-LD FAQPage schema + visible
    question headings), deduplicated across competitors with a count of
    how many competitors ask each.
    """
    per_question_competitors: Dict[str, Set[int]] = defaultdict(set)
    canonical: List[str] = []

    def register(question: str, index: int) -> None:
        if not question or not _looks_like_question(question):
            return
        match = None
        for existing in canonical:
            if fuzz.token_sort_ratio(existing, question) >= FUZZY_MATCH_THRESHOLD:
                match = existing
                break
        key = match or question
        if key not in canonical:
            canonical.append(key)
        per_question_competitors[key].add(index)

    for index, competitor in enumerate(competitors):
        html = _candidate_html(competitor)
        if html:
            for question in _faq_from_jsonld(html):
                register(question, index)
        for question in _faq_from_structure(competitor):
            register(question, index)

    rows = [
        {
            "question": question,
            "type": classify_question_type(question),
            "competitors": len(indexes),
        }
        for question, indexes in per_question_competitors.items()
    ]
    rows.sort(key=lambda row: row["competitors"], reverse=True)
    return rows[:MAX_ROWS]


# ---------------------------------------------------------------------------
# Merge + cluster into Question Intelligence
# ---------------------------------------------------------------------------


def _collect_question_records(
    paa: List[Dict[str, Any]],
    related: List[Dict[str, Any]],
    autosuggest: List[str],
    faqs: List[Dict[str, Any]],
) -> Dict[str, QuestionRecord]:
    records: Dict[str, QuestionRecord] = {}

    def add(text: str, source: str, competitor_hits: int = 0) -> None:
        text = compact_text(text)
        if not text:
            return
        key = text.lower()
        record = records.get(key)
        if record is None:
            record = QuestionRecord(text=text)
            records[key] = record
        record.sources.add(source)
        record.competitor_hits = max(record.competitor_hits, competitor_hits)

    for row in paa:
        add(row["question"], "paa")
    for row in related:
        if _looks_like_question(row["query"]):
            add(row["query"], "related")
    for suggestion in autosuggest:
        if _looks_like_question(suggestion):
            add(suggestion, "autosuggest")
    for row in faqs:
        add(row["question"], "faq", row.get("competitors", 0))
    return records


def build_question_intelligence(
    records: Dict[str, QuestionRecord],
) -> List[Dict[str, Any]]:
    """
    Cluster the merged questions so near-duplicates collapse into one
    intent, then report each cluster's representative question, type,
    contributing sources, and max competitor coverage.
    """
    items = list(records.values())
    if not items:
        return []
    texts = [item.text for item in items]
    if len(items) == 1:
        clusters = [[0]]
    else:
        try:
            embeddings = get_sentence_model().encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        except TypeError:
            embeddings = get_sentence_model().encode(
                texts, normalize_embeddings=True
            )
        embeddings = np.array(embeddings)
        labels = make_agglomerative(QUESTION_CLUSTER_DISTANCE).fit_predict(embeddings)
        grouped: Dict[int, List[int]] = defaultdict(list)
        for position, label in enumerate(labels):
            grouped[int(label)].append(position)
        clusters = list(grouped.values())

    rows = []
    for positions in clusters:
        members = [items[i] for i in positions]
        # Representative = shortest question (usually the cleanest form).
        representative = min(members, key=lambda item: len(item.text))
        sources: Set[str] = set()
        competitor_hits = 0
        for member in members:
            sources |= member.sources
            competitor_hits = max(competitor_hits, member.competitor_hits)
        rows.append(
            {
                "question": representative.text,
                "type": classify_question_type(representative.text),
                "sources": sorted(sources),
                "variants": len(members),
                "competitor_coverage": competitor_hits,
            }
        )
    rows.sort(key=lambda row: (len(row["sources"]), row["variants"]), reverse=True)
    return rows[:MAX_ROWS]


# ---------------------------------------------------------------------------
# Signal matrix: Google x Reddit x Competitors
# ---------------------------------------------------------------------------


def _reddit_signal(
    question: str,
    reddit_questions: List[str],
) -> str:
    best = 0
    for candidate in reddit_questions:
        score = fuzz.token_set_ratio(question, candidate)
        if score > best:
            best = score
    if best >= 88:
        return "High"
    if best >= FUZZY_MATCH_THRESHOLD:
        return "Low"
    return ""


def build_signal_matrix(
    question_intelligence: List[Dict[str, Any]],
    community: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    reddit_questions = [
        compact_text(row.get("question", ""))
        for row in ((community or {}).get("questions", []) or [])
    ]
    reddit_questions = [q for q in reddit_questions if q]

    rows = []
    for item in question_intelligence:
        google_sources = {
            s for s in item["sources"] if s in ("paa", "related", "autosuggest")
        }
        google = "Yes" if google_sources else ""
        reddit = _reddit_signal(item["question"], reddit_questions)
        rows.append(
            {
                "question": item["question"],
                "type": item["type"],
                "google": google,
                "reddit": reddit or "-",
                "competitors": item["competitor_coverage"],
                "opportunity": _opportunity_score(
                    bool(google_sources), reddit, item["competitor_coverage"]
                ),
            }
        )
    # Highest opportunity first: strong Google + Reddit demand, weak
    # competitor coverage.
    rows.sort(key=lambda row: row["opportunity"], reverse=True)
    return rows[:MAX_ROWS]


def _opportunity_score(has_google: bool, reddit: str, competitors: int) -> int:
    score = 0
    if has_google:
        score += 2
    if reddit == "High":
        score += 3
    elif reddit == "Low":
        score += 1
    # Low competitor coverage is the opportunity.
    score += max(0, 3 - competitors)
    return score


# ---------------------------------------------------------------------------
# Aggregator / entry point
# ---------------------------------------------------------------------------


def run_search_intelligence(
    keyword: str,
    serper_key: str,
    country: str,
    language: str,
    competitors: Optional[Sequence[Dict[str, Any]]] = None,
    community: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Public entry point used by app.py.

    Returns:
    {
        "paa": [...],
        "related_searches": [...],
        "autosuggest": [...],
        "faqs": [...],
        "question_intelligence": [...],
        "signal_matrix": [...]
    }
    """
    collector = SearchIntelligenceCollector(serper_key)

    try:
        serp = collector.fetch_serp(keyword, country, language)
    except Exception:
        serp = {}
    paa = parse_paa(serp)
    related = parse_related(serp)

    try:
        autosuggest = collector.fetch_autosuggest(keyword, country, language)
    except Exception:
        autosuggest = []

    faqs = extract_faqs(competitors or [])

    records = _collect_question_records(paa, related, autosuggest, faqs)
    question_intelligence = build_question_intelligence(records)
    signal_matrix = build_signal_matrix(question_intelligence, community)

    return {
        "paa": paa,
        "related_searches": related,
        "autosuggest": [{"suggestion": s} for s in autosuggest],
        "faqs": faqs,
        "question_intelligence": question_intelligence,
        "signal_matrix": signal_matrix,
    }