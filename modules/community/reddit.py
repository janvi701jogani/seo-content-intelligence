"""
Community Intelligence module.

Self-contained: collection (Serper -> site:reddit.com -> praw) and the
12-layer community processing pipeline both live here. Reuses utilities
from modules.extractor (which is NOT modified):

Raw Reddit Threads
        |
        +-- Question Extractor
        +-- Pain Point Extractor
        +-- Recommendation Extractor
        +-- Entity Extractor (brands, reuses spaCy pipeline)
        +-- Feature Extractor
        +-- Decision Factor Extractor
        +-- Vocabulary Extractor
        +-- Mistake Extractor
        +-- Myth Extractor
        +-- Experience Extractor
        +-- Aggregator (statistics + competitor gap)
                 |
                 v
      Community Intelligence Dashboard

Every row produced by every layer below carries a "thread_links" field
(one or more Reddit permalinks, joined with "; ") pointing back to the
exact thread(s) it was derived from, so nothing in the dashboard is an
unsourced claim.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set
from urllib.parse import urlparse

import numpy as np
import praw
import requests
from rapidfuzz import fuzz

from modules.extractor import (
    cap_label,
    compact_text,
    generate_label_from_content,
    get_nlp,
    get_sentence_model,
    get_yake_extractor,
    make_agglomerative,
    normalize_entity_text,
    valid_entity_phrase,
)

SERPER_ENDPOINT = "https://google.serper.dev/search"

# Streamlit's selectbox over a dict returns the KEY ("India"), not the
# value ("in"). Serper requires 2-letter gl/hl codes, so normalize both
# forms here to keep this module self-contained.

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


def _normalize_geo_code(value: str, mapping: Dict[str, str]) -> str:
    """
    Returns a 2-letter code, or "" when the value can't be resolved.
    Unresolvable values are OMITTED from the Serper payload entirely --
    the verified-working minimal payload is {"q": ...} alone, so it is
    always safer to drop a parameter than to send a guessed one.
    """
    cleaned = (value or "").strip().lower()
    if len(cleaned) == 2:
        return mapping.get(cleaned, cleaned)
    return mapping.get(cleaned, "")

# -----------------------------
# Community processing settings
# -----------------------------

MAX_COMMUNITY_SENTENCES = 4000
MIN_SENTENCE_WORDS = 4
MAX_SENTENCE_WORDS = 60
QUESTION_CLUSTER_DISTANCE = 0.35
PATTERN_CLUSTER_DISTANCE = 0.40
MAX_ROWS_PER_LAYER = 30
GAP_MATCH_THRESHOLD = 85

# Generic discourse patterns (structural, niche-agnostic).

BOT_BOILERPLATE = re.compile(
    r"i am a bot|action (was )?performed automatically|contact the moderators",
    re.IGNORECASE,
)

QUESTION_START = re.compile(
    r"^(what|which|why|how|when|where|who|is|are|does|do|did|should|"
    r"can|could|would|will|has|have|am)\b",
    re.IGNORECASE,
)

PAIN_PATTERN = re.compile(
    r"\b(problem|issue|annoy\w*|frustrat\w*|too expensive|overpriced|"
    r"waste of money|doesn'?t work|didn'?t work|stopped working|hate|"
    r"worst|terrible|awful|disappoint\w*|scam|fake|broke|struggl\w*|"
    r"can'?t stand)\b",
    re.IGNORECASE,
)

RECOMMENDATION_PATTERN = re.compile(
    r"\b(you should|i('| wou|'d| would)?d? recommend|my advice|pro tip|"
    r"make sure|always|never|avoid|stick (to|with)|go (with|for)|"
    r"buy from|start with|check the|worth it to|do yourself a favor)\b",
    re.IGNORECASE,
)

MISTAKE_PATTERN = re.compile(
    r"\b(my mistake|i regret|i wish i (had|knew)|i made the mistake|"
    r"learned (this|it|that) the hard way|don'?t make my mistake|"
    r"i messed up|i should have|i shouldn'?t have)\b",
    re.IGNORECASE,
)

MYTH_PATTERN = re.compile(
    r"\b(myth|misconception|contrary to popular belief|"
    r"people (think|believe|say|assume)|it'?s not true|"
    r"not actually true|commonly believed|despite what)\b",
    re.IGNORECASE,
)

EXPERIENCE_PATTERN = re.compile(
    r"\b(i noticed|i felt|i('| ha)?ve been using|after (using|taking|"
    r"switching|buying)|in my experience|for me it|i tried|"
    r"worked for me|helped me|i experienced|been using (it|this) for)\b",
    re.IGNORECASE,
)

DECISION_PATTERN = re.compile(
    r"\b(i chose|i went with|i picked|i prefer|i switched|"
    r"deciding factor|main reason|because of the|instead of|"
    r"what made me (buy|choose|pick|switch))\b",
    re.IGNORECASE,
)

EVALUATIVE_PATTERN = re.compile(
    r"\b(too|really|very|so|extremely|way too)\b",
    re.IGNORECASE,
)

POSITIVE_WORDS = {
    "good", "great", "love", "loved", "best", "better", "amazing",
    "excellent", "happy", "works", "worked", "helpful", "effective",
    "worth", "recommend", "solid", "reliable", "easy", "perfect",
}

NEGATIVE_WORDS = {
    "bad", "worst", "hate", "hated", "terrible", "awful", "poor",
    "disappointed", "disappointing", "waste", "scam", "fake", "broken",
    "broke", "issue", "problem", "expensive", "overpriced", "useless",
    "hard", "difficult", "annoying",
}


@dataclass
class RedditComment:
    body: str
    score: int
    author: str
    created_utc: float
    depth: int


@dataclass
class RedditThread:
    title: str
    subreddit: str
    url: str
    permalink: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    selftext: str
    comments: List[RedditComment]


class RedditCollector:

    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    # -----------------------------
    # Primary discovery: Google (Serper) -> site:reddit.com -> URLs
    # -----------------------------

    def search_via_google(
        self,
        keyword: str,
        serper_key: str,
        limit: int = 10,
        country: str = "in",
        language: str = "en",
    ) -> List[RedditThread]:
        """
        Keyword -> Serper Search -> site:reddit.com -> Extract Reddit URLs
        -> submission(url) -> Download comments.

        Google's ranking decides which Reddit threads matter for the
        keyword, instead of Reddit's own search.
        """
        urls = self._collect_reddit_urls(
            keyword=keyword,
            serper_key=serper_key,
            limit=limit,
            country=country,
            language=language,
        )
        threads = []
        for url in urls:
            try:
                submission = self.reddit.submission(url=url)
                threads.append(self._build_thread(submission))
            except Exception:
                continue
        return threads

    def _collect_reddit_urls(
        self,
        keyword: str,
        serper_key: str,
        limit: int,
        country: str,
        language: str,
    ) -> List[str]:
        # Serper free accounts reject advanced operators like site:
        # ("Query pattern not allowed for free accounts"). Use the plain
        # "<keyword> reddit" pattern instead -- Google ranks Reddit
        # threads high for it, and _normalize_reddit_url() already
        # discards every non-Reddit result. num is raised because the
        # results are now mixed with non-Reddit pages.
        payload: Dict[str, Any] = {
            "q": f"{keyword} reddit",
            "num": max(limit * 3, 30),
        }
        gl = _normalize_geo_code(country, COUNTRY_CODES)
        hl = _normalize_geo_code(language, LANGUAGE_CODES)
        if gl:
            payload["gl"] = gl
        if hl:
            payload["hl"] = hl
        response = requests.post(
            SERPER_ENDPOINT,
            headers={
                "X-API-KEY": serper_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        organic = response.json().get("organic", [])

        urls: List[str] = []
        seen: set = set()
        for result in organic:
            url = self._normalize_reddit_url(result.get("link", ""))
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
            if len(urls) >= limit:
                break
        return urls

    @staticmethod
    def _normalize_reddit_url(url: str) -> str:
        """
        Keep only thread URLs (/r/<sub>/comments/<id>/...), drop subreddit
        home pages, wikis, user pages, and media links. Strip query params
        and normalize amp/old/m subdomains so praw accepts the permalink
        and duplicates collapse.
        """
        if not url or "/comments/" not in url:
            return ""
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if "reddit.com" not in host:
            return ""
        return f"https://www.reddit.com{parsed.path}"

    # -----------------------------
    # Fallback discovery: Reddit's own search
    # -----------------------------

    def search(
        self,
        keyword: str,
        limit: int = 10,
        sort: str = "relevance",
        time_filter: str = "all"
    ) -> List[RedditThread]:
        threads = []
        submissions = self.reddit.subreddit("all").search(
            keyword,
            sort=sort,
            time_filter=time_filter,
            limit=limit,
        )
        for submission in submissions:
            threads.append(self._build_thread(submission))
        return threads

    # -----------------------------
    # Shared submission -> RedditThread builder
    # -----------------------------

    @staticmethod
    def _build_thread(submission: Any) -> RedditThread:
        submission.comments.replace_more(limit=0)
        comments = []
        for comment in submission.comments.list():
            comments.append(
                RedditComment(
                    body=comment.body,
                    score=comment.score,
                    author=str(comment.author),
                    created_utc=comment.created_utc,
                    depth=getattr(comment, "depth", 0),
                )
            )
        return RedditThread(
            title=submission.title,
            subreddit=str(submission.subreddit),
            url=submission.url,
            permalink=f"https://reddit.com{submission.permalink}",
            author=str(submission.author),
            score=submission.score,
            upvote_ratio=submission.upvote_ratio,
            num_comments=submission.num_comments,
            created_utc=submission.created_utc,
            selftext=submission.selftext,
            comments=comments,
        )

    @staticmethod
    def to_dict(threads: List[RedditThread]) -> List[Dict[str, Any]]:
        return [
            {
                **asdict(thread),
                "comments": [asdict(c) for c in thread.comments],
            }
            for thread in threads
        ]


# ---------------------------------------------------------------------------
# Community Intelligence pipeline
# ---------------------------------------------------------------------------


def _sentence_polarity(sentence: str) -> int:
    words = set(re.findall(r"[a-z']+", sentence.lower()))
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)
    if positive > negative:
        return 1
    if negative > positive:
        return -1
    return 0


def _collect_sentences(threads: Sequence[RedditThread]) -> List[Dict[str, Any]]:
    """
    Flatten every thread (title + selftext + comments) into sentence
    records: {"thread": index, "text": sentence, "polarity": -1|0|1}.
    Bot boilerplate and deleted content are dropped. "thread" is the
    index into the `threads` sequence this sentence came from, which is
    how every downstream row traces back to a permalink.
    """
    records: List[Dict[str, Any]] = []
    for index, thread in enumerate(threads):
        texts = [thread.title, thread.selftext]
        texts.extend(comment.body for comment in thread.comments)
        for text in texts:
            cleaned = compact_text(text)
            if not cleaned or cleaned.lower() in ("[deleted]", "[removed]"):
                continue
            if BOT_BOILERPLATE.search(cleaned):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned):
                sentence = compact_text(sentence)
                word_count = len(sentence.split())
                if word_count < MIN_SENTENCE_WORDS or word_count > MAX_SENTENCE_WORDS:
                    continue
                records.append(
                    {
                        "thread": index,
                        "text": sentence,
                        "polarity": _sentence_polarity(sentence),
                    }
                )
                if len(records) >= MAX_COMMUNITY_SENTENCES:
                    return records
    return records


def _thread_links(indices: Any, threads: Sequence[RedditThread]) -> str:
    """
    Resolves a set/iterable of thread indices to their Reddit permalinks,
    deduped and in a stable order, joined into one string so it drops
    straight into a table cell (both the Streamlit dataframe view and the
    Excel export already know how to display a plain string column).
    """
    links: List[str] = []
    seen: Set[str] = set()
    for idx in sorted(set(indices or [])):
        if 0 <= idx < len(threads):
            link = threads[idx].permalink
            if link and link not in seen:
                seen.add(link)
                links.append(link)
    return "; ".join(links)


def _cluster_records(
    records: List[Dict[str, Any]],
    distance_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Embed sentences (one batched encode call) and cluster with the same
    complete-linkage agglomerative setup the extractor uses. Returns
    clusters: {"sentences", "threads", "representative", "polarities"}.
    "threads" is the set of thread INDICES contributing to the cluster.
    """
    if not records:
        return []
    sentences = [record["text"] for record in records]
    try:
        embeddings = get_sentence_model().encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except TypeError:
        embeddings = get_sentence_model().encode(
            sentences,
            normalize_embeddings=True,
        )
    embeddings = np.array(embeddings)
    if len(records) == 1:
        labels = np.array([0])
    else:
        labels = make_agglomerative(distance_threshold).fit_predict(embeddings)
    grouped: Dict[int, List[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        grouped[int(label)].append(position)
    clusters = []
    for positions in grouped.values():
        member_sentences = [sentences[i] for i in positions]
        member_embeddings = embeddings[positions]
        centroid = np.mean(member_embeddings, axis=0)
        similarities = member_embeddings @ centroid
        representative = member_sentences[int(np.argmax(similarities))]
        clusters.append(
            {
                "sentences": member_sentences,
                "threads": {records[i]["thread"] for i in positions},
                "polarities": [records[i]["polarity"] for i in positions],
                "representative": representative,
            }
        )
    clusters.sort(key=lambda cluster: len(cluster["sentences"]), reverse=True)
    return clusters


def _cluster_label(cluster: Dict[str, Any]) -> str:
    label = generate_label_from_content(cluster["sentences"])
    if label:
        return label
    return cap_label(cluster["representative"])


def _pattern_layer(
    records: List[Dict[str, Any]],
    pattern: re.Pattern,
    label_key: str,
    count_key: str,
    threads: Sequence[RedditThread],
) -> List[Dict[str, Any]]:
    matched = [record for record in records if pattern.search(record["text"])]
    clusters = _cluster_records(matched[:500], PATTERN_CLUSTER_DISTANCE)
    rows = []
    for cluster in clusters[:MAX_ROWS_PER_LAYER]:
        rows.append(
            {
                label_key: _cluster_label(cluster),
                count_key: len(cluster["sentences"]),
                "threads": len(cluster["threads"]),
                "example": cluster["representative"],
                "thread_links": _thread_links(cluster["threads"], threads),
            }
        )
    return rows


# -----------------------------
# Layer 1: statistics
# -----------------------------


def build_community_statistics(threads: Sequence[RedditThread]) -> Dict[str, Any]:
    if not threads:
        return {}
    comment_count = sum(len(thread.comments) for thread in threads)
    subreddits = {thread.subreddit for thread in threads}
    scores = [thread.score for thread in threads]
    timestamps = [thread.created_utc for thread in threads if thread.created_utc]
    if timestamps:
        start_year = datetime.fromtimestamp(min(timestamps), tz=timezone.utc).year
        end_year = datetime.fromtimestamp(max(timestamps), tz=timezone.utc).year
        time_span = str(start_year) if start_year == end_year else f"{start_year}-{end_year}"
    else:
        time_span = ""
    return {
        "threads": len(threads),
        "comments": comment_count,
        "subreddits": len(subreddits),
        "subreddit_names": sorted(subreddits),
        "average_upvotes": round(sum(scores) / len(scores)) if scores else 0,
        "time_span": time_span,
        "thread_links": _thread_links(range(len(threads)), threads),
    }


# -----------------------------
# Layer 2: questions
# -----------------------------


def build_questions(
    records: List[Dict[str, Any]],
    threads: Sequence[RedditThread],
) -> List[Dict[str, Any]]:
    question_records = [
        record
        for record in records
        if record["text"].endswith("?") and QUESTION_START.search(record["text"])
    ]
    clusters = _cluster_records(question_records[:800], QUESTION_CLUSTER_DISTANCE)
    rows = []
    for cluster in clusters[:MAX_ROWS_PER_LAYER]:
        rows.append(
            {
                "question": cluster["representative"],
                "mentions": len(cluster["sentences"]),
                "threads": len(cluster["threads"]),
                "thread_links": _thread_links(cluster["threads"], threads),
            }
        )
    return rows


# -----------------------------
# Layer 5: brands & products (reuses spaCy NER)
# -----------------------------


def build_brands(
    records: List[Dict[str, Any]],
    threads: Sequence[RedditThread],
) -> List[Dict[str, Any]]:
    nlp = get_nlp()
    if "ner" not in nlp.pipe_names:
        return []
    subset = records[:2500]
    texts = [record["text"] for record in subset]
    polarity_by_position = [record["polarity"] for record in subset]
    mentions: Counter = Counter()
    positive: Counter = Counter()
    negative: Counter = Counter()
    mention_threads: Dict[str, Set[int]] = defaultdict(set)
    for position, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        for ent in doc.ents:
            if ent.label_ not in ("ORG", "PRODUCT"):
                continue
            name = normalize_entity_text(ent.text)
            if not valid_entity_phrase(name):
                continue
            mentions[name] += 1
            mention_threads[name].add(subset[position]["thread"])
            if polarity_by_position[position] > 0:
                positive[name] += 1
            elif polarity_by_position[position] < 0:
                negative[name] += 1
    rows = []
    for name, count in mentions.most_common(MAX_ROWS_PER_LAYER):
        rows.append(
            {
                "brand": name,
                "mentions": count,
                "positive": positive.get(name, 0),
                "negative": negative.get(name, 0),
                "thread_links": _thread_links(mention_threads.get(name, set()), threads),
            }
        )
    return rows


# -----------------------------
# Layers 6 and 7: features and decision factors (noun mining)
# -----------------------------


def _noun_counts(
    records: List[Dict[str, Any]],
    sentence_filter,
) -> "tuple[Counter, Dict[str, Set[int]]]":
    """
    Returns (counts, thread_map): counts[lemma] is the mention count,
    thread_map[lemma] is the set of thread indices that mentioned it.
    """
    nlp = get_nlp()
    has_pos_model = any(
        pipe in nlp.pipe_names for pipe in ("tagger", "morphologizer")
    )
    if not has_pos_model:
        return Counter(), {}
    filtered = [record for record in records if sentence_filter(record)][:1500]
    texts = [record["text"] for record in filtered]
    counts: Counter = Counter()
    thread_map: Dict[str, Set[int]] = defaultdict(set)
    for record, doc in zip(filtered, nlp.pipe(texts, batch_size=64)):
        for token in doc:
            if token.pos_ != "NOUN" or token.is_stop or not token.is_alpha:
                continue
            lemma = token.lemma_.lower()
            if len(lemma) <= 2:
                continue
            counts[lemma] += 1
            thread_map[lemma].add(record["thread"])
    return counts, thread_map


def build_features(
    records: List[Dict[str, Any]],
    threads: Sequence[RedditThread],
) -> List[Dict[str, Any]]:
    counts, thread_map = _noun_counts(
        records,
        lambda record: record["polarity"] != 0
        or EVALUATIVE_PATTERN.search(record["text"]),
    )
    return [
        {
            "feature": name,
            "mentions": count,
            "thread_links": _thread_links(thread_map.get(name, set()), threads),
        }
        for name, count in counts.most_common(MAX_ROWS_PER_LAYER)
        if count >= 3
    ]


def build_decision_factors(
    records: List[Dict[str, Any]],
    threads: Sequence[RedditThread],
) -> List[Dict[str, Any]]:
    counts, thread_map = _noun_counts(
        records,
        lambda record: DECISION_PATTERN.search(record["text"]),
    )
    return [
        {
            "decision_factor": name,
            "mentions": count,
            "thread_links": _thread_links(thread_map.get(name, set()), threads),
        }
        for name, count in counts.most_common(MAX_ROWS_PER_LAYER)
        if count >= 2
    ]


# -----------------------------
# Layer 8: community vocabulary
# -----------------------------


def build_vocabulary(threads: Sequence[RedditThread]) -> List[Dict[str, Any]]:
    thread_texts = []
    for thread in threads:
        parts = [thread.title, thread.selftext]
        parts.extend(comment.body for comment in thread.comments)
        thread_texts.append(compact_text(" ".join(parts)).lower())
    corpus = " ".join(thread_texts)[:40_000]
    if not corpus:
        return []
    try:
        keywords = get_yake_extractor().extract_keywords(corpus)
    except Exception:
        return []
    rows = []
    seen = set()
    for keyword, _ in keywords:
        term = normalize_entity_text(keyword)
        if not valid_entity_phrase(term) or term in seen:
            continue
        seen.add(term)
        pattern = re.compile(r"\b" + re.escape(term) + r"\b")
        mention_count = len(pattern.findall(corpus))
        matching_indices = {
            i for i, text in enumerate(thread_texts) if pattern.search(text)
        }
        if mention_count < 3:
            continue
        rows.append(
            {
                "term": term,
                "mentions": mention_count,
                "threads": len(matching_indices),
                "thread_links": _thread_links(matching_indices, threads),
            }
        )
        if len(rows) >= MAX_ROWS_PER_LAYER:
            break
    rows.sort(key=lambda row: row["mentions"], reverse=True)
    return rows


# -----------------------------
# Layer 11: real experiences
# -----------------------------


def build_experiences(
    records: List[Dict[str, Any]],
    threads: Sequence[RedditThread],
) -> List[Dict[str, Any]]:
    matched = [
        record for record in records if EXPERIENCE_PATTERN.search(record["text"])
    ]
    clusters = _cluster_records(matched[:500], PATTERN_CLUSTER_DISTANCE)
    rows = []
    for cluster in clusters[:MAX_ROWS_PER_LAYER]:
        polarity_total = sum(cluster["polarities"])
        if polarity_total > 0:
            polarity = "positive"
        elif polarity_total < 0:
            polarity = "negative"
        else:
            polarity = "neutral"
        rows.append(
            {
                "experience": _cluster_label(cluster),
                "polarity": polarity,
                "mentions": len(cluster["sentences"]),
                "threads": len(cluster["threads"]),
                "example": cluster["representative"],
                "thread_links": _thread_links(cluster["threads"], threads),
            }
        )
    return rows


# -----------------------------
# Layer 12: community vs competitor gap
# -----------------------------


def _competitor_coverage(
    term: str,
    competitor_entities: Optional[List[Dict[str, Any]]],
    competitor_topics: Optional[List[Dict[str, Any]]],
) -> int:
    coverage = 0
    lowered = term.lower()
    for entity in competitor_entities or []:
        name = str(entity.get("entity", "")).lower()
        if name and fuzz.partial_ratio(lowered, name) >= GAP_MATCH_THRESHOLD:
            coverage += 1
    for topic in competitor_topics or []:
        name = str(topic.get("topic", "")).lower()
        if name and fuzz.partial_ratio(lowered, name) >= GAP_MATCH_THRESHOLD:
            coverage += 1
    return coverage


def build_gaps(
    pain_points: List[Dict[str, Any]],
    features: List[Dict[str, Any]],
    vocabulary: List[Dict[str, Any]],
    competitor_entities: Optional[List[Dict[str, Any]]],
    competitor_topics: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    # term -> {"reddit_mentions": int, "links": set of permalink strings}
    candidates: Dict[str, Dict[str, Any]] = {}

    def _add(term: str, amount: int, links_str: str) -> None:
        entry = candidates.setdefault(term, {"reddit_mentions": 0, "links": set()})
        entry["reddit_mentions"] += amount
        if links_str:
            entry["links"].update(links_str.split("; "))

    for row in pain_points:
        _add(row["pain_point"], row["frequency"], row.get("thread_links", ""))
    for row in features:
        _add(row["feature"], row["mentions"], row.get("thread_links", ""))
    for row in vocabulary:
        _add(row["term"], row["mentions"], row.get("thread_links", ""))

    rows = []
    for term, data in candidates.items():
        rows.append(
            {
                "topic": term,
                "reddit_mentions": data["reddit_mentions"],
                "competitor_coverage": _competitor_coverage(
                    term,
                    competitor_entities,
                    competitor_topics,
                ),
                "thread_links": "; ".join(sorted(data["links"])),
            }
        )
    rows.sort(
        key=lambda row: (row["competitor_coverage"], -row["reddit_mentions"])
    )
    return rows[:MAX_ROWS_PER_LAYER]


# -----------------------------
# Aggregator / entry point
# -----------------------------


def run_community_intelligence(
    threads: Sequence[RedditThread],
    competitor_entities: Optional[List[Dict[str, Any]]] = None,
    competitor_topics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Public entry point used by app.py.

    Returns:
    {
        "statistics": {...},
        "questions": [...],
        "pain_points": [...],
        "recommendations": [...],
        "brands": [...],
        "features": [...],
        "decision_factors": [...],
        "vocabulary": [...],
        "mistakes": [...],
        "myths": [...],
        "experiences": [...],
        "gaps": [...]
    }

    Every row in every list above includes a "thread_links" field (one or
    more Reddit permalinks joined with "; ") tracing it back to its
    source thread(s).
    """
    threads = list(threads or [])
    if not threads:
        return {}
    records = _collect_sentences(threads)
    pain_points = _pattern_layer(records, PAIN_PATTERN, "pain_point", "frequency", threads)
    features = build_features(records, threads)
    vocabulary = build_vocabulary(threads)
    return {
        "statistics": build_community_statistics(threads),
        "questions": build_questions(records, threads),
        "pain_points": pain_points,
        "recommendations": _pattern_layer(
            records, RECOMMENDATION_PATTERN, "advice", "mentions", threads
        ),
        "brands": build_brands(records, threads),
        "features": features,
        "decision_factors": build_decision_factors(records, threads),
        "vocabulary": vocabulary,
        "mistakes": _pattern_layer(
            records, MISTAKE_PATTERN, "mistake", "frequency", threads
        ),
        "myths": _pattern_layer(records, MYTH_PATTERN, "myth", "frequency", threads),
        "experiences": build_experiences(records, threads),
        "gaps": build_gaps(
            pain_points,
            features,
            vocabulary,
            competitor_entities,
            competitor_topics,
        ),
    }