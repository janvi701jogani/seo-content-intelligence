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
    "india": "in",
    "united states": "us",
    "united kingdom": "gb",
    "uk": "gb",
    "australia": "au",
    "canada": "ca",
}
LANGUAGE_CODES = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
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