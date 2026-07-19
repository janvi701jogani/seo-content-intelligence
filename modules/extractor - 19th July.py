"""
Generic SEO Intelligence Engine.

Single-file extractor for competitor content intelligence:
- competitor statistics and structure
- generic entity extraction with spaCy, GLiNER, KeyBERT, YAKE
- generic normalization and RapidFuzz merging
- Section Intelligence: section-based (not sentence- or heading-based)
  editorial section clustering across competitors, with oversized sections
  split into paragraph-group sub-sections before embedding so a single long
  section can yield more than one topic
- Structure Intelligence: heading/table/list/FAQ structural comparison
- coverage, gaps, co-occurrence, and aggregate statistics

No GPT calls. No niche-specific dictionaries.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import spacy
import yake
from bs4 import BeautifulSoup
from gliner import GLiNER
from keybert import KeyBERT
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

READING_SPEED_WPM = 220
MAX_SPACY_CHARS = 120_000
MAX_GLINER_CHARS = 30_000
MAX_KEYBERT_CHARS = 40_000
MAX_EMBEDDING_CHARS = 30_000
MAX_YAKE_CHARS = 40_000
ENTITY_MERGE_THRESHOLD = 92

# Section Intelligence engine.
MIN_SECTION_WORDS = 30
HEADING_QUALITY_THRESHOLD = 55.0
# Complete linkage (see make_agglomerative()) plus a conservative distance
# threshold. Average linkage is prone to chaining: one loosely-related
# section can bridge two otherwise-unrelated sections into the same
# cluster, producing one giant "everything about the topic" cluster plus a
# long tail of singletons -- which is what "20% coverage on nearly
# everything" looks like in the dashboard. Complete linkage requires every
# pair inside a cluster to be close, not just the average pair, so clusters
# stay tighter and closer to actual heading-level granularity.
SECTION_CLUSTER_DISTANCE_THRESHOLD = 0.22
SECTION_SUMMARY_SENTENCE_COUNT = 2
MIN_SECTION_SENTENCE_WORDS = 6
# Hard cap on every Section Intelligence label, regardless of source. A
# label is a section name ("Understanding Mutual Fund Fees"), never a
# sentence or disclaimer. Enforced by cap_label().
MAX_LABEL_WORDS = 8

# Sub-section splitting. Many real-world sections are 600-1200+ words
# covering several distinct sub-topics under one heading (e.g. a "Choosing
# a Mutual Fund" H2 that actually covers investment goals, risk tolerance,
# fees, and share classes one after another). Embedding the whole section
# as a single centroid vector collapses all of that into one point, so
# clustering can only ever produce one topic for it no matter how the
# distance threshold is tuned. Splitting oversized sections into smaller
# paragraph-group sub-sections before embedding gives clustering
# finer-grained input to work with, which is what actually increases topic
# count and comprehension -- clustering still decides the final topic
# count, this only gives it more (and better) material to cluster.
# Paragraphs are never split internally; a sub-section is always a
# contiguous run of whole paragraphs.
MAX_SECTION_WORDS_BEFORE_SPLIT = 250
SUBSECTION_TARGET_WORDS = 200
MIN_SUBSECTION_WORDS = 70

# Generic forum/meta-discourse patterns. These are structural (apply to any
# forum content regardless of niche), not domain vocabulary, so they don't
# violate the "no niche-specific dictionaries" design goal.
BOILERPLATE_HEADING_PATTERNS = [
    r"^op\b",
    r"^edit\b",
    r"^update\b",
    r"\bmoderator\b",
    r"\bautomoderator\b",
    r"you may find (these|this)",
    r"if i was in your shoes",
    r"if you a?re? reading this",
    r"^tl;?dr\b",
    r"thanks for reading",
    r"^welcome to\b",
    r"^comments?$",
    r"^share\b",
    r"^related posts?$",
]

GLINER_LABELS = [
    "person",
    "organization",
    "location",
    "product",
    "event",
    "concept",
    "technology",
    "law",
    "date",
    "work",
    "service",
]


@dataclass
class EntityCandidate:
    text: str
    normalized: str
    type: str
    source: str
    confidence: float
    competitor_index: int
    url: str
    position: Optional[int]
    mentions: int = 1
    first_char: Optional[int] = None
    in_heading: bool = False


@dataclass
class EntityAggregate:
    entity: str
    type: str
    mentions: int = 0
    competitors: Set[int] = field(default_factory=set)
    urls: Set[str] = field(default_factory=set)
    positions: Set[int] = field(default_factory=set)
    sources: Counter = field(default_factory=Counter)
    confidence_values: List[float] = field(default_factory=list)
    heading_hits: int = 0
    early_hits: int = 0
    co_occurrence: Counter = field(default_factory=Counter)


@dataclass
class Section:
    heading: str
    heading_level: str
    paragraphs: List[str]
    order_index: int
    competitor_index: int
    url: str
    tables: List[Dict[str, Any]] = field(default_factory=list)
    lists: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    internal_links: List[Dict[str, Any]] = field(default_factory=list)
    external_links: List[Dict[str, Any]] = field(default_factory=list)
    is_faq: bool = False

    @property
    def text(self) -> str:
        return " ".join(self.paragraphs)

    @property
    def word_count(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)

    @property
    def paragraph_count(self) -> int:
        return len(self.paragraphs)


@dataclass
class SectionProfile:
    section: Section
    embedding: np.ndarray
    representative_sentences: List[str]
    label: str
    heading_quality: float
    label_source: str  # "heading" or "summary"


@dataclass
class SectionTopicAggregate:
    """
    related_entities is populated by attach_related_entities() (Part 3
    holdover, reused here via duck typing on competitors/phrases/
    related_entities).
    """

    topic: str
    members: List[SectionProfile] = field(default_factory=list)
    competitors: Set[int] = field(default_factory=set)
    urls: Set[str] = field(default_factory=set)
    sources: Counter = field(default_factory=Counter)
    importance_values: List[float] = field(default_factory=list)
    heading_hits: int = 0
    phrases: Counter = field(default_factory=Counter)
    related_entities: Counter = field(default_factory=Counter)
    word_counts: List[int] = field(default_factory=list)
    table_counts: List[int] = field(default_factory=list)
    list_counts: List[int] = field(default_factory=list)
    faq_flags: List[bool] = field(default_factory=list)


@lru_cache(maxsize=1)
def get_nlp():
    """Load spaCy lazily and ensure sentence boundaries exist."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=1)
def get_keybert_model():
    return KeyBERT(get_sentence_model())


@lru_cache(maxsize=1)
def get_gliner_model():
    return GLiNER.from_pretrained("urchade/gliner_medium-v2.1")


@lru_cache(maxsize=1)
def get_yake_extractor():
    return yake.KeywordExtractor(lan="en", n=3, top=150)


@lru_cache(maxsize=1)
def get_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).replace("\xa0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def compact_text(text: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def clean_metadata(metadata: Any) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values = [compact_text(item) for item in value if compact_text(item)]
            if values:
                cleaned[key] = values
        elif isinstance(value, dict):
            nested = clean_metadata(value)
            if nested:
                cleaned[key] = nested
        else:
            cleaned_value = compact_text(value)
            if cleaned_value:
                cleaned[key] = cleaned_value
    return cleaned


def safe_number(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def candidate_html(competitor: Dict[str, Any]) -> str:
    for key in ("html", "raw_html", "content_html", "body_html", "scraped_html"):
        value = competitor.get(key)
        if isinstance(value, str) and "<" in value and ">" in value:
            return value
    return ""


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    return clean_text(soup.get_text("\n"))


def split_paragraphs(text: str) -> List[str]:
    blocks = [
        compact_text(block)
        for block in re.split(r"\n{2,}|\r\n{2,}", clean_text(text))
    ]
    if len(blocks) <= 1:
        blocks = [compact_text(line) for line in clean_text(text).splitlines()]
    return [block for block in blocks if block]


def extract_structure_from_html(html: str, base_url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()

    parsed_base = urlparse(base_url or "")
    base_host = parsed_base.netloc.lower()

    headings: Dict[str, List[str]] = {}
    for level in range(1, 7):
        tag_name = f"h{level}"
        headings[tag_name] = [
            compact_text(tag.get_text(" "))
            for tag in soup.find_all(tag_name)
        ]
        headings[tag_name] = [item for item in headings[tag_name] if item]

    lists = []
    for list_tag in soup.find_all(["ul", "ol"]):
        items = [
            compact_text(item.get_text(" "))
            for item in list_tag.find_all("li", recursive=False)
        ]
        items = [item for item in items if item]
        if items:
            lists.append(
                {
                    "type": list_tag.name,
                    "items": items,
                }
            )

    tables = []
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [
                compact_text(cell.get_text(" "))
                for cell in row.find_all(["th", "td"])
            ]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(
                {
                    "rows": rows,
                    "row_count": len(rows),
                }
            )

    images = []
    for image in soup.find_all("img"):
        alt_text = compact_text(image.get("alt", ""))
        src = compact_text(image.get("src", ""))
        if alt_text or src:
            images.append(
                {
                    "alt": alt_text,
                    "src": src,
                }
            )

    internal_links = []
    external_links = []
    for anchor in soup.find_all("a", href=True):
        href = compact_text(anchor.get("href", ""))
        label = compact_text(anchor.get_text(" "))
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        link_host = urlparse(href).netloc.lower()
        link = {
            "text": label,
            "href": href,
        }
        if not link_host or (base_host and link_host == base_host):
            internal_links.append(link)
        else:
            external_links.append(link)

    paragraphs = [
        compact_text(p.get_text(" "))
        for p in soup.find_all("p")
    ]
    paragraphs = [p for p in paragraphs if p]

    title = compact_text(soup.title.get_text(" ")) if soup.title else ""

    # Document-order block sequence, used by the section engine to know
    # which paragraphs, tables, lists, images, and links belong under which
    # heading. find_all with multiple tag names returns matches in document
    # order, so headings/paragraphs/tables/lists/images/links interleave
    # exactly as they appear in the page.
    blocks: List[Dict[str, Any]] = []
    block_tag_names = [f"h{level}" for level in range(1, 7)] + [
        "p",
        "table",
        "ul",
        "ol",
        "img",
        "a",
    ]
    for tag in soup.find_all(block_tag_names):
        if tag.name.startswith("h"):
            block_text = compact_text(tag.get_text(" "))
            if block_text:
                blocks.append({"type": "heading", "level": tag.name, "text": block_text})
        elif tag.name == "p":
            block_text = compact_text(tag.get_text(" "))
            if block_text:
                blocks.append({"type": "paragraph", "level": "", "text": block_text})
        elif tag.name == "table":
            table_rows = []
            for row in tag.find_all("tr"):
                cells = [
                    compact_text(cell.get_text(" "))
                    for cell in row.find_all(["th", "td"])
                ]
                cells = [cell for cell in cells if cell]
                if cells:
                    table_rows.append(cells)
            if table_rows:
                blocks.append(
                    {"type": "table", "level": "", "row_count": len(table_rows)}
                )
        elif tag.name in ("ul", "ol"):
            list_items = [
                compact_text(item.get_text(" "))
                for item in tag.find_all("li", recursive=False)
            ]
            list_items = [item for item in list_items if item]
            if list_items:
                blocks.append(
                    {
                        "type": "list",
                        "level": "",
                        "list_type": tag.name,
                        "item_count": len(list_items),
                    }
                )
        elif tag.name == "img":
            alt_text = compact_text(tag.get("alt", ""))
            src = compact_text(tag.get("src", ""))
            if alt_text or src:
                blocks.append(
                    {"type": "image", "level": "", "alt": alt_text, "src": src}
                )
        elif tag.name == "a":
            href = compact_text(tag.get("href", ""))
            label = compact_text(tag.get_text(" "))
            if href and not href.startswith(("#", "mailto:", "tel:", "javascript:")):
                link_host = urlparse(href).netloc.lower()
                is_internal = not link_host or (base_host and link_host == base_host)
                blocks.append(
                    {
                        "type": "link",
                        "level": "",
                        "text": label,
                        "href": href,
                        "internal": is_internal,
                    }
                )

    return {
        "source": "html",
        "title": title,
        "headings": headings,
        "logical_headings": [],
        "paragraphs": paragraphs,
        "lists": lists,
        "tables": tables,
        "images": images,
        "internal_links": internal_links,
        "external_links": external_links,
        "blocks": blocks,
    }


def is_explicit_text_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^#{1,6}\s+\S+", stripped):
        return True
    if re.match(r"^(\d+(\.\d+)*|[A-Z])[\.)]\s+\S+", stripped):
        return True
    if stripped.endswith(":") and len(stripped.split()) <= 12:
        return True
    return False


def extract_structure_from_text(text: str, fallback_title: str = "") -> Dict[str, Any]:
    lines = [compact_text(line) for line in clean_text(text).splitlines()]
    lines = [line for line in lines if line]
    paragraphs = split_paragraphs(text)

    logical_headings = []
    list_items = []
    blocks: List[Dict[str, str]] = []
    for line in lines:
        if is_explicit_text_heading(line):
            heading = re.sub(r"^#{1,6}\s+", "", line).rstrip(":").strip()
            logical_headings.append(heading)
            blocks.append({"type": "heading", "level": "text", "text": heading})
        elif re.match(r"^[-*+]\s+\S+", line) or re.match(r"^\d+[\.)]\s+\S+", line):
            item = re.sub(r"^([-*+]|\d+[\.)])\s+", "", line).strip()
            list_items.append(item)
            blocks.append({"type": "paragraph", "level": "", "text": item})
        else:
            blocks.append({"type": "paragraph", "level": "", "text": line})

    return {
        "source": "text",
        "title": compact_text(fallback_title) or (lines[0] if lines else ""),
        "headings": {f"h{level}": [] for level in range(1, 7)},
        "logical_headings": logical_headings,
        "paragraphs": paragraphs,
        "lists": [{"type": "text", "items": list_items}] if list_items else [],
        "tables": [],
        "images": [],
        "internal_links": [],
        "external_links": [],
        "blocks": blocks,
    }


def extract_structure(competitor: Dict[str, Any], text: str) -> Dict[str, Any]:
    html = candidate_html(competitor)
    if html:
        return extract_structure_from_html(html, competitor.get("url", ""))
    return extract_structure_from_text(text, competitor.get("title", ""))


def document_statistics(text: str, doc: Any, paragraph_count: int) -> Dict[str, Any]:
    words = [
        token
        for token in doc
        if not token.is_space and not token.is_punct
    ]
    sentences = [sent for sent in doc.sents] if doc.has_annotation("SENT_START") else []
    if not sentences:
        sentences = [
            sent
            for sent in re.split(r"(?<=[.!?])\s+", text)
            if compact_text(sent)
        ]

    word_count = len(words) if words else len(re.findall(r"\b\w+\b", text))
    sentence_count = len(sentences)
    average_sentence_length = (
        round(word_count / sentence_count, 2)
        if sentence_count
        else 0
    )
    reading_time = round(word_count / READING_SPEED_WPM, 2)

    return {
        "words": word_count,
        "word_count": word_count,
        "characters": len(text),
        "character_count": len(text),
        "sentences": sentence_count,
        "sentence_count": sentence_count,
        "paragraphs": paragraph_count,
        "paragraph_count": paragraph_count,
        "average_sentence_length": average_sentence_length,
        "reading_time": reading_time,
        "estimated_reading_time": max(1, math.ceil(word_count / READING_SPEED_WPM))
        if word_count
        else 0,
    }


def normalize_entity_text(text: str) -> str:
    value = compact_text(text).lower()
    value = re.sub(r"[^\w\s-]", " ", value)
    value = re.sub(r"[_-]+", " ", value)
    value = compact_text(value)
    if not value:
        return ""

    doc = get_nlp()(value)
    lemmas = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.strip().lower() if token.lemma_ else token.text.lower()
        if lemma and lemma != "-pron-":
            lemmas.append(lemma)
    return compact_text(" ".join(lemmas))


def valid_entity_phrase(normalized: str) -> bool:
    if not normalized or len(normalized) < 2:
        return False
    if normalized.isdigit():
        return False
    if not re.search(r"[a-zA-Z]", normalized):
        return False
    words = normalized.split()
    if len(words) > 7:
        return False
    if all(len(word) <= 1 for word in words):
        return False

    doc = get_nlp()(normalized)
    lexical_tokens = [
        token
        for token in doc
        if not token.is_space and not token.is_punct and not token.like_num
    ]
    if not lexical_tokens:
        return False
    if all(token.is_stop for token in lexical_tokens):
        return False

    has_pos_model = any(
        pipe in get_nlp().pipe_names
        for pipe in ("tagger", "morphologizer")
    )
    if not has_pos_model:
        return True

    return any(
        token.pos_ in {"NOUN", "PROPN", "ADJ", "NUM", "X"}
        for token in lexical_tokens
    )


def find_mentions(normalized: str, normalized_text: str) -> Tuple[int, Optional[int]]:
    if not normalized:
        return 0, None
    pattern = r"\b" + re.escape(normalized) + r"\b"
    matches = list(re.finditer(pattern, normalized_text))
    if matches:
        return len(matches), matches[0].start()
    if fuzz.partial_ratio(
        normalized,
        normalized_text[: max(2000, len(normalized) * 20)],
    ) >= 94:
        return 1, 0
    return 1, None


def heading_blob(structure: Dict[str, Any]) -> str:
    headings = []
    for values in structure.get("headings", {}).values():
        headings.extend(values)
    headings.extend(structure.get("logical_headings", []))
    return " ".join(headings)


def make_candidate(
    raw_text: str,
    entity_type: str,
    source: str,
    confidence: float,
    competitor_index: int,
    competitor: Dict[str, Any],
    normalized_text: str,
    headings_text: str,
) -> Optional[EntityCandidate]:
    normalized = normalize_entity_text(raw_text)
    if not valid_entity_phrase(normalized):
        return None

    mentions, first_char = find_mentions(normalized, normalized_text)
    normalized_headings = normalize_entity_text(headings_text)
    in_heading = bool(
        normalized_headings
        and fuzz.partial_ratio(normalized, normalized_headings) >= 92
    )

    return EntityCandidate(
        text=normalized,
        normalized=normalized,
        type=entity_type.upper(),
        source=source,
        confidence=float(max(0.0, min(1.0, confidence))),
        competitor_index=competitor_index,
        url=competitor.get("url", ""),
        position=safe_number(competitor.get("position")),
        mentions=max(1, mentions),
        first_char=first_char,
        in_heading=in_heading,
    )


def extract_spacy_candidates(
    doc: Any,
    competitor_index: int,
    competitor: Dict[str, Any],
    normalized_text: str,
    headings_text: str,
) -> List[EntityCandidate]:
    candidates: List[EntityCandidate] = []
    for ent in getattr(doc, "ents", []):
        candidate = make_candidate(
            ent.text,
            ent.label_,
            "spacy",
            0.86,
            competitor_index,
            competitor,
            normalized_text,
            headings_text,
        )
        if candidate:
            candidates.append(candidate)

    if hasattr(doc, "noun_chunks"):
        try:
            for chunk in doc.noun_chunks:
                candidate = make_candidate(
                    chunk.text,
                    "KEYPHRASE",
                    "spacy_noun_chunk",
                    0.62,
                    competitor_index,
                    competitor,
                    normalized_text,
                    headings_text,
                )
                if candidate:
                    candidates.append(candidate)
        except ValueError:
            pass

    return candidates


def extract_gliner_candidates(
    text: str,
    competitor_index: int,
    competitor: Dict[str, Any],
    normalized_text: str,
    headings_text: str,
) -> List[EntityCandidate]:
    try:
        results = get_gliner_model().predict_entities(
            text[:MAX_GLINER_CHARS],
            GLINER_LABELS,
        )
    except Exception:
        return []

    candidates = []
    for result in results:
        candidate = make_candidate(
            result.get("text", ""),
            result.get("label", "ENTITY"),
            "gliner",
            float(result.get("score", 0.75)),
            competitor_index,
            competitor,
            normalized_text,
            headings_text,
        )
        if candidate:
            candidates.append(candidate)
    return candidates


def extract_keybert_candidates(
    text: str,
    competitor_index: int,
    competitor: Dict[str, Any],
    normalized_text: str,
    headings_text: str,
) -> List[EntityCandidate]:
    if len(text.split()) < 8:
        return []
    try:
        keywords = get_keybert_model().extract_keywords(
            text[:MAX_KEYBERT_CHARS],
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=120,
            use_mmr=True,
            diversity=0.55,
        )
    except Exception:
        return []

    candidates = []
    for keyword, score in keywords:
        candidate = make_candidate(
            keyword,
            "KEYPHRASE",
            "keybert",
            float(score),
            competitor_index,
            competitor,
            normalized_text,
            headings_text,
        )
        if candidate:
            candidates.append(candidate)
    return candidates


def extract_yake_candidates(
    text: str,
    competitor_index: int,
    competitor: Dict[str, Any],
    normalized_text: str,
    headings_text: str,
) -> List[EntityCandidate]:
    if len(text.split()) < 8:
        return []
    try:
        keywords = get_yake_extractor().extract_keywords(
            text[:MAX_YAKE_CHARS]
        )
    except Exception:
        return []

    candidates = []
    for keyword, score in keywords:
        confidence = 1.0 / (1.0 + float(score))
        candidate = make_candidate(
            keyword,
            "KEYPHRASE",
            "yake",
            confidence,
            competitor_index,
            competitor,
            normalized_text,
            headings_text,
        )
        if candidate:
            candidates.append(candidate)
    return candidates


def merge_candidate_group(candidates: List[EntityCandidate]) -> List[EntityCandidate]:
    merged: Dict[str, EntityCandidate] = {}
    source_sets: Dict[str, Set[str]] = defaultdict(set)
    confidence_values: Dict[str, List[float]] = defaultdict(list)

    for candidate in candidates:
        match = None
        for existing in merged:
            if fuzz.token_sort_ratio(existing, candidate.normalized) >= ENTITY_MERGE_THRESHOLD:
                match = existing
                break

        key = match or candidate.normalized
        if key not in merged:
            merged[key] = candidate
        else:
            current = merged[key]
            current.mentions += candidate.mentions
            current.in_heading = current.in_heading or candidate.in_heading
            if current.first_char is None or (
                candidate.first_char is not None
                and candidate.first_char < current.first_char
            ):
                current.first_char = candidate.first_char
            if candidate.type != "KEYPHRASE" and current.type == "KEYPHRASE":
                current.type = candidate.type

        source_sets[key].add(candidate.source)
        confidence_values[key].append(candidate.confidence)

    output = []
    for key, candidate in merged.items():
        candidate.source = "|".join(sorted(source_sets[key]))
        candidate.confidence = (
            float(np.mean(confidence_values[key]))
            if confidence_values[key]
            else candidate.confidence
        )
        output.append(candidate)
    return output


def extract_entities_for_document(
    doc: Any,
    text: str,
    structure: Dict[str, Any],
    competitor: Dict[str, Any],
    competitor_index: int,
) -> List[EntityCandidate]:
    headings_text = heading_blob(structure)
    normalized_text = normalize_entity_text(text)

    candidates: List[EntityCandidate] = []
    candidates.extend(
        extract_spacy_candidates(
            doc,
            competitor_index,
            competitor,
            normalized_text,
            headings_text,
        )
    )
    candidates.extend(
        extract_gliner_candidates(
            text,
            competitor_index,
            competitor,
            normalized_text,
            headings_text,
        )
    )
    candidates.extend(
        extract_keybert_candidates(
            text,
            competitor_index,
            competitor,
            normalized_text,
            headings_text,
        )
    )
    candidates.extend(
        extract_yake_candidates(
            text,
            competitor_index,
            competitor,
            normalized_text,
            headings_text,
        )
    )
    return merge_candidate_group(candidates)


def merge_entities_across_competitors(
    candidates: Iterable[EntityCandidate],
) -> Dict[str, EntityAggregate]:
    aggregates: Dict[str, EntityAggregate] = {}
    canonical_keys: List[str] = []

    for candidate in candidates:
        match = None
        for key in canonical_keys:
            if fuzz.token_sort_ratio(key, candidate.normalized) >= ENTITY_MERGE_THRESHOLD:
                match = key
                break

        key = match or candidate.normalized
        if key not in aggregates:
            aggregates[key] = EntityAggregate(
                entity=key,
                type=candidate.type,
            )
            canonical_keys.append(key)

        aggregate = aggregates[key]
        aggregate.mentions += candidate.mentions
        aggregate.competitors.add(candidate.competitor_index)
        if candidate.url:
            aggregate.urls.add(candidate.url)
        if candidate.position is not None:
            aggregate.positions.add(candidate.position)
        for source in candidate.source.split("|"):
            aggregate.sources[source] += 1
        aggregate.confidence_values.append(candidate.confidence)
        aggregate.heading_hits += int(candidate.in_heading)
        aggregate.early_hits += int(
            candidate.first_char is not None
            and candidate.first_char <= 750
        )
        if aggregate.type == "KEYPHRASE" and candidate.type != "KEYPHRASE":
            aggregate.type = candidate.type

    return aggregates


def add_entity_co_occurrence(
    aggregates: Dict[str, EntityAggregate],
    document_entities: List[List[EntityCandidate]],
) -> None:
    canonical_lookup = list(aggregates.keys())

    def canonical(name: str) -> Optional[str]:
        for key in canonical_lookup:
            if fuzz.token_sort_ratio(key, name) >= ENTITY_MERGE_THRESHOLD:
                return key
        return None

    for entities in document_entities:
        names = sorted({canonical(entity.normalized) for entity in entities})
        names = [name for name in names if name]
        for left, right in combinations(names, 2):
            aggregates[left].co_occurrence[right] += 1
            aggregates[right].co_occurrence[left] += 1


def normalized_score(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    maximum = max(values.values()) or 1.0
    return {
        key: value / maximum
        for key, value in values.items()
    }


def build_entity_dashboard(
    aggregates: Dict[str, EntityAggregate],
    total_competitors: int,
) -> List[Dict[str, Any]]:
    if not aggregates:
        return []

    coverage_values = {
        key: len(value.competitors) / total_competitors if total_competitors else 0
        for key, value in aggregates.items()
    }
    mention_scores = normalized_score(
        {
            key: math.log1p(value.mentions)
            for key, value in aggregates.items()
        }
    )
    co_scores = normalized_score(
        {
            key: sum(value.co_occurrence.values())
            for key, value in aggregates.items()
        }
    )

    rows = []
    for key, aggregate in aggregates.items():
        competitors_using = len(aggregate.competitors)
        coverage_ratio = coverage_values[key]
        extractor_agreement = min(1.0, len(aggregate.sources) / 4)
        heading_score = aggregate.heading_hits / max(1, competitors_using)
        early_score = aggregate.early_hits / max(1, competitors_using)
        confidence = (
            float(np.mean(aggregate.confidence_values))
            if aggregate.confidence_values
            else 0
        )

        importance = (
            coverage_ratio * 0.28
            + mention_scores.get(key, 0) * 0.20
            + (competitors_using / max(1, total_competitors)) * 0.16
            + extractor_agreement * 0.14
            + heading_score * 0.10
            + early_score * 0.07
            + co_scores.get(key, 0) * 0.05
        ) * 100

        rows.append(
            {
                "entity": aggregate.entity,
                "type": aggregate.type,
                "mentions": aggregate.mentions,
                "competitors_using": competitors_using,
                "competitors": competitors_using,
                "coverage": round(coverage_ratio * 100, 2),
                "coverage_percent": round(coverage_ratio * 100, 2),
                "average_mentions": round(
                    aggregate.mentions / max(1, competitors_using),
                    2,
                ),
                "avg_mentions": round(
                    aggregate.mentions / max(1, competitors_using),
                    2,
                ),
                "importance": round(importance, 2),
                "confidence": round(confidence, 4),
                "extractor_sources": sorted(aggregate.sources.keys()),
                "sources": sorted(aggregate.sources.keys()),
                "related_entities": [
                    item
                    for item, _ in aggregate.co_occurrence.most_common(20)
                ],
                "urls": sorted(aggregate.urls),
                "positions": sorted(aggregate.positions),
                "competitor_indexes": sorted(aggregate.competitors),
            }
        )

    rows.sort(
        key=lambda item: (
            item["importance"],
            item["coverage"],
            item["mentions"],
        ),
        reverse=True,
    )
    return rows


def attach_related_entities(
    aggregate: "SectionTopicAggregate",
    entities: List[Dict[str, Any]],
) -> None:
    """
    Reuses the existing entity dashboard instead of re-deriving entities
    from section phrases. An entity is attached to a section aggregate when
    its competitor_indexes overlap significantly with the aggregate's
    competitors and its normalized name occurs in the aggregate's phrases.
    """
    if not entities or not aggregate.competitors:
        return

    phrase_blob = " ".join(aggregate.phrases.keys())
    if not phrase_blob:
        return

    for entity in entities:
        entity_competitors = set(entity.get("competitor_indexes", []))
        if not entity_competitors:
            continue

        overlap = entity_competitors & aggregate.competitors
        if not overlap:
            continue
        if len(overlap) / len(entity_competitors) < 0.5:
            continue

        if fuzz.partial_ratio(entity["entity"], phrase_blob) >= 88:
            aggregate.related_entities[entity["entity"]] += 1


# ---------------------------------------------------------------------------
# Section Intelligence engine.
#
# Article -> sections (heading + its paragraphs) -> oversized sections split
# into paragraph-group sub-sections (see split_section_into_subsections()) ->
# extractive summary per (sub-)section -> embed summaries -> cluster
# summaries globally (complete linkage, see make_agglomerative()) -> section
# labels (ranked candidate pipeline, see generate_label_from_content() and
# choose_group_label()).
#
# Headings are candidates, not sections: a heading is only used verbatim as a
# label if it clears a quality bar and isn't forum/meta boilerplate.
# Otherwise the label is generated from the cluster's own content (KeyBERT >
# YAKE > noun chunk > capped sentence as absolute last resort -- never a raw
# sentence). Every label is capped to MAX_LABEL_WORDS via cap_label(). No
# GPT calls.
# ---------------------------------------------------------------------------


def is_boilerplate_heading(heading: str) -> bool:
    lowered = compact_text(heading).lower()
    if not lowered:
        return True
    return any(re.search(pattern, lowered) for pattern in BOILERPLATE_HEADING_PATTERNS)


def score_heading_quality(heading: str) -> float:
    """
    Generic, structural heading-quality score (0-100). No SEO/finance
    vocabulary: only length, POS shape, and stopword ratio. Boilerplate
    headings (forum meta-text) score 0.
    """
    cleaned = compact_text(heading)
    if not cleaned:
        return 0.0
    if is_boilerplate_heading(cleaned):
        return 0.0

    words = cleaned.split()
    word_count = len(words)
    if word_count == 0:
        return 0.0

    doc = get_nlp()(cleaned)
    lexical_tokens = [
        token
        for token in doc
        if not token.is_space and not token.is_punct
    ]
    if not lexical_tokens:
        return 0.0

    stopword_ratio = sum(1 for token in lexical_tokens if token.is_stop) / len(
        lexical_tokens
    )

    has_pos_model = any(
        pipe in get_nlp().pipe_names
        for pipe in ("tagger", "morphologizer")
    )
    has_verb = False
    has_noun = False
    if has_pos_model:
        has_verb = any(token.pos_ in {"VERB", "AUX"} for token in doc)
        has_noun = any(token.pos_ in {"NOUN", "PROPN"} for token in doc)
    else:
        has_noun = True

    length_score = min(1.0, word_count / 6)
    structure_score = 0.6 if (has_verb or word_count >= 3) and has_noun else 0.25
    stopword_penalty = max(0.0, 1.0 - stopword_ratio)

    score = (
        length_score * 0.35
        + structure_score * 0.45
        + stopword_penalty * 0.20
    ) * 100
    return round(score, 2)


def clean_heading_label(heading: str) -> str:
    cleaned = compact_text(heading)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned)
    cleaned = re.sub(r"^\d+(\.\d+)*[\.\)]?\s+", "", cleaned)
    return cleaned.strip()


def split_into_sections(
    structure: Dict[str, Any],
    competitor: Dict[str, Any],
    competitor_index: int,
) -> List[Section]:
    blocks = structure.get("blocks", [])
    sections: List[Section] = []

    current_heading = ""
    current_level = ""
    current_paragraphs: List[str] = []
    current_tables: List[Dict[str, Any]] = []
    current_lists: List[Dict[str, Any]] = []
    current_images: List[Dict[str, Any]] = []
    current_internal_links: List[Dict[str, Any]] = []
    current_external_links: List[Dict[str, Any]] = []
    order_index = 0

    def flush() -> None:
        nonlocal order_index
        word_total = sum(len(p.split()) for p in current_paragraphs)
        if current_paragraphs and word_total >= MIN_SECTION_WORDS:
            sections.append(
                Section(
                    heading=current_heading,
                    heading_level=current_level,
                    paragraphs=list(current_paragraphs),
                    order_index=order_index,
                    competitor_index=competitor_index,
                    url=competitor.get("url", ""),
                    tables=list(current_tables),
                    lists=list(current_lists),
                    images=list(current_images),
                    internal_links=list(current_internal_links),
                    external_links=list(current_external_links),
                    is_faq=is_faq_heading(current_heading),
                )
            )
            order_index += 1

    for block in blocks:
        block_type = block.get("type")
        if block_type == "heading":
            flush()
            current_heading = block.get("text", "")
            current_level = block.get("level", "")
            current_paragraphs = []
            current_tables = []
            current_lists = []
            current_images = []
            current_internal_links = []
            current_external_links = []
        elif block_type == "paragraph":
            text_value = block.get("text", "")
            if text_value:
                current_paragraphs.append(text_value)
        elif block_type == "table":
            current_tables.append(block)
        elif block_type == "list":
            current_lists.append(block)
        elif block_type == "image":
            current_images.append(block)
        elif block_type == "link":
            if block.get("internal"):
                current_internal_links.append(block)
            else:
                current_external_links.append(block)

    flush()
    return sections


def split_section_into_subsections(section: Section) -> List[Section]:
    """
    Splits an oversized section into smaller paragraph-group sub-sections so
    each sub-topic can be embedded and clustered independently, instead of
    being averaged away inside one whole-section centroid. Paragraphs are
    never split internally -- a sub-section is always a contiguous run of
    whole paragraphs, greedily grouped up to roughly SUBSECTION_TARGET_WORDS.

    Only the first sub-section keeps the original heading, so heading-based
    labeling still applies to it. Later sub-sections get an empty heading:
    score_heading_quality("") returns 0, so build_section_profiles()
    correctly falls back to content-derived labeling for them -- no heading
    text is invented for a sub-section that never had one of its own.

    Tables/lists/images/links are attached only to the first sub-section.
    split_into_sections() does not record which paragraph a table or list
    originally sat next to, so there's no reliable way to attribute them to
    one specific sub-section, and duplicating them across every sub-section
    would inflate average_table_count/average_list_count downstream.
    """
    if (
        section.word_count <= MAX_SECTION_WORDS_BEFORE_SPLIT
        or len(section.paragraphs) <= 1
    ):
        return [section]

    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_words = 0
    for paragraph in section.paragraphs:
        paragraph_words = len(paragraph.split())
        if current_chunk and current_words + paragraph_words > SUBSECTION_TARGET_WORDS:
            chunks.append(current_chunk)
            current_chunk = []
            current_words = 0
        current_chunk.append(paragraph)
        current_words += paragraph_words
    if current_chunk:
        chunks.append(current_chunk)

    # A too-small trailing chunk becomes an under-powered orphan rather than
    # a real sub-topic -- merge it back into the previous chunk instead.
    if len(chunks) > 1 and sum(len(p.split()) for p in chunks[-1]) < MIN_SUBSECTION_WORDS:
        chunks[-2].extend(chunks.pop())

    if len(chunks) <= 1:
        return [section]

    subsections: List[Section] = []
    for chunk_index, chunk_paragraphs in enumerate(chunks):
        is_first = chunk_index == 0
        subsections.append(
            Section(
                heading=section.heading if is_first else "",
                heading_level=section.heading_level,
                paragraphs=chunk_paragraphs,
                order_index=section.order_index * 1000 + chunk_index,
                competitor_index=section.competitor_index,
                url=section.url,
                tables=list(section.tables) if is_first else [],
                lists=list(section.lists) if is_first else [],
                images=list(section.images) if is_first else [],
                internal_links=list(section.internal_links) if is_first else [],
                external_links=list(section.external_links) if is_first else [],
                is_faq=section.is_faq,
            )
        )
    return subsections


def expand_large_sections(sections: List[Section]) -> List[Section]:
    """Applies split_section_into_subsections() across a document's sections."""
    expanded: List[Section] = []
    for section in sections:
        expanded.extend(split_section_into_subsections(section))
    return expanded


def split_section_sentences(
    paragraphs: List[str],
    min_words: int = MIN_SECTION_SENTENCE_WORDS,
) -> List[str]:
    sentences = []
    for paragraph in paragraphs:
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            sentence = compact_text(sentence)
            if len(sentence.split()) >= min_words:
                sentences.append(sentence)
    return sentences


def cap_label(label: str, max_words: int = MAX_LABEL_WORDS) -> str:
    """
    Enforces the "a label is a section name, never a sentence" rule. Applied
    to every label regardless of source (heading or generated), so a
    disclaimer or a full sentence can never surface as a topic name.
    """
    cleaned = compact_text(label).rstrip(".,;:").strip()
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words])


def generate_label_from_content(sentences: Sequence[str]) -> str:
    """
    Ranked candidate pipeline for a content-derived label. No raw sentence
    is ever used as a topic name except as an absolute last resort, and even
    then it is capped to MAX_LABEL_WORDS. Priority order:

    1. KeyBERT 3-gram phrase (best at surfacing a short, contextual noun
       phrase from a handful of sentences -- e.g. "mutual fund fees" out of
       several sentences about expense ratios, loads, and annual charges).
    2. YAKE keyword.
    3. Noun chunk (spaCy).
    4. Capped leading sentence (last resort; still <= MAX_LABEL_WORDS).

    The label is generated from ALL sentences passed in together (the
    combined content), not the single most-central sentence -- a centroid
    sentence is mathematically representative but editorially arbitrary
    (it can just as easily be a disclaimer as a topic sentence).
    """
    joined = compact_text(" ".join(sentences[:12]))
    if not joined:
        return ""

    try:
        keybert_candidates = get_keybert_model().extract_keywords(
            joined,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=5,
            use_mmr=True,
            diversity=0.5,
        )
    except Exception:
        keybert_candidates = []
    for phrase, _ in keybert_candidates:
        normalized = normalize_entity_text(phrase)
        if valid_entity_phrase(normalized):
            return cap_label(normalized)

    try:
        yake_candidates = get_yake_extractor().extract_keywords(joined)
    except Exception:
        yake_candidates = []
    for keyword, _ in yake_candidates:
        normalized = normalize_entity_text(keyword)
        if valid_entity_phrase(normalized):
            return cap_label(normalized)

    doc = get_nlp()(joined)
    if hasattr(doc, "noun_chunks"):
        try:
            chunks = [
                normalize_entity_text(chunk.text)
                for chunk in doc.noun_chunks
            ]
            chunks = [
                chunk
                for chunk in chunks
                if valid_entity_phrase(chunk)
            ]
            if chunks:
                return cap_label(Counter(chunks).most_common(1)[0][0])
        except ValueError:
            pass

    return cap_label(sentences[0]) if sentences else ""


def make_agglomerative(distance_threshold: float):
    """
    Complete linkage, not average: average linkage chains loosely-related
    sections together transitively (A-B close, B-C close => A-C merged even
    if A-C are unrelated), which is what produced one giant cluster plus a
    long tail of singleton clusters. Complete linkage requires every pair in
    a cluster to be within distance_threshold, so a cluster can't grow by a
    single weak bridge.
    """
    try:
        return AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="complete",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="complete",
            distance_threshold=distance_threshold,
        )


def build_section_profiles(sections: List[Section]) -> List[SectionProfile]:
    """
    One batched encode() call per document (per call to this function),
    covering every sentence in every section of that document. No
    per-section or per-sentence encode calls.
    """
    if not sections:
        return []

    section_sentences: List[List[str]] = []
    for section in sections:
        sentences = split_section_sentences(section.paragraphs)
        if not sentences:
            fallback = compact_text(" ".join(section.paragraphs))[:280]
            sentences = [fallback] if fallback else []
        section_sentences.append(sentences)

    flat_sentences: List[str] = []
    owners: List[int] = []
    for section_index, sentences in enumerate(section_sentences):
        for sentence in sentences:
            flat_sentences.append(sentence)
            owners.append(section_index)

    if not flat_sentences:
        return []

    try:
        embeddings = get_sentence_model().encode(
            flat_sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except TypeError:
        embeddings = get_sentence_model().encode(
            flat_sentences,
            normalize_embeddings=True,
        )

    per_section_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
    per_section_sentences: Dict[int, List[str]] = defaultdict(list)
    for owner, embedding, sentence in zip(owners, embeddings, flat_sentences):
        per_section_embeddings[owner].append(embedding)
        per_section_sentences[owner].append(sentence)

    profiles: List[SectionProfile] = []
    for section_index, section in enumerate(sections):
        section_embeddings = per_section_embeddings.get(section_index, [])
        if not section_embeddings:
            continue

        matrix = np.array(section_embeddings)
        centroid = np.mean(matrix, axis=0)
        sentence_texts = per_section_sentences[section_index]

        if len(sentence_texts) > 1:
            similarities = cosine_similarity([centroid], matrix)[0]
            ranked = [sentence_texts[i] for i in np.argsort(similarities)[::-1]]
        else:
            ranked = sentence_texts
        representative_sentences = ranked[:SECTION_SUMMARY_SENTENCE_COUNT]

        heading_quality = score_heading_quality(section.heading)
        if section.heading and heading_quality >= HEADING_QUALITY_THRESHOLD:
            label = cap_label(clean_heading_label(section.heading))
            label_source = "heading"
        else:
            label = generate_label_from_content(representative_sentences)
            label_source = "summary"

        if not label:
            continue

        profiles.append(
            SectionProfile(
                section=section,
                embedding=centroid,
                representative_sentences=representative_sentences,
                label=label,
                heading_quality=heading_quality,
                label_source=label_source,
            )
        )

    return profiles


def cluster_section_profiles(
    profiles: List[SectionProfile],
) -> List[List[SectionProfile]]:
    """
    Single global clustering pass over every section's embedding across all
    competitors: embed summaries -> cluster summaries, matching the diagram
    directly rather than clustering per-document and merging afterward.
    """
    if not profiles:
        return []

    embeddings = np.array([profile.embedding for profile in profiles])

    if len(profiles) == 1:
        labels = np.array([0])
    else:
        clustering = make_agglomerative(SECTION_CLUSTER_DISTANCE_THRESHOLD)
        labels = clustering.fit_predict(embeddings)

    groups: Dict[int, List[SectionProfile]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(profiles[index])

    return list(groups.values())


def choose_group_label(members: List[SectionProfile]) -> str:
    """
    Ranked candidate pipeline for the cluster's final label:

    1. Common heading across competitors: the mode of heading-sourced
       member labels (already cleaned + capped in build_section_profiles()).
    2. Otherwise, generate a fresh label from every member's representative
       sentences combined, not just one member's -- the label should
       reflect the whole cluster's content, not one section's most-central
       sentence (which is representative but not necessarily meaningful:
       e.g. a disclaimer can be the mathematical centroid of 100 sentences
       while being editorially useless as a topic name).
    3. Absolute last resort: the highest-quality member's own label.
    """
    heading_members = [member for member in members if member.label_source == "heading"]
    if heading_members:
        counts = Counter(member.label for member in heading_members)
        return cap_label(counts.most_common(1)[0][0])

    ranked = sorted(members, key=lambda member: member.heading_quality, reverse=True)

    aggregated_sentences: List[str] = []
    for member in ranked:
        aggregated_sentences.extend(member.representative_sentences)
    label = generate_label_from_content(aggregated_sentences)
    if label:
        return label

    if ranked:
        return cap_label(ranked[0].label)
    return cap_label(members[0].label)


def build_section_topic_aggregates(
    groups: List[List[SectionProfile]],
) -> List[SectionTopicAggregate]:
    aggregates: List[SectionTopicAggregate] = []
    for members in groups:
        aggregate = SectionTopicAggregate(topic=choose_group_label(members))
        for member in members:
            aggregate.members.append(member)
            aggregate.competitors.add(member.section.competitor_index)
            if member.section.url:
                aggregate.urls.add(member.section.url)
            aggregate.sources[member.label_source] += 1
            aggregate.importance_values.append(member.heading_quality)
            if member.label_source == "heading":
                aggregate.heading_hits += 1
            aggregate.phrases.update(member.representative_sentences)
            aggregate.word_counts.append(member.section.word_count)
            aggregate.table_counts.append(len(member.section.tables))
            aggregate.list_counts.append(len(member.section.lists))
            aggregate.faq_flags.append(member.section.is_faq)
        aggregates.append(aggregate)
    return aggregates


def build_section_topic_dashboard(
    aggregates: List[SectionTopicAggregate],
    entities: List[Dict[str, Any]],
    total_competitors: int,
) -> List[Dict[str, Any]]:
    if not aggregates:
        return []

    heading_scores = normalized_score(
        {id(aggregate): aggregate.heading_hits for aggregate in aggregates}
    )

    rows = []
    for aggregate in aggregates:
        competitors_using = len(aggregate.competitors)
        coverage_ratio = competitors_using / max(1, total_competitors)
        average_quality = (
            float(np.mean(aggregate.importance_values))
            if aggregate.importance_values
            else 0.0
        )
        heading_frequency = heading_scores.get(id(aggregate), 0)

        importance = (
            coverage_ratio * 0.35
            + (average_quality / 100) * 0.30
            + heading_frequency * 0.35
        ) * 100

        attach_related_entities(aggregate, entities)

        average_word_count = (
            round(sum(aggregate.word_counts) / len(aggregate.word_counts), 1)
            if aggregate.word_counts
            else 0.0
        )
        average_table_count = (
            round(sum(aggregate.table_counts) / len(aggregate.table_counts), 2)
            if aggregate.table_counts
            else 0.0
        )
        average_list_count = (
            round(sum(aggregate.list_counts) / len(aggregate.list_counts), 2)
            if aggregate.list_counts
            else 0.0
        )
        faq_count = sum(1 for flag in aggregate.faq_flags if flag)
        faq_ratio = (
            round(faq_count / len(aggregate.faq_flags), 2)
            if aggregate.faq_flags
            else 0.0
        )

        rows.append(
            {
                "topic": aggregate.topic,
                "coverage": round(coverage_ratio * 100, 2),
                "coverage_percent": round(coverage_ratio * 100, 2),
                "competitors_using": competitors_using,
                "importance": round(importance, 2),
                "heading_hits": aggregate.heading_hits,
                "sources": sorted(aggregate.sources.keys()),
                "urls": sorted(aggregate.urls),
                "average_word_count": average_word_count,
                "average_table_count": average_table_count,
                "average_list_count": average_list_count,
                "faq_count": faq_count,
                "faq_ratio": faq_ratio,
                "related_phrases": [
                    phrase
                    for phrase, _ in aggregate.phrases.most_common(20)
                ],
                "related_entities": [
                    entity
                    for entity, _ in aggregate.related_entities.most_common(20)
                ],
                "competitor_indexes": sorted(aggregate.competitors),
                "competitors": sorted(aggregate.competitors),
                "missing_competitors": [
                    index
                    for index in range(total_competitors)
                    if index not in aggregate.competitors
                ],
            }
        )

    rows.sort(
        key=lambda item: (
            item["importance"],
            item["coverage"],
            item["heading_hits"],
        ),
        reverse=True,
    )
    return rows


def serialize_competitor_section_topics(
    profiles: List[SectionProfile],
) -> List[Dict[str, Any]]:
    rows = []
    for profile in sorted(profiles, key=lambda item: item.heading_quality, reverse=True):
        rows.append(
            {
                "topic": profile.label,
                "importance": round(profile.heading_quality, 2),
                "heading_hits": 1 if profile.label_source == "heading" else 0,
                "sources": [profile.label_source],
                "related_phrases": profile.representative_sentences,
                "word_count": profile.section.word_count,
                "paragraph_count": profile.section.paragraph_count,
                "table_count": len(profile.section.tables),
                "list_count": len(profile.section.lists),
                "image_count": len(profile.section.images),
                "internal_link_count": len(profile.section.internal_links),
                "external_link_count": len(profile.section.external_links),
                "is_faq": profile.section.is_faq,
            }
        )
    return rows


def generate_section_topic_intelligence(
    processed_competitors: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    The Section Intelligence entry point (formerly generate_topic_intelligence
    / cluster_sentence_topics in earlier, now-removed engine generations):

    For each competitor:
        split_into_sections() -> expand_large_sections() (oversized sections
        become paragraph-group sub-sections) -> build_section_profiles()
        -> competitor["topics"]
    Then, across the whole corpus at once:
        cluster_section_profiles() -> build_section_topic_aggregates()
        -> build_section_topic_dashboard()
    """
    all_profiles: List[SectionProfile] = []
    competitor_profiles: Dict[int, List[SectionProfile]] = defaultdict(list)

    for index, competitor in enumerate(processed_competitors):
        sections = split_into_sections(
            competitor.get("structure", {}),
            competitor,
            index,
        )
        sections = expand_large_sections(sections)
        profiles = build_section_profiles(sections)
        all_profiles.extend(profiles)
        competitor_profiles[index].extend(profiles)

    for index, competitor in enumerate(processed_competitors):
        competitor["topics"] = serialize_competitor_section_topics(
            competitor_profiles.get(index, [])
        )

    groups = cluster_section_profiles(all_profiles)
    aggregates = build_section_topic_aggregates(groups)
    return build_section_topic_dashboard(
        aggregates,
        entities,
        len(processed_competitors),
    )


def build_coverage(
    entities: List[Dict[str, Any]],
    topics: List[Dict[str, Any]],
    total_competitors: int,
) -> Dict[str, Any]:
    all_indexes = set(range(total_competitors))

    common_entities = [
        item["entity"]
        for item in entities
        if item["competitors_using"] == total_competitors
    ]
    rare_entities = [
        item["entity"]
        for item in entities
        if item["competitors_using"] == 1
    ]
    high_value_entities = [
        item["entity"]
        for item in entities
        if item["importance"] >= 70
        and item["competitors_using"] < total_competitors
    ]

    common_topics = [
        item["topic"]
        for item in topics
        if item["competitors_using"] == total_competitors
    ]
    rare_topics = [
        item["topic"]
        for item in topics
        if item["competitors_using"] == 1
    ]
    high_value_topics = [
        item["topic"]
        for item in topics
        if item["importance"] >= 70
        and item["competitors_using"] < total_competitors
    ]

    return {
        "total_competitors": total_competitors,
        "entities": [
            {
                "entity": item["entity"],
                "coverage": item["coverage"],
                "competitors_using": item["competitors_using"],
                "average_mentions": item["average_mentions"],
                "missing_competitors": sorted(
                    all_indexes - set(item.get("competitor_indexes", []))
                ),
            }
            for item in entities
        ],
        "topics": [
            {
                "topic": item["topic"],
                "coverage": item["coverage"],
                "competitors_using": item["competitors_using"],
                "missing_competitors": item["missing_competitors"],
            }
            for item in topics
        ],
        "gap_analysis": {
            "topics_missing_from_competitors": [
                {
                    "topic": item["topic"],
                    "missing_competitors": item["missing_competitors"],
                }
                for item in topics
                if item["missing_competitors"]
            ],
            "topics_unique_to_competitors": rare_topics,
            "entities_unique_to_competitors": rare_entities,
            "common_entities": common_entities,
            "common_topics": common_topics,
            "rare_topics": rare_topics,
            "high_value_topics": high_value_topics,
            "high_value_entities": high_value_entities,
        },
    }


def aggregate_statistics(
    competitors: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    topics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stats = [competitor.get("statistics", {}) for competitor in competitors]
    frame = pd.DataFrame(stats) if stats else pd.DataFrame()

    def total(column: str) -> int:
        if column not in frame:
            return 0
        return int(frame[column].fillna(0).sum())

    def average(column: str) -> float:
        if column not in frame or frame.empty:
            return 0.0
        return round(float(frame[column].fillna(0).mean()), 2)

    return {
        "competitors": len(competitors),
        "words": total("words"),
        "characters": total("characters"),
        "sentences": total("sentences"),
        "paragraphs": total("paragraphs"),
        "average_words": average("words"),
        "average_characters": average("characters"),
        "average_sentences": average("sentences"),
        "average_paragraphs": average("paragraphs"),
        "average_sentence_length": average("average_sentence_length"),
        "reading_time": round(
            sum(stat.get("reading_time", 0) for stat in stats),
            2,
        ),
        "entity_count": sum(item.get("mentions", 0) for item in entities),
        "topic_count": sum(item.get("sentence_count", 0) for item in topics),
        "unique_entities": len(entities),
        "unique_topics": len(topics),
    }


def competitor_text(competitor: Dict[str, Any]) -> str:
    text = clean_text(competitor.get("text", ""))
    html = candidate_html(competitor)
    if html and not text:
        text = extract_text_from_html(html)
    return text


def serialize_competitor_entities(
    entities: List[EntityCandidate],
) -> List[Dict[str, Any]]:
    rows = []
    for entity in sorted(
        entities,
        key=lambda item: (item.mentions, item.confidence),
        reverse=True,
    ):
        rows.append(
            {
                "entity": entity.normalized,
                "type": entity.type,
                "mentions": entity.mentions,
                "confidence": round(entity.confidence, 4),
                "extractor_sources": entity.source.split("|"),
                "in_heading": entity.in_heading,
                "first_position": entity.first_char,
            }
        )
    return rows


def process_competitors(
    competitors: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    processed: List[Dict[str, Any]] = []
    texts: List[str] = []

    for index, competitor in enumerate(competitors):
        item = dict(competitor)
        text = competitor_text(item)
        item["text"] = text
        item["metadata"] = clean_metadata(item.get("metadata", {}))
        item["position"] = safe_number(item.get("position")) or index + 1
        item["title"] = (
            compact_text(item.get("title", ""))
            or item["metadata"].get("title", "")
        )
        item["url"] = compact_text(item.get("url", ""))
        item["structure"] = extract_structure(item, text)
        texts.append(text[:MAX_SPACY_CHARS])
        processed.append(item)

    docs = list(get_nlp().pipe(texts, batch_size=2))

    for item, doc in zip(processed, docs):
        paragraph_count = len(item.get("structure", {}).get("paragraphs", []))
        item["statistics"] = document_statistics(
            item.get("text", ""),
            doc,
            paragraph_count,
        )

    return processed, docs


# ---------------------------------------------------------------------------
# Part 5: Structure Intelligence.
#
# Heading strings are ground truth: no embeddings needed to decide whether a
# heading is a "topic", only to merge differently-worded headings that mean
# the same section (e.g. "Types of Mutual Funds" vs "Different Kinds of
# Mutual Fund"). Clustering is done per heading level (h1..h6, plus "text"
# for non-HTML sources) so H2 sections and H3 sections are never mixed.
#
# Deferred: canonical heading *sequence* (a single recommended outline order
# via something like longest-common-subsequence across competitors). This
# pass only computes average position per section; the full ordering
# algorithm is a separate piece of work.
# ---------------------------------------------------------------------------

HEADING_SECTION_MERGE_THRESHOLD = 88
HEADING_SECTION_SEMANTIC_THRESHOLD = 0.80

FAQ_HEADING_PATTERNS = [
    r"\bfaq\b",
    r"frequently asked questions",
    r"common questions",
]


@dataclass
class HeadingCandidate:
    text: str
    normalized: str
    level: str  # "h1".."h6", or "text" for non-HTML sources
    position: int  # index among headings of the same level in this document
    competitor_index: int
    url: str


@dataclass
class HeadingSectionAggregate:
    label: str
    level: str
    members: List[HeadingCandidate] = field(default_factory=list)
    competitors: Set[int] = field(default_factory=set)
    urls: Set[str] = field(default_factory=set)
    positions: List[int] = field(default_factory=list)


def is_faq_heading(text: str) -> bool:
    lowered = compact_text(text).lower()
    if not lowered:
        return False
    if lowered.endswith("?"):
        return True
    return any(re.search(pattern, lowered) for pattern in FAQ_HEADING_PATTERNS)


def build_heading_candidates_for_competitor(
    structure: Dict[str, Any],
    competitor: Dict[str, Any],
    competitor_index: int,
) -> List[HeadingCandidate]:
    candidates: List[HeadingCandidate] = []

    headings = structure.get("headings", {})
    for level, values in headings.items():
        position = 0
        for value in values:
            if not value or is_boilerplate_heading(value):
                continue
            normalized = normalize_entity_text(value)
            if not normalized:
                continue
            candidates.append(
                HeadingCandidate(
                    text=value,
                    normalized=normalized,
                    level=level,
                    position=position,
                    competitor_index=competitor_index,
                    url=competitor.get("url", ""),
                )
            )
            position += 1

    position = 0
    for value in structure.get("logical_headings", []):
        if not value or is_boilerplate_heading(value):
            continue
        normalized = normalize_entity_text(value)
        if not normalized:
            continue
        candidates.append(
            HeadingCandidate(
                text=value,
                normalized=normalized,
                level="text",
                position=position,
                competitor_index=competitor_index,
                url=competitor.get("url", ""),
            )
        )
        position += 1

    return candidates


def embed_heading_candidates(candidates: List[HeadingCandidate]) -> List[np.ndarray]:
    if not candidates:
        return []

    texts = [candidate.normalized for candidate in candidates]
    try:
        embeddings = get_sentence_model().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except TypeError:
        embeddings = get_sentence_model().encode(
            texts,
            normalize_embeddings=True,
        )
    return list(embeddings)


def cluster_heading_candidates(
    candidates: List[HeadingCandidate],
) -> List[HeadingSectionAggregate]:
    """
    One batched encode() call for every heading candidate at this level.
    Merges via fuzzy label match OR cosine similarity on the already-computed
    embeddings (no repeated encode calls).
    """
    if not candidates:
        return []

    embeddings = embed_heading_candidates(candidates)

    aggregates: List[HeadingSectionAggregate] = []
    aggregate_embeddings: List[np.ndarray] = []

    for candidate, embedding in zip(candidates, embeddings):
        match_index = None
        for index, aggregate in enumerate(aggregates):
            representative = aggregate.members[0]
            if (
                fuzz.token_sort_ratio(representative.normalized, candidate.normalized)
                >= HEADING_SECTION_MERGE_THRESHOLD
            ):
                match_index = index
                break
            semantic_score = float(
                cosine_similarity([aggregate_embeddings[index]], [embedding])[0][0]
            )
            if semantic_score >= HEADING_SECTION_SEMANTIC_THRESHOLD:
                match_index = index
                break

        if match_index is None:
            aggregates.append(
                HeadingSectionAggregate(label=candidate.text, level=candidate.level)
            )
            aggregate_embeddings.append(embedding)
            match_index = len(aggregates) - 1

        aggregate = aggregates[match_index]
        aggregate.members.append(candidate)
        aggregate.competitors.add(candidate.competitor_index)
        if candidate.url:
            aggregate.urls.add(candidate.url)
        aggregate.positions.append(candidate.position)

    return aggregates


def cluster_headings_by_level(
    candidates: List[HeadingCandidate],
) -> Dict[str, List[HeadingSectionAggregate]]:
    by_level: Dict[str, List[HeadingCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_level[candidate.level].append(candidate)

    return {
        level: cluster_heading_candidates(level_candidates)
        for level, level_candidates in by_level.items()
    }


def build_heading_section_dashboard(
    aggregates: List[HeadingSectionAggregate],
    total_competitors: int,
) -> List[Dict[str, Any]]:
    if not aggregates:
        return []

    rows = []
    for aggregate in aggregates:
        competitors_using = len(aggregate.competitors)
        coverage_ratio = competitors_using / max(1, total_competitors)
        average_position = (
            round(sum(aggregate.positions) / len(aggregate.positions), 2)
            if aggregate.positions
            else None
        )

        rows.append(
            {
                "section": aggregate.label,
                "level": aggregate.level,
                "coverage": round(coverage_ratio * 100, 2),
                "coverage_percent": round(coverage_ratio * 100, 2),
                "competitors_using": competitors_using,
                "competitor_indexes": sorted(aggregate.competitors),
                "missing_competitors": [
                    index
                    for index in range(total_competitors)
                    if index not in aggregate.competitors
                ],
                "average_position": average_position,
                "occurrences": len(aggregate.members),
                "urls": sorted(aggregate.urls),
            }
        )

    rows.sort(
        key=lambda item: (item["coverage"], item["occurrences"]),
        reverse=True,
    )
    return rows


def compute_competitor_structure_profile(
    structure: Dict[str, Any],
    competitor_index: int,
) -> Dict[str, Any]:
    headings = structure.get("headings", {})
    heading_counts = {
        level: len([value for value in values if not is_boilerplate_heading(value)])
        for level, values in headings.items()
    }

    all_heading_texts = [
        text
        for values in headings.values()
        for text in values
    ] + list(structure.get("logical_headings", []))
    faq_headings = [text for text in all_heading_texts if is_faq_heading(text)]

    return {
        "competitor_index": competitor_index,
        "heading_counts": heading_counts,
        "table_count": len(structure.get("tables", [])),
        "list_count": len(structure.get("lists", [])),
        "faq_heading_count": len(faq_headings),
        "has_faq_section": len(faq_headings) > 0,
    }


def build_structure_summary(
    profiles: List[Dict[str, Any]],
    total_competitors: int,
) -> Dict[str, Any]:
    if not profiles or not total_competitors:
        return {
            "table_usage_percent": 0.0,
            "list_usage_percent": 0.0,
            "faq_usage_percent": 0.0,
            "average_table_count": 0.0,
            "average_list_count": 0.0,
            "average_faq_heading_count": 0.0,
            "average_heading_counts": {},
        }

    table_users = sum(1 for profile in profiles if profile["table_count"] > 0)
    list_users = sum(1 for profile in profiles if profile["list_count"] > 0)
    faq_users = sum(1 for profile in profiles if profile["has_faq_section"])

    levels = {level for profile in profiles for level in profile["heading_counts"]}
    average_heading_counts = {
        level: round(
            sum(profile["heading_counts"].get(level, 0) for profile in profiles)
            / total_competitors,
            2,
        )
        for level in levels
    }

    return {
        "table_usage_percent": round(table_users / total_competitors * 100, 2),
        "list_usage_percent": round(list_users / total_competitors * 100, 2),
        "faq_usage_percent": round(faq_users / total_competitors * 100, 2),
        "average_table_count": round(
            sum(profile["table_count"] for profile in profiles) / total_competitors, 2
        ),
        "average_list_count": round(
            sum(profile["list_count"] for profile in profiles) / total_competitors, 2
        ),
        "average_faq_heading_count": round(
            sum(profile["faq_heading_count"] for profile in profiles)
            / total_competitors,
            2,
        ),
        "average_heading_counts": average_heading_counts,
    }


def generate_structure_intelligence(
    processed_competitors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    For each competitor: extract H1-H6 (+ logical/text headings), tables,
    lists, FAQ-style headings. Cluster headings per level across the whole
    corpus (common/unique/missing sections, average position). Also writes
    competitor["structure_sections"], mirroring competitor["entities"] and
    competitor["topics"].
    """
    total_competitors = len(processed_competitors)

    all_heading_candidates: List[HeadingCandidate] = []
    structure_profiles: List[Dict[str, Any]] = []

    for index, competitor in enumerate(processed_competitors):
        structure = competitor.get("structure", {})
        all_heading_candidates.extend(
            build_heading_candidates_for_competitor(structure, competitor, index)
        )
        structure_profiles.append(
            compute_competitor_structure_profile(structure, index)
        )

    by_level = cluster_headings_by_level(all_heading_candidates)
    sections_by_level: Dict[str, List[Dict[str, Any]]] = {
        level: build_heading_section_dashboard(aggregates, total_competitors)
        for level, aggregates in by_level.items()
    }

    for index, competitor in enumerate(processed_competitors):
        competitor["structure_sections"] = {
            level: [row for row in rows if index in row["competitor_indexes"]]
            for level, rows in sections_by_level.items()
        }

    return {
        "sections_by_level": sections_by_level,
        "most_common_by_level": {
            level: (rows[0]["section"] if rows else None)
            for level, rows in sections_by_level.items()
        },
        "common_sections_by_level": {
            level: [
                row["section"]
                for row in rows
                if row["competitors_using"] == total_competitors
            ]
            for level, rows in sections_by_level.items()
        },
        "unique_sections_by_level": {
            level: [row["section"] for row in rows if row["competitors_using"] == 1]
            for level, rows in sections_by_level.items()
        },
        "competitor_profiles": structure_profiles,
        "summary": build_structure_summary(structure_profiles, total_competitors),
    }


def run_intelligence_engine(
    competitors: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Public entry point used by app.py.

    Returns:
    {
        "competitors": ...,
        "entities": ...,
        "topics": ...,
        "coverage": ...,
        "statistics": ...,
        "structure": ...
    }
    """
    processed_competitors, docs = process_competitors(competitors or [])

    document_entities: List[List[EntityCandidate]] = []
    all_candidates: List[EntityCandidate] = []

    for index, (competitor, doc) in enumerate(zip(processed_competitors, docs)):
        candidates = extract_entities_for_document(
            doc=doc,
            text=competitor.get("text", ""),
            structure=competitor.get("structure", {}),
            competitor=competitor,
            competitor_index=index,
        )
        competitor["entities"] = serialize_competitor_entities(candidates)
        document_entities.append(candidates)
        all_candidates.extend(candidates)

    entity_aggregates = merge_entities_across_competitors(all_candidates)
    add_entity_co_occurrence(entity_aggregates, document_entities)
    entities = build_entity_dashboard(
        entity_aggregates,
        len(processed_competitors),
    )

    topics = generate_section_topic_intelligence(
        processed_competitors,
        entities,
    )

    coverage = build_coverage(
        entities,
        topics,
        len(processed_competitors),
    )
    statistics = aggregate_statistics(
        processed_competitors,
        entities,
        topics,
    )

    structure = generate_structure_intelligence(processed_competitors)

    return {
        "competitors": processed_competitors,
        "entities": entities,
        "topics": topics,
        "coverage": coverage,
        "statistics": statistics,
        "structure": structure,
    }