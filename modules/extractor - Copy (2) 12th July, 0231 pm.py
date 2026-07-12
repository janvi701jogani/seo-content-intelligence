"""
Generic SEO Intelligence Engine.

Single-file extractor for competitor content intelligence:
- competitor statistics and structure
- generic entity extraction with spaCy, GLiNER, KeyBERT, YAKE
- generic normalization and RapidFuzz merging
- semantic sentence topic clustering
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
TOPIC_MERGE_THRESHOLD = 90

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
class TopicAggregate:
    topic: str
    competitors: Set[int] = field(default_factory=set)
    sentence_indexes: List[Tuple[int, int]] = field(default_factory=list)
    representative_sentences: List[str] = field(default_factory=list)
    related_entities: Counter = field(default_factory=Counter)
    co_occurrence: Counter = field(default_factory=Counter)
    centroid: Optional[np.ndarray] = None


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

    for line in lines:
        if is_explicit_text_heading(line):
            heading = re.sub(r"^#{1,6}\s+", "", line).rstrip(":").strip()
            logical_headings.append(heading)

        elif re.match(r"^[-*+]\s+\S+", line) or re.match(r"^\d+[\.)]\s+\S+", line):
            item = re.sub(r"^([-*+]|\d+[\.)])\s+", "", line).strip()
            list_items.append(item)

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


def extract_sentences(doc: Any, min_words: int = 5) -> List[str]:
    sentences = []

    spans = doc.sents if doc.has_annotation("SENT_START") else []

    for sent in spans:
        sentence = compact_text(sent.text)

        if len(sentence.split()) >= min_words:
            sentences.append(sentence)

    if not sentences:
        for sentence in re.split(r"(?<=[.!?])\s+", doc.text):
            sentence = compact_text(sentence)

            if len(sentence.split()) >= min_words:
                sentences.append(sentence)

    return sentences


def topic_label_from_sentences(sentences: Sequence[str]) -> str:
    joined = " ".join(sentences[:8])

    try:
        keywords = get_yake_extractor().extract_keywords(joined)
    except Exception:
        keywords = []

    for keyword, _ in keywords:
        normalized = normalize_entity_text(keyword)

        if valid_entity_phrase(normalized):
            return normalized

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
                return Counter(chunks).most_common(1)[0][0]

        except ValueError:
            pass

    return compact_text(sentences[0])[:90].rstrip(".")


def make_agglomerative(distance_threshold: float):
    try:
        return AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )


def cluster_sentence_topics(
    processed_competitors: List[Dict[str, Any]],
    document_entities: List[List[EntityCandidate]],
) -> List[Dict[str, Any]]:
    sentence_records: List[Dict[str, Any]] = []

    for competitor_index, competitor in enumerate(processed_competitors):
        for sentence_index, sentence in enumerate(competitor.get("_sentences", [])):
            sentence_records.append(
                {
                    "competitor_index": competitor_index,
                    "sentence_index": sentence_index,
                    "sentence": sentence,
                }
            )

    if not sentence_records:
        return []

    sentences = [record["sentence"] for record in sentence_records]

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

    if len(sentences) == 1:
        labels = np.array([0])
    else:
        distance_threshold = 0.38 if len(sentences) < 80 else 0.42
        clustering = make_agglomerative(distance_threshold)
        labels = clustering.fit_predict(embeddings)

    entity_lookup: Dict[int, Set[str]] = {
        competitor_index: {entity.normalized for entity in entities}
        for competitor_index, entities in enumerate(document_entities)
    }

    topic_groups: Dict[int, List[int]] = defaultdict(list)

    for index, label in enumerate(labels):
        topic_groups[int(label)].append(index)

    raw_topics: List[TopicAggregate] = []

    for indexes in topic_groups.values():
        cluster_sentences = [sentences[index] for index in indexes]
        cluster_embeddings = embeddings[indexes]
        centroid = np.mean(cluster_embeddings, axis=0)
        similarities = cosine_similarity([centroid], cluster_embeddings)[0]
        ranked = [indexes[i] for i in np.argsort(similarities)[::-1]]
        representative = [
            sentence_records[index]["sentence"]
            for index in ranked[:3]
        ]

        topic = TopicAggregate(
            topic=topic_label_from_sentences(representative),
            centroid=centroid,
        )

        for index in indexes:
            record = sentence_records[index]
            competitor_index = record["competitor_index"]

            topic.competitors.add(competitor_index)
            topic.sentence_indexes.append(
                (
                    competitor_index,
                    record["sentence_index"],
                )
            )

            sentence_norm = normalize_entity_text(record["sentence"])

            for entity in entity_lookup.get(competitor_index, set()):
                if fuzz.partial_ratio(entity, sentence_norm) >= 88:
                    topic.related_entities[entity] += 1

        topic.representative_sentences = representative
        raw_topics.append(topic)

    merged = merge_topic_aggregates(raw_topics)
    add_topic_co_occurrence(merged)

    return build_topic_dashboard(merged, len(processed_competitors))


def merge_topic_aggregates(topics: List[TopicAggregate]) -> List[TopicAggregate]:
    merged: List[TopicAggregate] = []

    for topic in topics:
        match = None

        for existing in merged:
            if fuzz.token_sort_ratio(existing.topic, topic.topic) >= TOPIC_MERGE_THRESHOLD:
                match = existing
                break

        if match is None:
            merged.append(topic)
            continue

        match.competitors.update(topic.competitors)
        match.sentence_indexes.extend(topic.sentence_indexes)
        match.representative_sentences.extend(topic.representative_sentences)
        match.representative_sentences = list(
            dict.fromkeys(match.representative_sentences)
        )[:5]
        match.related_entities.update(topic.related_entities)

        if match.centroid is not None and topic.centroid is not None:
            match.centroid = np.mean([match.centroid, topic.centroid], axis=0)

    return merged


def add_topic_co_occurrence(topics: List[TopicAggregate]) -> None:
    by_competitor: Dict[int, List[TopicAggregate]] = defaultdict(list)

    for topic in topics:
        for competitor_index in topic.competitors:
            by_competitor[competitor_index].append(topic)

    for competitor_topics in by_competitor.values():
        for left, right in combinations(competitor_topics, 2):
            left.co_occurrence[right.topic] += 1
            right.co_occurrence[left.topic] += 1


def build_topic_dashboard(
    topics: List[TopicAggregate],
    total_competitors: int,
) -> List[Dict[str, Any]]:
    if not topics:
        return []

    frequency_scores = normalized_score(
        {
            topic.topic: len(topic.sentence_indexes)
            for topic in topics
        }
    )

    co_scores = normalized_score(
        {
            topic.topic: sum(topic.co_occurrence.values())
            for topic in topics
        }
    )

    rows = []

    for topic in topics:
        competitors_using = len(topic.competitors)
        coverage_ratio = competitors_using / max(1, total_competitors)
        related_score = min(1.0, len(topic.related_entities) / 12)

        importance = (
            coverage_ratio * 0.42
            + frequency_scores.get(topic.topic, 0) * 0.24
            + related_score * 0.18
            + co_scores.get(topic.topic, 0) * 0.16
        ) * 100

        rows.append(
            {
                "topic": topic.topic,
                "coverage": round(coverage_ratio * 100, 2),
                "coverage_percent": round(coverage_ratio * 100, 2),
                "competitors_using": competitors_using,
                "competitors": sorted(topic.competitors),
                "missing_competitors": [
                    index
                    for index in range(total_competitors)
                    if index not in topic.competitors
                ],
                "importance": round(importance, 2),
                "representative_sentences": topic.representative_sentences[:5],
                "related_entities": [
                    entity
                    for entity, _ in topic.related_entities.most_common(15)
                ],
                "related_topics": [
                    topic_name
                    for topic_name, _ in topic.co_occurrence.most_common(10)
                ],
                "sentence_count": len(topic.sentence_indexes),
            }
        )

    rows.sort(
        key=lambda item: (
            item["importance"],
            item["coverage"],
            item["sentence_count"],
        ),
        reverse=True,
    )

    return rows


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
        item["_sentences"] = extract_sentences(doc)

    return processed, docs


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
        "statistics": ...
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

    topics = cluster_sentence_topics(
        processed_competitors,
        document_entities,
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

    for competitor in processed_competitors:
        competitor.pop("_sentences", None)

    return {
        "competitors": processed_competitors,
        "entities": entities,
        "topics": topics,
        "coverage": coverage,
        "statistics": statistics,
    }