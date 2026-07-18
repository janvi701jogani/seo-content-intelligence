"""
Research Intelligence module.

Self-contained: collection (OpenAlex by default, Semantic Scholar when a
key is supplied) and the information-gain processing pipeline both live
here. Reuses utilities from modules.extractor (which is NOT modified).

Information gain = what peer-reviewed research establishes that the
ranking competitors (and, optionally, the community) do not cover.

Research Papers (recent + reviews first, no books, abstract required)
        |
        +-- Statistics (corpus overview)
        +-- Papers (title, journal, year, review, citations, abstract, url)
        +-- Research Concepts (entities/keyphrases from abstracts)
        +-- Data Points (numeric findings competitors can cite)
        +-- Information Gain (research concepts vs competitor coverage)
                 |
                 v
      Research Insights Dashboard
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import requests

from modules.extractor import (
    compact_text,
    get_nlp,
    normalize_entity_text,
    valid_entity_phrase,
)

from rapidfuzz import fuzz

SEMANTIC_SCHOLAR_ENDPOINT = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)
SEMANTIC_SCHOLAR_FIELDS = (
    "title,abstract,year,citationCount,authors,venue,url,externalIds,"
    "publicationTypes,publicationDate,journal"
)

# OpenAlex: free, no API key, reliable rate limits -> default source so
# the tool works for everyone without keys. Semantic Scholar is used only
# when an API key is supplied (it rate-limits keyless traffic to 429).
OPENALEX_ENDPOINT = "https://api.openalex.org/works"
OPENALEX_SELECT = (
    "id,title,publication_year,publication_date,cited_by_count,type,"
    "primary_location,abstract_inverted_index,doi,authorships"
)
# Adding a contact email puts requests in OpenAlex's faster "polite pool".
# Optional; leave blank if none.
OPENALEX_MAILTO = ""

# Semantic Scholar publication types that are not scannable article
# content. Books / book sections excluded so full books never enter.
EXCLUDED_PUBLICATION_TYPES = {"Book", "BookSection"}

# OpenAlex work types excluded (books and non-article material).
EXCLUDED_OPENALEX_TYPES = {
    "book",
    "book-chapter",
    "dataset",
    "paratext",
    "reference-entry",
    "peer-review",
    "grant",
}

# Types treated as review / synthesis articles, surfaced first.
REVIEW_PUBLICATION_TYPES = {"Review", "MetaAnalysis"}

# An item with no abstract cannot be scanned for concepts or data points,
# so abstract-less results (common for books and datasets) are dropped.
REQUIRE_ABSTRACT = True

# Over-fetch, then filter + sort client-side. Semantic Scholar's search
# endpoint has no sort option, so recency/review ordering is applied here.
FETCH_MULTIPLIER = 3

# -----------------------------
# Research processing settings
# -----------------------------

MAX_ROWS_PER_LAYER = 30
RECENT_YEARS = 3
GAP_MATCH_THRESHOLD = 85
MIN_CONCEPT_PAPERS = 2
REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 3

# Numeric findings worth citing: percentages, multipliers, money, sample
# sizes, p-values, basis points, ranges. Structural, niche-agnostic.
DATA_POINT_PATTERN = re.compile(
    r"(\d[\d,\.]*\s?%|"
    r"\$\s?\d[\d,\.]*(?:\s?(?:billion|million|trillion|bn|m|k))?|"
    r"\b\d[\d,\.]*\s?(?:x|times|fold)\b|"
    r"\bp\s?[<=>]\s?0?\.\d+|"
    r"\bn\s?=\s?\d[\d,]*|"
    r"\b\d[\d,\.]*\s?(?:bps|basis points)\b)",
    re.IGNORECASE,
)


@dataclass
class Paper:
    title: str
    abstract: str
    year: Optional[int]
    citation_count: int
    authors: List[str]
    venue: str
    journal: str
    url: str
    paper_id: str
    is_review: bool = False
    publication_date: str = ""


class ResearchCollector:
    """
    Multi-source, key-optional.

    - No key (default): OpenAlex, which is reliable and keyless.
    - With a Semantic Scholar key: Semantic Scholar first, OpenAlex as
      fallback.

    If the chosen primary returns nothing (rate limit, outage, over-
    filtered), the other source is tried automatically. The query is the
    keyword verbatim; each API does its own relevance matching. Results
    are filtered (no books, abstract required) and sorted recency +
    reviews first.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        contact_email: Optional[str] = None,
    ):
        self.api_key = api_key or ""
        self.contact_email = contact_email or OPENALEX_MAILTO

    def search(self, keyword: str, limit: int = 25) -> List[Paper]:
        if self.api_key:
            sources = [self._search_semantic_scholar, self._search_openalex]
        else:
            sources = [self._search_openalex, self._search_semantic_scholar]

        papers: List[Paper] = []
        for source in sources:
            try:
                papers = source(keyword, limit)
            except Exception:
                papers = []
            if papers:
                break

        papers = _sort_recent_reviews_first(papers)
        return papers[:limit]

    # -----------------------------
    # OpenAlex (default, keyless)
    # -----------------------------

    def _search_openalex(self, keyword: str, limit: int) -> List[Paper]:
        params = {
            "search": keyword,
            "per_page": min(max(limit * FETCH_MULTIPLIER, 1), 200),
            "select": OPENALEX_SELECT,
        }
        if self.contact_email:
            params["mailto"] = self.contact_email
        data = self._get_with_retries(OPENALEX_ENDPOINT, params, {})
        papers: List[Paper] = []
        for item in data.get("results", []) or []:
            title = compact_text(item.get("title") or item.get("display_name") or "")
            if not title:
                continue

            work_type = (item.get("type") or "").lower()
            if work_type in EXCLUDED_OPENALEX_TYPES:
                continue

            abstract = _abstract_from_inverted(
                item.get("abstract_inverted_index")
            )
            if REQUIRE_ABSTRACT and not abstract:
                continue

            location = item.get("primary_location") or {}
            source = location.get("source") or {}
            journal = compact_text(source.get("display_name") or "")

            doi = item.get("doi") or ""
            url = doi or location.get("landing_page_url") or item.get("id") or ""

            authors = [
                compact_text((a.get("author") or {}).get("display_name", ""))
                for a in (item.get("authorships") or [])
                if (a.get("author") or {}).get("display_name")
            ]

            is_review = work_type == "review" or bool(
                re.search(r"\breview\b|\bmeta-?analysis\b", title, re.IGNORECASE)
            )

            papers.append(
                Paper(
                    title=title,
                    abstract=abstract,
                    year=item.get("publication_year"),
                    citation_count=int(item.get("cited_by_count") or 0),
                    authors=authors,
                    venue=journal,
                    journal=journal,
                    url=url,
                    paper_id=str(item.get("id") or ""),
                    is_review=is_review,
                    publication_date=compact_text(item.get("publication_date") or ""),
                )
            )
        return papers

    # -----------------------------
    # Semantic Scholar (used when a key is supplied)
    # -----------------------------

    def _search_semantic_scholar(self, keyword: str, limit: int) -> List[Paper]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        params = {
            "query": keyword,
            "limit": min(max(limit * FETCH_MULTIPLIER, 1), 100),
            "fields": SEMANTIC_SCHOLAR_FIELDS,
        }
        data = self._get_with_retries(
            SEMANTIC_SCHOLAR_ENDPOINT, params, headers
        )
        papers: List[Paper] = []
        for item in data.get("data", []) or []:
            title = compact_text(item.get("title") or "")
            if not title:
                continue

            publication_types = item.get("publicationTypes") or []
            if any(t in EXCLUDED_PUBLICATION_TYPES for t in publication_types):
                continue

            abstract = compact_text(item.get("abstract") or "")
            if REQUIRE_ABSTRACT and not abstract:
                continue

            authors = [
                compact_text(author.get("name", ""))
                for author in (item.get("authors") or [])
                if author.get("name")
            ]
            external = item.get("externalIds") or {}
            url = item.get("url") or ""
            if not url and external.get("DOI"):
                url = f"https://doi.org/{external['DOI']}"

            journal_obj = item.get("journal") or {}
            journal = compact_text(
                journal_obj.get("name", "") if isinstance(journal_obj, dict) else ""
            ) or compact_text(item.get("venue") or "")

            is_review = any(
                t in REVIEW_PUBLICATION_TYPES for t in publication_types
            ) or bool(re.search(r"\breview\b|\bmeta-?analysis\b", title, re.IGNORECASE))

            papers.append(
                Paper(
                    title=title,
                    abstract=abstract,
                    year=item.get("year"),
                    citation_count=int(item.get("citationCount") or 0),
                    authors=authors,
                    venue=compact_text(item.get("venue") or ""),
                    journal=journal,
                    url=url,
                    paper_id=str(item.get("paperId") or ""),
                    is_review=is_review,
                    publication_date=compact_text(item.get("publicationDate") or ""),
                )
            )
        return papers

    @staticmethod
    def _get_with_retries(
        endpoint: str,
        params: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        for attempt in range(REQUEST_RETRIES):
            response = requests.get(
                endpoint,
                headers=headers or None,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            # 429 = rate limited; back off and retry.
            if response.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        # Persistent rate limiting: return empty so the caller can fall
        # back to the other source instead of raising.
        return {}

    @staticmethod
    def to_dict(papers: List[Paper]) -> List[Dict[str, Any]]:
        return [asdict(paper) for paper in papers]


# ---------------------------------------------------------------------------
# Research Intelligence pipeline
# ---------------------------------------------------------------------------


def _abstract_from_inverted(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    OpenAlex returns abstracts as an inverted index {word: [positions]}.
    Reconstruct the running text by placing each word at its positions.
    """
    if not inverted_index:
        return ""
    positioned: List[tuple] = []
    for word, positions in inverted_index.items():
        for position in positions:
            positioned.append((position, word))
    positioned.sort(key=lambda pair: pair[0])
    return compact_text(" ".join(word for _, word in positioned))


def _paper_text(paper: Paper) -> str:
    return compact_text(f"{paper.title}. {paper.abstract}")


def _sort_recent_reviews_first(papers: List[Paper]) -> List[Paper]:
    """
    Ordering priority:
    1. Year, newest first (recency is the primary ask).
    2. Reviews / meta-analyses ahead of primary studies in the same year.
    3. Exact publication date, newest first (tie-break within a year).
    4. Citation count, highest first.
    Papers with no year sink to the bottom.
    """
    return sorted(
        papers,
        key=lambda paper: (
            paper.year if paper.year is not None else -1,
            1 if paper.is_review else 0,
            paper.publication_date,
            paper.citation_count,
        ),
        reverse=True,
    )


def _paper_row(paper: Paper) -> Dict[str, Any]:
    return {
        "title": paper.title,
        "journal": paper.journal,
        "year": paper.year,
        "review": "Yes" if paper.is_review else "",
        "citations": paper.citation_count,
        "abstract": paper.abstract,
        "url": paper.url,
    }


# -----------------------------
# Layer 1: statistics
# -----------------------------


def build_research_statistics(papers: Sequence[Paper]) -> Dict[str, Any]:
    if not papers:
        return {}
    years = [paper.year for paper in papers if paper.year]
    citations = [paper.citation_count for paper in papers]
    journals = {paper.journal for paper in papers if paper.journal}
    reviews = sum(1 for paper in papers if paper.is_review)
    if years:
        start, end = min(years), max(years)
        year_span = str(start) if start == end else f"{start}-{end}"
    else:
        year_span = ""
    return {
        "papers": len(papers),
        "reviews": reviews,
        "journals": len(journals),
        "total_citations": sum(citations),
        "average_citations": round(sum(citations) / len(citations)) if citations else 0,
        "year_span": year_span,
    }


# -----------------------------
# Layer 2: papers (recent + reviews first, full fields)
# -----------------------------


def build_papers(papers: Sequence[Paper]) -> List[Dict[str, Any]]:
    """
    Every kept paper, already ordered recent + reviews first by search().
    Columns: title, journal, year, review, citations, abstract, url.
    """
    return [_paper_row(paper) for paper in papers]


# -----------------------------
# Layer 3: research concepts (spaCy NER + noun chunks over abstracts)
# -----------------------------


def _extract_concepts(papers: Sequence[Paper]) -> Dict[str, Dict[str, Any]]:
    """
    Returns {concept: {"papers": count, "citations": weighted_citations}}.
    A concept is credited once per paper it appears in, weighted by that
    paper's citation count so well-supported concepts rank higher.
    """
    nlp = get_nlp()
    has_ner = "ner" in nlp.pipe_names
    concepts: Dict[str, Dict[str, Any]] = {}
    texts = [_paper_text(paper)[:2000] for paper in papers]
    for paper, doc in zip(papers, nlp.pipe(texts, batch_size=32)):
        seen: set = set()
        spans = []
        if has_ner:
            spans.extend(ent.text for ent in doc.ents)
        if hasattr(doc, "noun_chunks"):
            try:
                spans.extend(chunk.text for chunk in doc.noun_chunks)
            except ValueError:
                pass
        for span in spans:
            name = normalize_entity_text(span)
            if not valid_entity_phrase(name) or name in seen:
                continue
            seen.add(name)
            record = concepts.setdefault(name, {"papers": 0, "citations": 0})
            record["papers"] += 1
            record["citations"] += paper.citation_count
    return concepts


def build_research_concepts(
    concepts: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for name, record in concepts.items():
        if record["papers"] < MIN_CONCEPT_PAPERS:
            continue
        rows.append(
            {
                "concept": name,
                "papers": record["papers"],
                "weighted_citations": record["citations"],
            }
        )
    rows.sort(
        key=lambda row: (row["papers"], row["weighted_citations"]),
        reverse=True,
    )
    return rows[:MAX_ROWS_PER_LAYER]


# -----------------------------
# Layer 4: data points (numeric findings to cite)
# -----------------------------


def build_data_points(papers: Sequence[Paper]) -> List[Dict[str, Any]]:
    rows = []
    seen: set = set()
    for paper in papers:
        if not paper.abstract:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paper.abstract):
            if not DATA_POINT_PATTERN.search(sentence):
                continue
            sentence = compact_text(sentence)
            key = sentence.lower()
            if key in seen or len(sentence.split()) < 4:
                continue
            seen.add(key)
            figures = DATA_POINT_PATTERN.findall(sentence)
            rows.append(
                {
                    "finding": sentence,
                    "figures": ", ".join(figures[:5]),
                    "source": paper.title,
                    "year": paper.year,
                    "url": paper.url,
                }
            )
            if len(rows) >= MAX_ROWS_PER_LAYER:
                return rows
    return rows


# -----------------------------
# Layer 5: information gain (research vs competitor coverage)
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


def build_information_gain(
    concepts: List[Dict[str, Any]],
    competitor_entities: Optional[List[Dict[str, Any]]],
    competitor_topics: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Concepts that research discusses (research_papers) but competitors do
    not cover (competitor_coverage). Sorted so the highest research
    support with the least competitor coverage rises to the top -- those
    are the unique, defensible content angles.
    """
    rows = []
    for concept in concepts:
        term = concept["concept"]
        coverage = _competitor_coverage(
            term,
            competitor_entities,
            competitor_topics,
        )
        rows.append(
            {
                "concept": term,
                "research_papers": concept["papers"],
                "weighted_citations": concept["weighted_citations"],
                "competitor_coverage": coverage,
            }
        )
    rows.sort(
        key=lambda row: (
            row["competitor_coverage"],
            -row["research_papers"],
            -row["weighted_citations"],
        )
    )
    return rows[:MAX_ROWS_PER_LAYER]


# -----------------------------
# Aggregator / entry point
# -----------------------------


def run_research_intelligence(
    papers: Sequence[Paper],
    competitor_entities: Optional[List[Dict[str, Any]]] = None,
    competitor_topics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Public entry point used by app.py.

    Returns:
    {
        "statistics": {...},
        "papers": [...],          # recent + reviews first
        "concepts": [...],
        "data_points": [...],
        "information_gain": [...]
    }
    """
    papers = list(papers or [])
    if not papers:
        return {}
    concept_map = _extract_concepts(papers)
    concepts = build_research_concepts(concept_map)
    return {
        "statistics": build_research_statistics(papers),
        "papers": build_papers(papers),
        "concepts": concepts,
        "data_points": build_data_points(papers),
        "information_gain": build_information_gain(
            concepts,
            competitor_entities,
            competitor_topics,
        ),
    }