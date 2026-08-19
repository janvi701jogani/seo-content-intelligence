"""
Funnel Strategy module.

Self-contained. Uses the OpenAI client already created in app.py from the
sidebar key. Classifies the keyword's funnel stage and generates
downstream content-angle recommendations:

    TOFU keyword -> MOFU + BOFU content ideas + FAQs
    MOFU keyword -> BOFU content ideas + FAQs
    BOFU keyword -> FAQs + supporting educational angles

Recommendations are grounded in the signals already computed (topics,
questions, pain points) so they are specific, not generic. All output is
educational and brand-neutral: no brand, company, or product names, and
no competitor names (an exclusion list is passed to the model and any
leakage is scrubbed from the result).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

FUNNEL_MODEL = "gpt-4o-mini"
MAX_TOPICS = 15
MAX_QUESTIONS = 20
MAX_PAIN_POINTS = 10
MAX_CONCEPTS = 15
MAX_BRANDS = 50
ITEMS_PER_STAGE = 6

SYSTEM_PROMPT = (
    "You are an SEO content strategist. You map a keyword to a marketing "
    "funnel stage and recommend downstream content angles.\n\n"
    "Funnel stages:\n"
    "- TOFU (top): awareness / informational intent (what is, how does, "
    "basics, guides).\n"
    "- MOFU (middle): consideration intent (how to choose, comparisons, "
    "criteria, types, pros and cons).\n"
    "- BOFU (bottom): decision intent (best options, pricing, "
    "step-by-step to act, checklists to decide).\n\n"
    "Rules:\n"
    "1. First classify the given keyword's stage.\n"
    "2. Recommend content for the stages BELOW it in the funnel: a TOFU "
    "keyword gets MOFU and BOFU recommendations; a MOFU keyword gets BOFU "
    "recommendations. Always also give FAQs.\n"
    "3. Every recommendation must be EDUCATIONAL and brand-neutral.\n"
    "4. Do NOT mention or reference any brand, company, product, tool, or "
    "publication name. Never use any name from the exclusion list.\n"
    "5. Ground recommendations in the provided audience questions, topics, "
    "and pain points where relevant.\n"
    "6. Return STRICT JSON only, matching the requested schema."
)


def _is_topic_word(candidate: str, keyword_tokens: set) -> bool:
    """
    True if every word in `candidate` is also a word in the keyword itself.
    Guards against excluding the topic's own core word (e.g. a competitor
    domain like 'gold.org' would otherwise add 'gold' to the brand
    exclusion list for the keyword 'how to invest in gold', which then
    scrubs EVERY recommendation -- they all legitimately mention gold).
    """
    words = candidate.split()
    return bool(words) and all(w in keyword_tokens for w in words)


def _competitor_brands(
    competitors: Optional[Sequence[Dict[str, Any]]],
    entities: Optional[Sequence[Dict[str, Any]]],
    keyword: str = "",
) -> List[str]:
    keyword_tokens = set(re.findall(r"[a-z0-9]+", (keyword or "").lower()))
    brands: set = set()
    for competitor in competitors or []:
        url = competitor.get("url", "") or ""
        host = urlparse(url).netloc.lower().replace("www.", "")
        root = host.split(".")[0] if host else ""
        if len(root) >= 3 and not _is_topic_word(root, keyword_tokens):
            brands.add(root)
    for entity in entities or []:
        if entity.get("type") in ("ORG", "PRODUCT"):
            name = str(entity.get("entity", "")).strip().lower()
            if len(name) >= 3 and not _is_topic_word(name, keyword_tokens):
                brands.add(name)
    return sorted(brands)[:MAX_BRANDS]


def _build_context(
    keyword: str,
    topics: Optional[Sequence[Dict[str, Any]]],
    entities: Optional[Sequence[Dict[str, Any]]],
    community: Optional[Dict[str, Any]],
    search_intel: Optional[Dict[str, Any]],
) -> str:
    topic_names = [
        str(t.get("topic", "")).strip()
        for t in (topics or [])[:MAX_TOPICS]
        if t.get("topic")
    ]

    questions: List[str] = []
    for row in (search_intel or {}).get("question_intelligence", [])[:MAX_QUESTIONS]:
        if row.get("question"):
            questions.append(str(row["question"]).strip())
    for row in (community or {}).get("questions", [])[:MAX_QUESTIONS]:
        if row.get("question"):
            questions.append(str(row["question"]).strip())
    # Dedup, preserve order.
    seen: set = set()
    questions = [
        q for q in questions
        if not (q.lower() in seen or seen.add(q.lower()))
    ][:MAX_QUESTIONS]

    pain_points = [
        str(p.get("pain_point", "")).strip()
        for p in (community or {}).get("pain_points", [])[:MAX_PAIN_POINTS]
        if p.get("pain_point")
    ]

    concepts = [
        str(e.get("entity", "")).strip()
        for e in (entities or [])
        if e.get("type") not in ("ORG", "PRODUCT") and e.get("entity")
    ][:MAX_CONCEPTS]

    lines = [f"Keyword: {keyword}"]
    if topic_names:
        lines.append("Topics competitors cover: " + "; ".join(topic_names))
    if concepts:
        lines.append("Key concepts: " + "; ".join(concepts))
    if questions:
        lines.append("Audience questions: " + " | ".join(questions))
    if pain_points:
        lines.append("Audience pain points: " + "; ".join(pain_points))
    return "\n".join(lines)


def _user_prompt(context: str, exclusion: List[str]) -> str:
    schema = (
        '{\n'
        '  "stage": "TOFU|MOFU|BOFU",\n'
        '  "stage_rationale": "one sentence",\n'
        '  "recommendations": {\n'
        '    "MOFU": [{"title": "...", "angle": "...", "format": "..."}],\n'
        '    "BOFU": [{"title": "...", "angle": "...", "format": "..."}],\n'
        '    "FAQs": [{"question": "...", "why": "..."}]\n'
        '  }\n'
        '}'
    )
    return (
        f"{context}\n\n"
        f"Exclusion list (never mention any of these names): "
        f"{', '.join(exclusion) if exclusion else 'none'}\n\n"
        f"Give up to {ITEMS_PER_STAGE} items per applicable stage. Only "
        f"include stages below the classified stage (omit or leave empty a "
        f"stage that does not apply). Always include FAQs.\n\n"
        f"Return JSON exactly in this shape:\n{schema}"
    )


def _contains_brand(text: str, exclusion: List[str]) -> bool:
    lowered = text.lower()
    return any(
        re.search(r"\b" + re.escape(brand) + r"\b", lowered)
        for brand in exclusion
    )


def _scrub_recommendations(
    data: Dict[str, Any],
    exclusion: List[str],
) -> Dict[str, Any]:
    """
    Belt-and-suspenders: drop any recommendation/FAQ item that still
    references an excluded brand, in case the model ignored the rule.
    """
    if not exclusion:
        return data
    recommendations = data.get("recommendations", {}) or {}
    cleaned: Dict[str, List[Dict[str, Any]]] = {}
    for stage, items in recommendations.items():
        kept = []
        for item in items or []:
            blob = " ".join(str(v) for v in item.values())
            if not _contains_brand(blob, exclusion):
                kept.append(item)
        cleaned[stage] = kept
    data["recommendations"] = cleaned
    return data


def run_funnel_strategy(
    client: Any,
    keyword: str,
    topics: Optional[Sequence[Dict[str, Any]]] = None,
    entities: Optional[Sequence[Dict[str, Any]]] = None,
    community: Optional[Dict[str, Any]] = None,
    search_intel: Optional[Dict[str, Any]] = None,
    competitors: Optional[Sequence[Dict[str, Any]]] = None,
    model: str = FUNNEL_MODEL,
) -> Dict[str, Any]:
    """
    Public entry point used by app.py. Requires an OpenAI client.

    Returns:
    {
        "stage": "TOFU|MOFU|BOFU",
        "stage_rationale": "...",
        "recommendations": {
            "MOFU": [{"title","angle","format"}],
            "BOFU": [{"title","angle","format"}],
            "FAQs": [{"question","why"}]
        }
    }
    Returns {} if the client is missing or the call fails.
    """
    if client is None or not keyword:
        return {}

    exclusion = _competitor_brands(competitors, entities, keyword)
    context = _build_context(keyword, topics, entities, community, search_intel)
    user_prompt = _user_prompt(context, exclusion)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
    except Exception as error:
        # Surface the real failure (e.g. rate limit / quota) instead of a
        # silent {} that just looks like "no ideas generated" in the UI.
        return {"error": f"{type(error).__name__}: {error}"}

    if not isinstance(data, dict) or "recommendations" not in data:
        return {"error": "Model did not return the expected JSON shape."}

    return _scrub_recommendations(data, exclusion)