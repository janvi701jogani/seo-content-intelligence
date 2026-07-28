"""
Content Brief agent.

Self-contained. Uses the OpenAI client from the sidebar key to research
EVERY tab and produce a skeletal content structure (a brief), not prose.

Design: tool-calling agent. The model gets a compact digest of all tabs
up front, plus tools to pull RAW data on demand (full competitor text,
Reddit threads with comments, paper abstracts, and any community/search
layer). It researches first, verifies unclear points against raw data,
then writes the brief. This is what "go back, check, gain clarity" needs
-- a flat dump can't do that and would blow the context window.

The brief:
- Sets focus by funnel stage inferred from the insights (a MOFU/BOFU
  topic must not be padded with TOFU basics).
- Gives short bullet suggestions (1-2 lines), never long prose -- the
  writer writes.
- Links every source it references (competitor URL, Reddit permalink,
  paper URL) inline.
- Weaves in Reddit insights, research evidence + citable data points,
  competitor gaps, new angles, and FAQs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

BRIEF_MODEL = "gpt-4o-mini"
MAX_TOOL_ITERATIONS = 16
COMPETITOR_CHARS = 12000
REDDIT_SELFTEXT_CHARS = 1500
REDDIT_COMMENT_CHARS = 400
REDDIT_MAX_COMMENTS = 15
DIGEST_TOP_N = 15

SYSTEM_PROMPT = (
    "You are a senior SEO content strategist building an EXHAUSTIVE content "
    "brief (an outline skeleton) for a writer. You have tools to read the "
    "raw research behind the summary you are given.\n\n"
    "NON-NEGOTIABLE DEPTH RULE: your outline must be at least as detailed "
    "as the single most detailed competitor, then go further. A lone "
    "competitor page must never cover more than your brief. Before "
    "writing, call get_competitor_structure AND get_competitor_content on "
    "the top competitors (at least 3-5), read the most relevant Reddit "
    "threads, and open key papers. Enumerate every sub-topic any strong "
    "competitor covers; your structure is a SUPERSET of all of them, plus "
    "the gaps surfaced by Reddit and research. Miss nothing.\n\n"
    "SYNTHESIS MANDATE (READ FIRST): ground the brief in every available "
    "source, then IMPROVE on it. Strike a balance and avoid both failure "
    "modes - do NOT merely mirror the competitors, and do NOT ignore the "
    "data and write from your own knowledge alone.\n"
    "- CONSIDER ALL SOURCES before writing: competitor section outlines "
    "(get_competitor_structure), Topic Intelligence, entities, Reddit, "
    "research, and search (PAA / autosuggest / related). Reviewing them is "
    "mandatory; how you use them is your judgement.\n"
    "- Treat competitor coverage as a MINIMUM completeness bar - cover the "
    "table-stakes topics they all cover so nothing essential is missed - "
    "but NOT as a template or a ceiling. You own the final structure.\n"
    "- Then exercise editorial judgement: reorganise, merge, reorder and "
    "rename sections into a more logical, comprehensive flow than any "
    "single competitor, and ADD sections/FAQs that Reddit, research, "
    "search, or your own expertise justify. Mark additions '(added)'.\n"
    "- The Content Recommendations list and your own ideas are welcome "
    "additions, but they supplement the grounded structure; they do not "
    "replace looking at the sources.\n"
    "- If a signal is disabled (e.g. entities or topics), still ground in "
    "the remaining sources; never collapse to competitors-only or to "
    "your-ideas-only.\n\n"
    "PROCESS:\n"
    "1. RESEARCH FIRST. Inspect competitor outlines and pages, Reddit "
    "threads, and papers. Do not invent facts; if unsure, open the raw "
    "source and check.\n"
    "2. SET FOCUS BY STAGE. Infer the search intent / funnel stage from "
    "the insights. Weight the structure to that stage: a MOFU/BOFU topic "
    "must NOT be padded with top-of-funnel basics; a TOFU topic stays "
    "educational. State the chosen focus at the top in 1-2 lines.\n"
    "3. FULL HEADING DEPTH (MANDATORY). Every substantive H2 MUST be broken "
    "into H3 sub-sections, and H4 where warranted. An outline that stops at "
    "H2 is unacceptable - revise it before returning. If a topic is a "
    "process, present the steps as ordered, numbered sub-sections.\n"
    "4. RECOMMEND ASSETS per section where useful: comparison TABLES (name "
    "the columns), IMAGES / DIAGRAMS / screenshots (say what each should "
    "show), CHECKLISTS, worked EXAMPLES or mini case snapshots, and "
    "callouts / warning boxes. State the asset and its purpose only, not "
    "its full contents.\n"
    "5. STRUCTURE, DON'T WRITE. Under each heading give SHORT bullets (max "
    "1-2 lines each): what to include, the angle, the asset/format, and "
    "source links. Never write paragraphs of prose. The writer writes.\n"
    "6. COVER THE FULL LIFECYCLE of the topic where relevant: definition, "
    "prerequisites, step-by-step process in order, tools/resources, costs "
    "and timelines, decision criteria and comparisons, common mistakes and "
    "pitfalls, edge cases, and FAQs.\n"
    "7. INTEGRATE ALL SOURCES. Weave in Reddit pain points, questions, "
    "myths and real experiences; research findings and specific citable "
    "data points; competitor coverage gaps; new angles; and an FAQ section "
    "aligned to People-Also-Ask and community questions.\n"
    "7a. TOPICS DEFINE STRUCTURE. Treat Topic Intelligence as the primary "
    "blueprint. Topics determine WHAT sections exist. High-importance "
    "topics with strong competitor coverage should normally be H2s; "
    "medium-importance topics become H3s; low-importance topics become "
    "FAQs or supporting bullets. Do not ignore a topic because competitors "
    "phrase it differently - semantic similarity matters more than exact "
    "wording.\n"
    "7ab. FALLBACK WHEN SIGNALS ARE OFF. If Topic Intelligence is empty "
    "(topic generation was skipped), build the structure from the "
    "competitor outlines (get_competitor_structure) and the SERP. If "
    "Entity Intelligence is empty (entity extraction was skipped), rely on "
    "the topics and competitor content to decide what to cover. Never "
    "block on a missing signal.\n"
    "7aa. ENTITIES ENRICH CONTENT. Entity Intelligence identifies the "
    "concepts, terminology, products, mechanisms, measurements and "
    "technologies that belong inside the article. Entities do NOT normally "
    "become headings. Instead: place each entity into the most relevant "
    "section; ensure the important entities appear somewhere in the "
    "outline; if several important entities belong to one topic, cover "
    "them together; never create an unnecessary heading just because an "
    "entity exists. Topics make the structure; entities make sections "
    "comprehensive.\n"
    "7b. RESEARCH WHERE IT ADDS VALUE. Check list_papers / get_paper and "
    "the research data points. Where a finding strengthens a section, add "
    "an 'Evidence to cite' bullet that states the finding AND includes the "
    "actual paper URL from the provided research. NEVER cite a bare "
    "publisher or authority name (e.g. 'Healthline', 'Harvard', 'NCCIH') "
    "with no link - that is not allowed. Prefer the provided papers and "
    "reuse the ones listed in the Evidence Summary. If you must use outside "
    "knowledge, follow rule 12. If literature is thin, say so once instead "
    "of padding.\n"
    "7c. RESEARCH CONFLICTS. If studies disagree on a claim, add a short "
    "'Research conflict' note under that section giving both sides and a "
    "cautious recommendation. This is required for YMYL topics (health, "
    "finance, nutrition, safety).\n"
    "7d. COMPETITOR COVERAGE IS SEMANTIC, NOT EXACT. get_topics counts are "
    "based on exact section clustering and UNDER-count real coverage (e.g. "
    "'Reduce Stress', 'Improve Mood', 'Emotional Wellbeing' are all the "
    "Mental Health topic but may appear as separate low-coverage rows). "
    "Judge coverage by reading competitor outlines (get_competitor_"
    "structure across competitors) and counting any competitor whose "
    "content addresses the concept under ANY wording. Use get_topics only "
    "as a hint, and prefer your semantic count.\n"
    "7e. FLOW SEARCH MODIFIERS INTO THE OUTLINE. Mine autosuggest and "
    "related searches for modifier variants (e.g. 'for women', 'for men', "
    "'for beginners', 'every day', 'at home', 'for weight loss'). Turn the "
    "relevant ones into H3 sub-sections or FAQ entries. Do not collect them "
    "and then ignore them.\n"
    "7f. FOLD IN CONTENT RECOMMENDATIONS AS ADDITIONS. The digest lists "
    "funnel-stage Content Recommendations (MOFU/BOFU ideas and suggested "
    "FAQs). These are ADDITIONS layered on top of the competitor+topic "
    "base (step c of the synthesis mandate), never the base itself. Fold "
    "each relevant one in as an H2/H3 or FAQ, marked '(added)'. They must "
    "sit alongside the competitor/topic/Reddit/research sections, not "
    "replace them.\n"
    "8. LINK SELECTIVELY, NOT EVERYWHERE. The outline must read as plain, "
    "writeable content that stands on its own. Attach a source link or "
    "resource nudge ONLY where it materially helps the writer: a specific "
    "statistic, a real user quote or experience, a contested claim to "
    "verify, or a gap a competitor covers well. Generic bullets get no "
    "link. End with a 'Sources' section listing everything referenced.\n"
    "9. PRIORITISE INFORMATION GAIN. Lead sections with what competitors "
    "miss but users discuss or research supports. Note target depth "
    "relative to competitor averages.\n"
    "10. EEAT + INTERNAL LINKS. Flag where to cite research or real user "
    "experience, and where to internally link to the next funnel stage.\n"
    "11. Keep the article educational and brand-neutral; reference "
    "competitors only as source links for the writer, not names to "
    "promote.\n"
    "12. EXTERNAL EVIDENCE FALLBACK. If a claim needs support and no "
    "provided paper covers it, you MAY cite a well-established external "
    "study or authoritative source from your own knowledge and add its "
    "link, but ONLY if you are confident it genuinely exists. NEVER "
    "fabricate titles, authors, journals, DOIs, or URLs. A bare source "
    "name with no working URL is NOT allowed - either give the real link "
    "or write 'evidence limited'. Mark any source not from the provided "
    "research as '(external - verify)'.\n\n"
    "REQUIRED OUTPUT STRUCTURE (produce these sections in order):\n"
    "A. SEARCH INTELLIGENCE (before the outline): Primary intent (1 line); "
    "Secondary intents (bullets); counts: PAA, Related Searches, "
    "Autosuggest, FAQs found; and KEY MODIFIERS - the notable variants from "
    "autosuggest/related (e.g. for women, for beginners, every day, at "
    "home) that will become sub-sections or FAQs.\n"
    "B. FOCUS & FUNNEL STAGE: the stage and how it weights the article "
    "(1-2 lines).\n"
    "C. PRIORITISATION - DON'T MISS: three short lists. MUST INCLUDE (table "
    "stakes: high search demand AND/OR high competitor coverage AND/OR "
    "strong evidence); NICE TO HAVE (opportunities and niche demographics/"
    "modifiers with real but smaller demand); CAN SKIP (low demand, low "
    "relevance, e.g. history, schools/lineages) - so the writer knows what "
    "to prioritise and what to drop.\n"
    "D. INFORMATION GAIN OPPORTUNITIES: bullet list of differentiators - "
    "angles competitors miss but users discuss or research supports "
    "(from content gaps, research information-gain, low-coverage topics, "
    "unique Reddit angles).\n"
    "E. COMMUNITY INSIGHTS (synthesised, not raw links): Most-asked "
    "questions; Most-reported and unexpected benefits; Common complaints; "
    "Misconceptions; Advice from experienced practitioners; Common beginner "
    "mistakes - distilled from the community layers (questions, features, "
    "experiences, recommendations, mistakes, myths, pain_points).\n"
    "F. EVIDENCE SUMMARY: for each key study, one line each - Finding, "
    "Strength (meta-analysis/review > single study), and the section it "
    "applies to. So the writer need not open every paper.\n"
    "G. THE OUTLINE: H1 > H2 > H3 > H4. Incorporate the search modifiers "
    "from section A as H3 sub-sections or FAQs. For each H2 section add:\n"
    "   - Annotation with star ratings (1-5): Competitor coverage (n/N, "
    "counted SEMANTICALLY per rule 7d), Evidence confidence, Search demand, "
    "Community interest, and a Recommendation (Essential / Recommended / "
    "Optional). High demand + low competitor coverage = flag as "
    "Opportunity.\n"
    "   - Reason: one line on why the section is included at this priority "
    "(e.g. high community interest, low competitor coverage, strong "
    "evidence).\n"
    "   - Section Inputs (list explicitly): 'Topics driving this section' "
    "(from Topic Intelligence); 'Important entities to cover' (from Entity "
    "Intelligence); 'Community signals' (relevant Reddit points); 'Research "
    "signals' (relevant studies/data). This shows why the section exists "
    "and what must appear in it.\n"
    "   - Coverage stars ONLY when the section maps to a real topic; for "
    "structural sections (Introduction, FAQs, Sources) omit coverage or "
    "mark n/a.\n"
    "   - Then the bullets (what to cover, assets, selective links, "
    "evidence to cite, research conflict if any).\n"
    "H. FAQs: align to PAA, the autosuggest question-modifiers, and "
    "community 'most-asked questions'.\n"
    "I. SOURCES (MANDATORY - never omit): every competitor, Reddit, and "
    "paper URL referenced anywhere above, as clickable markdown links.\n\n"
    "STAR RULES: rate 1-5 stars, be conservative. Evidence confidence 5 = "
    "multiple reviews/meta-analyses or many consistent papers; 1-2 = little "
    "or weak literature (do not overstate). Competitor coverage counted "
    "semantically across competitor outlines (rule 7d), not from raw "
    "get_topics rows. Search demand from how strongly the topic appears "
    "across PAA/related/autosuggest/signal matrix. Community interest from "
    "Reddit question/pain-point/experience volume.\n"
    "Return the brief as clean, deeply-nested Markdown."
)


# ---------------------------------------------------------------------------
# Value access that works for both dicts and dataclass objects
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _truncate(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


# ---------------------------------------------------------------------------
# Tool backends (over the data bundle)
# ---------------------------------------------------------------------------


def _tool_list_competitors(bundle: Dict[str, Any]) -> Any:
    return [
        {
            "index": index,
            "title": _get(c, "title"),
            "url": _get(c, "url"),
        }
        for index, c in enumerate(bundle.get("competitors", []) or [])
    ]


def _tool_get_competitor_content(bundle: Dict[str, Any], index: int) -> Any:
    competitors = bundle.get("competitors", []) or []
    if index < 0 or index >= len(competitors):
        return {"error": "index out of range"}
    c = competitors[index]
    return {
        "title": _get(c, "title"),
        "url": _get(c, "url"),
        "text": _truncate(_get(c, "text"), COMPETITOR_CHARS),
    }


def _tool_get_competitor_structure(bundle: Dict[str, Any], index: int) -> Any:
    competitors = bundle.get("competitors", []) or []
    if index < 0 or index >= len(competitors):
        return {"error": "index out of range"}
    c = competitors[index]
    structure = _get(c, "structure", {}) or {}
    headings = structure.get("headings", {}) if isinstance(structure, dict) else {}
    return {
        "title": _get(c, "title"),
        "url": _get(c, "url"),
        "headings": {level: values for level, values in headings.items() if values},
        "logical_headings": structure.get("logical_headings", []) if isinstance(structure, dict) else [],
        "table_count": len(structure.get("tables", []) if isinstance(structure, dict) else []),
        "list_count": len(structure.get("lists", []) if isinstance(structure, dict) else []),
        "image_count": len(structure.get("images", []) if isinstance(structure, dict) else []),
    }


def _tool_list_reddit_threads(bundle: Dict[str, Any]) -> Any:
    return [
        {
            "index": index,
            "title": _get(t, "title"),
            "subreddit": _get(t, "subreddit"),
            "score": _get(t, "score", 0),
            "permalink": _get(t, "permalink"),
        }
        for index, t in enumerate(bundle.get("reddit_threads", []) or [])
    ]


def _tool_get_reddit_thread(bundle: Dict[str, Any], index: int) -> Any:
    threads = bundle.get("reddit_threads", []) or []
    if index < 0 or index >= len(threads):
        return {"error": "index out of range"}
    thread = threads[index]
    comments = list(_get(thread, "comments", []) or [])
    comments = sorted(
        comments, key=lambda c: _get(c, "score", 0), reverse=True
    )[:REDDIT_MAX_COMMENTS]
    return {
        "title": _get(thread, "title"),
        "subreddit": _get(thread, "subreddit"),
        "permalink": _get(thread, "permalink"),
        "selftext": _truncate(_get(thread, "selftext"), REDDIT_SELFTEXT_CHARS),
        "top_comments": [
            _truncate(_get(c, "body"), REDDIT_COMMENT_CHARS) for c in comments
        ],
    }


def _tool_list_papers(bundle: Dict[str, Any]) -> Any:
    return [
        {
            "index": index,
            "title": _get(p, "title"),
            "journal": _get(p, "journal"),
            "year": _get(p, "year"),
            "citations": _get(p, "citation_count", 0),
            "url": _get(p, "url"),
        }
        for index, p in enumerate(bundle.get("research_papers", []) or [])
    ]


def _tool_get_paper(bundle: Dict[str, Any], index: int) -> Any:
    papers = bundle.get("research_papers", []) or []
    if index < 0 or index >= len(papers):
        return {"error": "index out of range"}
    p = papers[index]
    return {
        "title": _get(p, "title"),
        "journal": _get(p, "journal"),
        "year": _get(p, "year"),
        "url": _get(p, "url"),
        "abstract": _get(p, "abstract"),
    }


def _tool_get_topics(bundle: Dict[str, Any]) -> Any:
    total = len(bundle.get("competitors", []) or [])
    rows = []
    for t in bundle.get("topics", []) or []:
        if not isinstance(t, dict):
            continue
        rows.append({
            "topic": t.get("topic"),
            "competitors_using": t.get("competitors_using"),
            "total_competitors": total,
            "coverage_percent": t.get("coverage"),
            "importance": t.get("importance"),
        })
    return rows


def _tool_get_entities(bundle: Dict[str, Any]) -> Any:
    rows = []
    for e in bundle.get("entities", []) or []:
        if isinstance(e, dict) and e.get("entity"):
            rows.append({
                "entity": e.get("entity"),
                "type": e.get("type"),
                "importance": e.get("importance"),
                "competitors_using": e.get("competitors_using"),
            })
    return rows


def _tool_get_community_layer(bundle: Dict[str, Any], layer: str) -> Any:
    return (bundle.get("community", {}) or {}).get(layer, [])


def _tool_get_search_layer(bundle: Dict[str, Any], layer: str) -> Any:
    return (bundle.get("search_intel", {}) or {}).get(layer, [])


COMMUNITY_LAYERS = [
    "statistics", "questions", "pain_points", "recommendations", "brands",
    "features", "decision_factors", "vocabulary", "mistakes", "myths",
    "experiences", "gaps",
]
SEARCH_LAYERS = [
    "paa", "related_searches", "autosuggest", "faqs",
    "question_intelligence", "signal_matrix",
]

TOOLS = [
    {"type": "function", "function": {
        "name": "list_competitors",
        "description": "List ranking competitors with index, title, url.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_competitor_content",
        "description": "Full scraped text of one competitor page by index.",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer"}}, "required": ["index"]},
    }},
    {"type": "function", "function": {
        "name": "get_competitor_structure",
        "description": "One competitor's heading outline (H1-H6), plus table/"
                       "list/image counts. Use to build a superset outline.",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer"}}, "required": ["index"]},
    }},
    {"type": "function", "function": {
        "name": "list_reddit_threads",
        "description": "List Reddit threads with index, title, subreddit, permalink.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_reddit_thread",
        "description": "One Reddit thread's selftext and top comments by index.",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer"}}, "required": ["index"]},
    }},
    {"type": "function", "function": {
        "name": "list_papers",
        "description": "List research papers with index, title, journal, year, url.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_paper",
        "description": "One paper's abstract and metadata by index.",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer"}}, "required": ["index"]},
    }},
    {"type": "function", "function": {
        "name": "get_topics",
        "description": "All competitor topics with per-topic competitor "
                       "coverage (competitors_using / total) and importance. "
                       "Use for per-section competitor coverage.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_entities",
        "description": "All entities/concepts with type, importance, and "
                       "competitor coverage. Use to map relevant concepts "
                       "into each section.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_community_layer",
        "description": "Full rows of one Reddit/community layer.",
        "parameters": {"type": "object", "properties": {
            "layer": {"type": "string", "enum": COMMUNITY_LAYERS}},
            "required": ["layer"]},
    }},
    {"type": "function", "function": {
        "name": "get_search_layer",
        "description": "Full rows of one search-intelligence layer.",
        "parameters": {"type": "object", "properties": {
            "layer": {"type": "string", "enum": SEARCH_LAYERS}},
            "required": ["layer"]},
    }},
]

_DISPATCH = {
    "list_competitors": lambda b, a: _tool_list_competitors(b),
    "get_competitor_content": lambda b, a: _tool_get_competitor_content(b, a.get("index", -1)),
    "get_competitor_structure": lambda b, a: _tool_get_competitor_structure(b, a.get("index", -1)),
    "list_reddit_threads": lambda b, a: _tool_list_reddit_threads(b),
    "get_reddit_thread": lambda b, a: _tool_get_reddit_thread(b, a.get("index", -1)),
    "list_papers": lambda b, a: _tool_list_papers(b),
    "get_paper": lambda b, a: _tool_get_paper(b, a.get("index", -1)),
    "get_topics": lambda b, a: _tool_get_topics(b),
    "get_entities": lambda b, a: _tool_get_entities(b),
    "get_community_layer": lambda b, a: _tool_get_community_layer(b, a.get("layer", "")),
    "get_search_layer": lambda b, a: _tool_get_search_layer(b, a.get("layer", "")),
}


def dispatch_tool(bundle: Dict[str, Any], name: str, arguments: Dict[str, Any]) -> str:
    handler = _DISPATCH.get(name)
    if handler is None:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        return json.dumps(handler(bundle, arguments or {}), ensure_ascii=False)
    except Exception as error:
        return json.dumps({"error": str(error)})


# ---------------------------------------------------------------------------
# Digest + source index
# ---------------------------------------------------------------------------


def _rows(items: Any, key: str, limit: int = DIGEST_TOP_N) -> List[str]:
    out = []
    for item in (items or [])[:limit]:
        value = item.get(key) if isinstance(item, dict) else None
        if value:
            out.append(str(value))
    return out


def _build_digest(bundle: Dict[str, Any]) -> str:
    keyword = bundle.get("keyword", "")
    strategy = bundle.get("strategy", {}) or {}
    community = bundle.get("community", {}) or {}
    search_intel = bundle.get("search_intel", {}) or {}
    research = bundle.get("research", {}) or {}
    topics = bundle.get("topics", []) or []
    entities = bundle.get("entities", []) or []

    lines = [f"KEYWORD: {keyword}"]

    if strategy.get("stage"):
        lines.append(f"FUNNEL STAGE (heuristic): {strategy['stage']} - "
                     f"{strategy.get('stage_rationale','')}")

    recs = strategy.get("recommendations", {}) or {}
    if recs:
        lines.append("\nCONTENT RECOMMENDATIONS (fold each relevant one into "
                     "the outline as an H2/H3 section or an FAQ):")
        for stage_key in ("MOFU", "BOFU"):
            titles = [
                i.get("title") for i in (recs.get(stage_key, []) or [])
                if isinstance(i, dict) and i.get("title")
            ]
            if titles:
                lines.append(f"  {stage_key}: " + "; ".join(titles))
        faq_qs = [
            f.get("question") for f in (recs.get("FAQs", []) or [])
            if isinstance(f, dict) and f.get("question")
        ]
        if faq_qs:
            lines.append("  Suggested FAQs: " + "; ".join(faq_qs))

    comp = _tool_list_competitors(bundle)
    lines.append("\nCOMPETITORS (use get_competitor_structure for outline, "
                 "get_competitor_content for full text):")
    for c in comp[:DIGEST_TOP_N]:
        lines.append(f"  [{c['index']}] {c['title']} - {c['url']}")

    total = len(comp)
    lines.append(f"TOTAL COMPETITORS: {total}")

    # Topic Intelligence is core context: the structural blueprint.
    lines.append("\nTOPIC INTELLIGENCE (the blueprint for the structure; "
                 "get_topics for all):")
    if topics:
        for t in topics[:30]:
            if not isinstance(t, dict):
                continue
            lines.append(
                "- {topic} | Coverage: {used}/{total} competitors | "
                "Importance: {importance}".format(
                    topic=t.get("topic", ""),
                    used=t.get("competitors_using", 0),
                    total=total,
                    importance=t.get("importance", ""),
                )
            )
        lines.append(
            "These topics represent semantic coverage across competitors "
            "and should drive the document structure. High-coverage/high-"
            "importance topics usually become H2s; medium ones become H3s "
            "or FAQs depending on search intent."
        )
    else:
        lines.append(
            "(No topic generation this run. Derive the structure from the "
            "competitor outlines via get_competitor_structure and the SERP "
            "results.)"
        )

    # Entity Intelligence is core context: the concepts inside sections.
    lines.append("\nENTITY INTELLIGENCE (concepts to place inside sections; "
                 "get_entities for all):")
    sorted_entities = sorted(
        [e for e in (entities or []) if isinstance(e, dict)],
        key=lambda x: x.get("importance") or 0,
        reverse=True,
    )
    if sorted_entities:
        for e in sorted_entities[:50]:
            lines.append(
                "- {entity} ({etype}) | Importance: {importance} | "
                "Competitors: {used}".format(
                    entity=e.get("entity", ""),
                    etype=e.get("type", ""),
                    importance=e.get("importance", ""),
                    used=e.get("competitors_using", ""),
                )
            )
        lines.append(
            "These entities are important concepts that should be naturally "
            "covered within the relevant sections. They do NOT create new "
            "headings by themselves unless they represent a major topic."
        )
    else:
        lines.append(
            "(No entity extraction this run. Rely on the topics and the "
            "competitor content to decide what concepts to cover.)"
        )

    lines.append("\nSEARCH INTELLIGENCE (get_search_layer for full):")
    lines.append(
        "  COUNTS: PAA=%d, Related=%d, Autosuggest=%d, FAQs=%d" % (
            len(search_intel.get("paa", []) or []),
            len(search_intel.get("related_searches", []) or []),
            len(search_intel.get("autosuggest", []) or []),
            len(search_intel.get("faqs", []) or []),
        )
    )
    lines.append("  Signal matrix (top): " + "; ".join(
        _rows(search_intel.get("signal_matrix"), "question")))
    lines.append("  Questions: " + "; ".join(
        _rows(search_intel.get("question_intelligence"), "question")))
    lines.append("  Related searches: " + "; ".join(
        _rows(search_intel.get("related_searches"), "query")))
    lines.append("  Autosuggest modifiers: " + "; ".join(
        _rows(search_intel.get("autosuggest"), "suggestion", 25)))

    lines.append("\nCOMMUNITY (get_community_layer / get_reddit_thread for full):")
    lines.append("  Questions: " + "; ".join(_rows(community.get("questions"), "question")))
    lines.append("  Pain points: " + "; ".join(_rows(community.get("pain_points"), "pain_point")))
    lines.append("  Benefits/features: " + "; ".join(_rows(community.get("features"), "feature")))
    lines.append("  Experiences: " + "; ".join(_rows(community.get("experiences"), "experience")))
    lines.append("  Recommendations: " + "; ".join(_rows(community.get("recommendations"), "advice")))
    lines.append("  Myths: " + "; ".join(_rows(community.get("myths"), "myth")))
    lines.append("  Content gaps: " + "; ".join(_rows(community.get("gaps"), "topic")))

    threads = _tool_list_reddit_threads(bundle)
    lines.append("  Threads:")
    for t in threads[:DIGEST_TOP_N]:
        lines.append(f"    [{t['index']}] {t['title']} ({t['permalink']})")

    lines.append("\nRESEARCH (get_paper for abstracts):")
    ig_bits = []
    for r in (research.get("information_gain") or [])[:DIGEST_TOP_N]:
        if isinstance(r, dict) and r.get("concept"):
            ig_bits.append(
                f"{r['concept']} (papers={r.get('research_papers','?')}, "
                f"comp_cov={r.get('competitor_coverage','?')})"
            )
    lines.append("  Information gain: " + "; ".join(ig_bits))
    lines.append("  Data points: " + "; ".join(_rows(research.get("data_points"), "finding")))
    papers = _tool_list_papers(bundle)
    for p in papers[:DIGEST_TOP_N]:
        lines.append(f"    [{p['index']}] {p['title']} ({p.get('year')}) - {p['url']}")

    return "\n".join(lines)


def _source_index(bundle: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "competitors": _tool_list_competitors(bundle),
        "reddit_threads": _tool_list_reddit_threads(bundle),
        "papers": _tool_list_papers(bundle),
    }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_content_brief(
    client: Any,
    keyword: str,
    bundle: Dict[str, Any],
    model: str = BRIEF_MODEL,
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> Dict[str, Any]:
    """
    Public entry point used by app.py. Requires an OpenAI client.

    Returns {"brief": markdown_str, "sources": {...}, "iterations": n}
    or {} if the client is missing or the call fails.
    """
    if client is None or not keyword:
        return {}

    bundle = dict(bundle or {})
    bundle.setdefault("keyword", keyword)
    digest = _build_digest(bundle)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Here is a digest of every research tab.\n\n"
            "BEFORE writing the outline, follow this order:\n"
            "1. GROUND (mandatory to review): call get_competitor_structure "
            "on the top competitors and read Topic Intelligence to see what "
            "the strong pages cover. This sets the completeness floor - the "
            "table-stakes topics you must not miss - not a template to copy.\n"
            "2. ENRICH: map Entity Intelligence into sections (if present); "
            "layer in Reddit (get_reddit_thread), Research (list_papers / "
            "get_paper, data points), and Search (PAA, autosuggest, related "
            "-> H3s/FAQs).\n"
            "3. IMPROVE & ADD: design the best structure - reorganise, "
            "merge, reorder, rename - and add your own sections/FAQs plus "
            "the Content Recommendations to go beyond competitors. Mark "
            "additions '(added)'.\n"
            "4. ARRANGE into one logical, comprehensive flow with full "
            "H2>H3>H4 depth, process steps in order, and table/image/"
            "checklist assets. Add source links or 'Evidence to cite' "
            "nudges only where they materially help.\n"
            "Balance: exceed the competitors, but never miss what they "
            "cover, and never ignore the data to write from memory alone. "
            "If entities or topics are disabled, still ground in the "
            "remaining sources.\n\n"
            + digest
        )},
    ]

    try:
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                return {
                    "brief": message.content or "",
                    "sources": _source_index(bundle),
                    "iterations": iterations,
                }

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                result = dispatch_tool(bundle, tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # Iterations exhausted: force a final answer with no more tools.
        final = client.chat.completions.create(
            model=model,
            messages=messages + [{
                "role": "user",
                "content": "Stop researching and write the final brief now.",
            }],
            temperature=0.3,
        )
        return {
            "brief": final.choices[0].message.content or "",
            "sources": _source_index(bundle),
            "iterations": iterations,
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}