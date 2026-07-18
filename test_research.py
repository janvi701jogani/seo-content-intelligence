"""
Standalone check for the Research Intelligence pipeline. Run from the
project root:

    .venv\\Scripts\\python.exe test_research.py

Uses OpenAlex by default (no key). Semantic Scholar is used only if a key
is set below. Tests each stage separately so a failure points to the
exact stage.
"""

# ---- Fill these in ----
KEYWORD = "mutual fund investing"
SEMANTIC_SCHOLAR_KEY = ""   # optional; blank uses OpenAlex
CONTACT_EMAIL = ""          # optional; OpenAlex "polite pool"
RUN_INTELLIGENCE = True     # False = only fetch, skip processing
# -----------------------

from modules.research.literature import (
    ResearchCollector,
    run_research_intelligence,
)

print("=" * 60)
print("STAGE 1: literature fetch (OpenAlex / Semantic Scholar)")
print("=" * 60)
collector = ResearchCollector(
    api_key=SEMANTIC_SCHOLAR_KEY or None,
    contact_email=CONTACT_EMAIL or None,
)
papers = collector.search(keyword=KEYWORD, limit=25)
print(f"Papers fetched: {len(papers)}")
for paper in papers[:8]:
    tag = "REVIEW" if paper.is_review else "paper "
    print(
        f"  - {paper.year} | {tag} | cites={paper.citation_count} | "
        f"{paper.journal[:30]:30} | {paper.title[:50]}"
    )
if not papers:
    raise SystemExit(
        "No papers returned. Try a broader keyword, or you may be rate "
        "limited (retry in a minute, or add a Semantic Scholar key)."
    )

if RUN_INTELLIGENCE:
    print()
    print("=" * 60)
    print("STAGE 2: research intelligence")
    print("=" * 60)
    research = run_research_intelligence(papers)
    for layer, rows in research.items():
        if layer == "statistics":
            print(f"  statistics: {rows}")
        else:
            print(f"  {layer}: {len(rows)} rows")
            for row in rows[:3]:
                print("      ", row)

print()
print("ALL STAGES PASSED")