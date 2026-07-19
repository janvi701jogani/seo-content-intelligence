"""
Standalone check for the Search Intelligence pipeline. Run from the
project root:

    .venv\\Scripts\\python.exe test_search.py

Needs a Serper key (for PAA + related searches). Autosuggest uses Google
directly (no key). Competitor FAQ extraction is exercised with a small
inline fixture so it runs without a full scrape.
"""

# ---- Fill these in ----
SERPER_KEY = ""
KEYWORD = "how to invest in mutual funds"
COUNTRY = "United States"
LANGUAGE = "English"
# -----------------------

from modules.search.intelligence import run_search_intelligence

# Minimal competitor fixture: one with FAQ JSON-LD, one with a question
# heading in its parsed structure.
competitors = [
    {
        "html": """
        <script type="application/ld+json">
        {"@type":"FAQPage","mainEntity":[
          {"@type":"Question","name":"Are mutual funds safe?"},
          {"@type":"Question","name":"How much should I invest?"}
        ]}
        </script>
        """,
        "structure": {},
    },
    {
        "structure": {
            "headings": {"h2": ["Are mutual funds safe?", "Types of funds"]},
            "logical_headings": ["How do I start investing?"],
        }
    },
]

# Minimal community fixture (as run_community_intelligence would return).
community = {
    "questions": [
        {"question": "Are mutual funds actually safe for beginners?"},
        {"question": "How much money do I need to start?"},
    ]
}

print("=" * 60)
print("STAGE 1: run search intelligence")
print("=" * 60)
result = run_search_intelligence(
    keyword=KEYWORD,
    serper_key=SERPER_KEY,
    country=COUNTRY,
    language=LANGUAGE,
    competitors=competitors,
    community=community,
)

for layer in (
    "paa",
    "related_searches",
    "autosuggest",
    "faqs",
    "question_intelligence",
    "signal_matrix",
):
    rows = result.get(layer, [])
    print(f"\n{layer}: {len(rows)} rows")
    for row in rows[:5]:
        print("   ", row)

print()
print("ALL STAGES PASSED")