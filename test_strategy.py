"""
Standalone check for the Funnel Strategy module. Run from the project
root:

    .venv\\Scripts\\python.exe test_strategy.py

Uses a dummy OpenAI-style client so it runs with no key and no network,
exercising brand extraction, context building, JSON parsing, and the
brand-scrub safety net. Set USE_REAL_OPENAI + a key to hit the live API.
"""

import json

from modules.strategy.funnel import (
    run_funnel_strategy,
    _competitor_brands,
    _build_context,
)

# ---- fixtures ----
KEYWORD = "what is a mutual fund"
competitors = [
    {"url": "https://www.investopedia.com/x"},
    {"url": "https://www.nerdwallet.com/y"},
]
entities = [
    {"entity": "expense ratio", "type": "KEYPHRASE"},
    {"entity": "vanguard", "type": "ORG"},
]
community = {
    "questions": [{"question": "How do I pick my first fund?"}],
    "pain_points": [{"pain_point": "high fees eat returns"}],
}
search_intel = {
    "question_intelligence": [{"question": "How are mutual funds taxed?"}]
}


class _DummyMessage:
    def __init__(self, content):
        self.content = content


class _DummyChoice:
    def __init__(self, content):
        self.message = _DummyMessage(content)


class _DummyResponse:
    def __init__(self, content):
        self.choices = [_DummyChoice(content)]


class _DummyCompletions:
    def create(self, **kwargs):
        # Return one clean item and one that leaks an excluded brand, to
        # prove the scrub drops the leaking item.
        payload = {
            "stage": "TOFU",
            "stage_rationale": "Definitional query, top of funnel.",
            "recommendations": {
                "MOFU": [
                    {"title": "How to compare fund fees",
                     "angle": "Explain expense ratios and their impact",
                     "format": "guide"},
                    {"title": "Why Vanguard is best",
                     "angle": "brand comparison",
                     "format": "listicle"},
                ],
                "BOFU": [
                    {"title": "Checklist before buying your first fund",
                     "angle": "Decision checklist",
                     "format": "checklist"},
                ],
                "FAQs": [
                    {"question": "Are mutual funds taxed?",
                     "why": "Recurring audience question"},
                ],
            },
        }
        return _DummyResponse(json.dumps(payload))


class _DummyChat:
    def __init__(self):
        self.completions = _DummyCompletions()


class DummyClient:
    def __init__(self):
        self.chat = _DummyChat()


print("=" * 60)
print("Unit: brand extraction")
print("=" * 60)
brands = _competitor_brands(competitors, entities)
print("brands:", brands)
assert "investopedia" in brands
assert "nerdwallet" in brands
assert "vanguard" in brands  # from ORG entity
print("PASS")

print("\n" + "=" * 60)
print("Unit: context building")
print("=" * 60)
ctx = _build_context(KEYWORD, None, entities, community, search_intel)
print(ctx)
assert "expense ratio" in ctx           # concept included
assert "vanguard" not in ctx.lower()    # ORG excluded from concepts
assert "high fees eat returns" in ctx   # pain point included
print("PASS")

print("\n" + "=" * 60)
print("Integration: run with dummy client + scrub")
print("=" * 60)
result = run_funnel_strategy(
    DummyClient(),
    KEYWORD,
    entities=entities,
    community=community,
    search_intel=search_intel,
    competitors=competitors,
)
print(json.dumps(result, indent=2))
mofu = result["recommendations"]["MOFU"]
titles = [m["title"] for m in mofu]
assert "How to compare fund fees" in titles      # clean item kept
assert "Why Vanguard is best" not in titles      # brand-leaking item scrubbed
assert result["stage"] == "TOFU"
print("PASS: brand-leaking recommendation was scrubbed")

print("\nALL TESTS PASSED")