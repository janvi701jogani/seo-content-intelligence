"""
Standalone check for the Content Brief agent. Run from the project root:

    .venv\\Scripts\\python.exe test_brief.py

Uses a dummy OpenAI-style client that first issues a tool call, then
returns a final answer, so it runs with no key and no network. Verifies
tool dispatch, the agentic loop, and source indexing.
"""

import json

from modules.brief.content_brief import (
    run_content_brief,
    dispatch_tool,
    _build_digest,
)


# ---- data bundle fixtures (dataclass-like via dicts) ----
bundle = {
    "keyword": "how to choose a mutual fund",
    "topics": [{"topic": "expense ratios"}, {"topic": "fund types"}],
    "entities": [{"entity": "expense ratio", "type": "KEYPHRASE"},
                 {"entity": "vanguard", "type": "ORG"}],
    "search_intel": {
        "signal_matrix": [{"question": "Are index funds better?"}],
        "question_intelligence": [{"question": "How are funds taxed?"}],
    },
    "community": {
        "questions": [{"question": "How do I pick my first fund?"}],
        "pain_points": [{"pain_point": "high fees"}],
        "myths": [{"myth": "active always beats passive"}],
        "gaps": [{"topic": "exit loads"}],
    },
    "reddit_threads": [
        {"title": "Fund fees explained", "subreddit": "investing",
         "score": 120, "permalink": "https://reddit.com/r/investing/abc",
         "selftext": "Fees compound over decades.",
         "comments": [{"body": "Watch the expense ratio.", "score": 40},
                      {"body": "Low fee index funds win.", "score": 80}]},
    ],
    "research": {
        "information_gain": [{"concept": "fee drag on returns"}],
        "data_points": [{"finding": "1% fee cuts final balance ~20%"}],
    },
    "research_papers": [
        {"title": "Mutual fund fees and performance", "journal": "J Finance",
         "year": 2019, "citation_count": 210, "url": "https://doi.org/xxx",
         "abstract": "Higher fees predict lower net returns."},
    ],
    "competitors": [
        {"title": "Guide to funds", "url": "https://example.com/guide",
         "text": "A long guide about choosing funds and fees."},
    ],
    "strategy": {"stage": "MOFU", "stage_rationale": "Comparison intent."},
}


# ---- dummy OpenAI client: tool call once, then final answer ----
class _Fn:
    def __init__(self, name, args):
        self.name = name
        self.arguments = json.dumps(args)


class _ToolCall:
    def __init__(self, name, args):
        self.id = "call_1"
        self.function = _Fn(name, args)


class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message):
        self.message = message


class _Resp:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class _Completions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            # First: ask to read the reddit thread.
            return _Resp(_Msg(tool_calls=[_ToolCall("get_reddit_thread", {"index": 0})]))
        # Second: final markdown brief.
        return _Resp(_Msg(content="# Content Brief\n\n- Cover fee drag "
                                  "[source](https://reddit.com/r/investing/abc)"))


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class DummyClient:
    def __init__(self):
        self.chat = _Chat()


print("=" * 60)
print("Unit: digest references tools + indices")
print("=" * 60)
digest = _build_digest(bundle)
print(digest[:400], "...\n")
assert "get_competitor_content" in digest
assert "[0]" in digest
print("PASS")

print("=" * 60)
print("Unit: tool dispatch")
print("=" * 60)
out = json.loads(dispatch_tool(bundle, "get_reddit_thread", {"index": 0}))
print(out)
assert out["permalink"].startswith("https://reddit.com")
assert out["top_comments"][0] == "Low fee index funds win."  # highest score first
bad = json.loads(dispatch_tool(bundle, "get_paper", {"index": 99}))
assert "error" in bad
print("PASS")

print("=" * 60)
print("Integration: agent loop (tool call -> final answer)")
print("=" * 60)
result = run_content_brief(DummyClient(), bundle["keyword"], bundle)
print("iterations:", result.get("iterations"))
print(result["brief"])
assert result["brief"].startswith("# Content Brief")
assert result["iterations"] == 2
assert len(result["sources"]["competitors"]) == 1
print("PASS")

print("\nALL TESTS PASSED")