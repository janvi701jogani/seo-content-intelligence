"""
Standalone check for the Reddit pipeline. Run from the project root:

    .venv\\Scripts\\python.exe test_reddit.py

Tests each stage separately so a failure points to the exact stage:
1. Serper query (free-account-safe pattern) -> raw results
2. URL filtering -> Reddit thread URLs
3. praw fetch -> threads with comments
4. (optional) community intelligence layers
"""

# ---- Fill these in ----
SERPER_KEY = "ddd2d72e014dd1709e85db661d27e57508f6d410"
REDDIT_CLIENT_ID = "F4DN8b9iphaH6F4QmncX8Q"
REDDIT_CLIENT_SECRET = "lAPbLcqvxq3Pc_luy27CNRVHJSqOfA"
REDDIT_USER_AGENT = "SEO-Intelligence-Platform/1.0 by u/<janvi2>"
KEYWORD = "how to invest in mutual funds"
RUN_COMMUNITY_LAYERS = False  # True = also run the 12-layer pipeline (slow)
# -----------------------

import requests

from modules.community.reddit import (
    RedditCollector,
    run_community_intelligence,
)

print("=" * 60)
print("STAGE 1: Serper query")
print("=" * 60)
response = requests.post(
    "https://google.serper.dev/search",
    headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
    json={"q": f"{KEYWORD} reddit", "num": 30, "gl": "in", "hl": "en"},
    timeout=30,
)
print("Status:", response.status_code)
if response.status_code != 200:
    print("Body:", response.text[:300])
    raise SystemExit("Serper request failed. Fix this before continuing.")
organic = response.json().get("organic", [])
print(f"Organic results: {len(organic)}")
for result in organic[:5]:
    print("  -", result.get("link", ""))

print()
print("=" * 60)
print("STAGE 2: URL filtering")
print("=" * 60)
collector = RedditCollector(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)
urls = []
seen = set()
for result in organic:
    url = collector._normalize_reddit_url(result.get("link", ""))
    if url and url not in seen:
        seen.add(url)
        urls.append(url)
print(f"Reddit thread URLs after filtering: {len(urls)}")
for url in urls:
    print("  -", url)
if not urls:
    raise SystemExit(
        "No Reddit thread URLs in results. "
        "Try a different keyword or raise num."
    )

print()
print("=" * 60)
print("STAGE 3: praw fetch")
print("=" * 60)
threads = collector.search_via_google(
    keyword=KEYWORD,
    serper_key=SERPER_KEY,
    limit=5,
    country="in",
    language="en",
)
print(f"Threads fetched: {len(threads)}")
for thread in threads:
    print(
        f"  - r/{thread.subreddit} | score={thread.score} | "
        f"comments={len(thread.comments)} | {thread.title[:70]}"
    )
if not threads:
    raise SystemExit(
        "URLs were found but praw fetched nothing. "
        "Check Reddit credentials."
    )

if RUN_COMMUNITY_LAYERS:
    print()
    print("=" * 60)
    print("STAGE 4: community intelligence (this takes a while)")
    print("=" * 60)
    community = run_community_intelligence(threads)
    for layer, rows in community.items():
        if layer == "statistics":
            print(f"  statistics: {rows}")
        else:
            print(f"  {layer}: {len(rows)} rows")
            for row in rows[:3]:
                print("      ", row)

print()
print("ALL STAGES PASSED")