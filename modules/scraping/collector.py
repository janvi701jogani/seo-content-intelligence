import json
import re
from pathlib import Path

from modules.serp.serper import get_serp
from modules.scraping.competitor_scraper import scrape_competitor


# ---------------------------------------------------------------------------
# Boilerplate / promotional line removal.
#
# Text-only scrapes include ad teasers, "related articles" widgets, cookie
# notices, and affiliate CTAs. Left in, they become paragraphs that pollute
# Topic and Entity Intelligence (e.g. "Cancel Your Car Insurance",
# "Earn $1K/Month" showing up as topics for an unrelated query).
#
# The filter is deliberately HIGH PRECISION: a line is removed only if it
# matches an unambiguous nav/ad/legal/social pattern, OR it is a short line
# carrying a money/discount signal in mostly-title-case (a promotional CTA).
# Plain headings and prose have no money signal and are never touched, so
# real content is preserved. Structural, niche-agnostic.
# ---------------------------------------------------------------------------

BOILERPLATE_LINE_PATTERNS = [
    r"^\s*advertisement\b",
    r"^\s*sponsored\b",
    r"^\s*promoted\b",
    r"^\s*read more\b",
    r"^\s*related (articles?|posts?|reading|stories)\b",
    r"you (might|may) (also )?like",
    r"recommended for you",
    r"^\s*more from\b",
    r"^\s*trending\b",
    r"^\s*popular posts?\b",
    r"^\s*most read\b",
    r"^\s*up next\b",
    r"sign up for( our)? newsletter",
    r"subscribe to our",
    r"^\s*follow us\b",
    r"^\s*share (this|on)\b",
    r"we use cookies",
    r"accept (all )?cookies",
    r"cookie (policy|settings|preferences)",
    r"privacy policy",
    r"terms of (service|use)",
    r"all rights reserved",
]

MONEY_CTA_PATTERN = re.compile(
    r"(\$\s?\d|\d+\s?%|/month\b|/mo\b|\bper month\b|\bcash ?back\b|\bAPR\b|\bAPY\b)",
    re.IGNORECASE,
)


def _is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(re.search(pattern, lowered) for pattern in BOILERPLATE_LINE_PATTERNS):
        return True
    words = stripped.split()
    word_count = len(words)
    if 2 <= word_count <= 16 and MONEY_CTA_PATTERN.search(stripped):
        capitalized = sum(1 for word in words if word[:1].isupper())
        if capitalized / word_count >= 0.5:
            return True
    return False


def clean_scraped_text(text: str) -> str:
    """Drops boilerplate/promotional lines; preserves everything else."""
    if not text:
        return text
    kept = [
        line
        for line in text.splitlines()
        if not _is_boilerplate_line(line)
    ]
    return "\n".join(kept)


def collect_competitors(
    keyword,
    serper_key,
    project_name,
    country,
    language,
    num_results=10,
    organic_results=None
):
    # Single SERP fetch. Reuse the list the app already fetched (for the
    # SERP tab) instead of calling Serper a second time. This keeps the
    # SERP tab, Competitors, and Topics on the same real ranking set, and
    # saves one Serper credit per run. Only fetch here if the caller did
    # not pass results.
    if organic_results:
        results_list = organic_results
    else:
        results_list, _ = get_serp(
            keyword=keyword,
            serper_key=serper_key,
            country=country,
            language=language,
            num_results=num_results
        )

    # Dedupe by URL, drop empties, preserve ranking order.
    merged = []
    seen = set()
    for result in results_list or []:
        link = result.get("link", "")
        if not link or link in seen:
            continue
        seen.add(link)
        merged.append(result)

    competitors = []
    for position, result in enumerate(merged, start=1):
        print(f"Scraping {position}/{len(merged)}")
        scraped = scrape_competitor(
            result["link"],
            serper_key
        )
        competitors.append({
            "position": position,
            "title": result.get("title", ""),
            "url": result.get("link", ""),
            "snippet": result.get("snippet", ""),
            "text": clean_scraped_text(scraped.get("text", "")),
            "metadata": scraped.get("metadata", {}),
            "credits": scraped.get("credits", 0)
        })
    project_folder = Path(
        f"workspace/projects/{project_name}/data"
    )
    project_folder.mkdir(
        parents=True,
        exist_ok=True
    )
    with open(
        project_folder / "competitors.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            competitors,
            f,
            indent=4,
            ensure_ascii=False
        )
    return competitors