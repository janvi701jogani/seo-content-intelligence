import json
from pathlib import Path

from modules.serp.serper import get_serp
from modules.scraping.competitor_scraper import scrape_competitor


def collect_competitors(
    keyword,
    serper_key,
    project_name,
    country,
    language,
    num_results=10
):

    organic_results, _ = get_serp(
        keyword=keyword,
        serper_key=serper_key,
        country=country,
        language=language,
        num_results=num_results
    )

    competitors = []

    for position, result in enumerate(organic_results, start=1):

        print(f"Scraping {position}/{len(organic_results)}")

        scraped = scrape_competitor(
            result["link"],
            serper_key
        )

        competitors.append({

            "position": position,

            "title": result.get("title", ""),

            "url": result.get("link", ""),

            "snippet": result.get("snippet", ""),

            "text": scraped.get("text", ""),

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