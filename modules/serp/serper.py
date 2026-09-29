import requests


def get_serp(
    keyword: str,
    serper_key: str,
    country: str,
    language: str,
    num_results: int = 10
):
    """
    Returns (organic_results, serp_summary, raw_response).

    raw_response is the full, unmodified Serper JSON response -- kept
    around specifically so the caller can inspect raw_response.get(
    "searchParameters") to confirm exactly what q/gl/hl/num Serper
    actually received and used for this call, without needing a separate
    curl/Playground test to check. On error, raw_response is {} so the
    3-tuple contract is always honored.
    """
    url = "https://google.serper.dev/search"

    payload = {
        "q": keyword,
        "gl": country,
        "hl": language,
        "num": num_results
    }

    headers = {
        "X-API-KEY": serper_key,
        "Content-Type": "application/json"
    }

    try:

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        organic_results = data.get(
            "organic",
            []
        )

        serp_summary = ""

        for result in organic_results:

            title = result.get(
                "title",
                ""
            )

            snippet = result.get(
                "snippet",
                ""
            )

            link = result.get(
                "link",
                ""
            )

            serp_summary += f"""
Title: {title}
Snippet: {snippet}
URL: {link}

"""

        return (
            organic_results,
            serp_summary,
            data,
        )

    except Exception as e:

        return (
            [],
            f"SERP Error: {str(e)}",
            {},
        )