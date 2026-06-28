import requests


def scrape_competitor(
    url,
    serper_key
):
    """
    Scrapes a single competitor webpage using Serper Scrape API.

    Returns:
        dict
    """

    endpoint = "https://scrape.serper.dev"

    payload = {
        "url": url
    }

    headers = {
        "X-API-KEY": serper_key,
        "Content-Type": "application/json"
    }

    try:

        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=60
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:

        return {
            "success": False,
            "url": url,
            "error": str(e)
        }