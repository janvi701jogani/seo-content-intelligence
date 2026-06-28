"""
Downloader Module

Downloads raw HTML from competitor pages.

This module DOES NOT parse HTML.
It only returns the response.
"""

from datetime import datetime
from time import perf_counter

import requests
from fake_useragent import UserAgent


class CompetitorDownloader:

    def __init__(self):

        self.ua = UserAgent()

    def download(self, url: str) -> dict:

        headers = {
            "User-Agent": self.ua.random
        }

        start = perf_counter()

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=20
            )

            elapsed = round(
                perf_counter() - start,
                2
            )

            return {

                "success": True,

                "url": url,

                "status_code": response.status_code,

                "response_time": elapsed,

                "download_time": datetime.utcnow().isoformat(),

                "html": response.text

            }

        except Exception as e:

            return {

                "success": False,

                "url": url,

                "error": str(e)

            }