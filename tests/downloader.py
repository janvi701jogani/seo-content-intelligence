from modules.scraping.downloader import CompetitorDownloader


downloader = CompetitorDownloader()

result = downloader.download(

    "https://en.wikipedia.org/wiki/Coffee"

)

print(result["success"])

print(result["status_code"])

print(result["response_time"])

print(len(result["html"]))