from modules.scraping.competitor_scraper import scrape_competitor

api = input("Serper API Key: ")

url = input("URL: ")

result = scrape_competitor(
    url,
    api
)

print(result.keys())