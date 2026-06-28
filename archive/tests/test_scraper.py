from modules.scraping.headings import extract_headings

url = "https://www.nerdwallet.com/article/investing/how-to-invest-in-mutual-funds"

headings = extract_headings(url)

print("\n========== H1 ==========\n")

for h in headings["h1s"]:
    print("-", h)

print("\n========== H2 ==========\n")

for h in headings["h2s"]:
    print("-", h)

print("\n========== H3 ==========\n")

for h in headings["h3s"]:
    print("-", h)