# Diagnose why PAA is empty. Run: .venv\Scripts\python.exe test_paa.py

import json
import requests

# ---- Fill these in ----
SERPER_KEY = "ddd2d72e014dd1709e85db661d27e57508f6d410"
KEYWORD = "androgenetic alopecia treatment"
GL = "in"
HL = "en"
# -----------------------

resp = requests.post(
    "https://google.serper.dev/search",
    headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
    json={"q": KEYWORD, "gl": GL, "hl": HL},
    timeout=30,
)
print("HTTP status:", resp.status_code)
data = resp.json()

print("\nTop-level keys:")
for k in data.keys():
    v = data[k]
    n = len(v) if isinstance(v, (list, dict)) else ""
    print(f"  {k} ({type(v).__name__}{f', {n}' if n != '' else ''})")

paa = data.get("peopleAlsoAsk", [])
print(f"\npeopleAlsoAsk count: {len(paa)}")
for item in paa[:10]:
    print("  -", item.get("question"))

related = data.get("relatedSearches", [])
print(f"\nrelatedSearches count: {len(related)}")

print("\nRaw peopleAlsoAsk JSON:")
print(json.dumps(paa, indent=2)[:1500])