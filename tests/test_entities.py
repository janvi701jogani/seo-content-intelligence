from modules.entities.extractor import aggregate_entities
import json

with open(
    "workspace/projects/coffee-guide/data/competitors.json",
    encoding="utf-8"
) as f:

    competitors = json.load(f)

entities = aggregate_entities(
    competitors
)

for entity in entities[:30]:

    print(entity)