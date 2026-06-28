"""
Entity Extraction Module

Extracts named entities from competitor content using spaCy.

Future versions will merge:
- spaCy
- GLiNER
- KeyBERT
- YAKE
"""

from collections import Counter
import spacy

# Load once
nlp = spacy.load("en_core_web_sm")


VALID_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "PRODUCT",
    "GPE",
    "LOC",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "FAC"
}


def extract_entities(text: str):
    """
    Extract entities from a block of text.

    Returns
    -------
    list
    """

    if not text:
        return []

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        if ent.label_ not in VALID_ENTITY_TYPES:
            continue

        value = ent.text.strip()

        if len(value) < 2:
            continue

        entities.append({

            "entity": value,

            "label": ent.label_,

            "start": ent.start_char,

            "end": ent.end_char

        })

    return entities


def aggregate_entities(competitors):
    """
    Aggregate entities across all competitors.

    Parameters
    ----------
    competitors : list

    Returns
    -------
    list
    """

    frequency = Counter()

    coverage = Counter()

    labels = {}

    mentions = {}

    for competitor in competitors:

        text = competitor.get(
            "text",
            ""
        )

        extracted = extract_entities(text)

        seen = set()

        for entity in extracted:

            name = entity["entity"].strip()

            frequency[name] += 1

            labels[name] = entity["label"]

            mentions.setdefault(
                name,
                []
            ).append(entity)

            if name not in seen:

                coverage[name] += 1

                seen.add(name)

    output = []

    total_competitors = len(competitors)

    for entity in frequency:

        output.append({

            "entity": entity,

            "label": labels[entity],

            "mentions": frequency[entity],

            "competitors_using": coverage[entity],

            "coverage_percent": round(

                coverage[entity] /
                total_competitors *
                100,

                2

            )

        })

    output.sort(

        key=lambda x: (

            x["coverage_percent"],

            x["mentions"]

        ),

        reverse=True

    )

    return output