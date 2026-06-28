import re
import math
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

import spacy
from bs4 import BeautifulSoup

from keybert import KeyBERT
import yake
from rapidfuzz import fuzz
from gliner import GLiNER


###########################################################
# LOAD MODELS
###########################################################

nlp = spacy.load("en_core_web_sm")

keybert_model = KeyBERT(
    "all-MiniLM-L6-v2"
)

gliner_model = GLiNER.from_pretrained(
    "urchade/gliner_medium-v2.1"
)

yake_extractor = yake.KeywordExtractor(
    lan="en",
    n=3,
    top=200
)


###########################################################
# STOP ENTITIES
###########################################################

STOP_ENTITIES = {

    "home",
    "menu",
    "search",
    "login",
    "sign in",
    "sign up",
    "read more",
    "learn more",
    "click here",
    "cookies",
    "privacy",
    "newsletter",
    "copyright",
    "terms",
    "facebook",
    "instagram",
    "linkedin",
    "youtube",
    "twitter",
    "skip to content"

}


###########################################################
# BASIC CLEANING
###########################################################

def clean_text(text):

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    text = text.replace("\xa0", " ")

    return text.strip()


###########################################################
# DOCUMENT STATISTICS
###########################################################

def get_statistics(text):

    words = text.split()

    sentences = re.split(
        r"[.!?]",
        text
    )

    paragraphs = [

        p

        for p in text.split("\n")

        if p.strip()

    ]

    return {

        "characters": len(text),

        "words": len(words),

        "sentences": len(sentences),

        "paragraphs": len(paragraphs),

        "reading_time": round(
            len(words) / 220,
            2
        )

    }


###########################################################
# METADATA CLEANER
###########################################################

def clean_metadata(metadata):

    cleaned = {}

    for key, value in metadata.items():

        if value is None:

            continue

        cleaned[key] = clean_text(
            str(value)
        )

    return cleaned


###########################################################
# CONTENT STRUCTURE
###########################################################

def extract_structure(text):

    lines = [

        clean_text(line)

        for line in text.split("\n")

    ]

    lines = [

        line

        for line in lines

        if line

    ]

    structure = {

        "title": "",

        "headings": [],

        "paragraphs": []

    }

    if lines:

        structure["title"] = lines[0]

    paragraph = []

    for line in lines:

        if len(line.split()) <= 12:

            structure["headings"].append(line)

        else:

            paragraph.append(line)

    structure["paragraphs"] = paragraph

    return structure


###########################################################
# SPACY ENTITIES
###########################################################

def spacy_entities(text):

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        value = clean_text(ent.text)

        if len(value) < 2:
            continue

        entities.append({

            "entity": value,

            "type": ent.label_,

            "source": "spacy"

        })

    return entities


###########################################################
# GLINER ENTITIES
###########################################################

def gliner_entities(text):

    labels = [

        "person",

        "organization",

        "location",

        "product",

        "event",

        "technology",

        "financial term",

        "company",

        "service"

    ]

    results = gliner_model.predict_entities(

        text,

        labels

    )

    output = []

    for entity in results:

        output.append({

            "entity": clean_text(

                entity["text"]

            ),

            "type": entity["label"],

            "source": "gliner"

        })

    return output

###########################################################
# KEYBERT
###########################################################

def keybert_entities(text):

    keywords = keybert_model.extract_keywords(

        text,

        keyphrase_ngram_range=(1, 3),

        stop_words="english",

        top_n=200

    )

    output = []

    for keyword, score in keywords:

        keyword = clean_text(keyword)

        if len(keyword) < 2:
            continue

        output.append({

            "entity": keyword,

            "type": "KEYPHRASE",

            "score": float(score),

            "source": "keybert"

        })

    return output


###########################################################
# YAKE
###########################################################

def yake_entities(text):

    keywords = yake_extractor.extract_keywords(text)

    output = []

    for keyword, score in keywords:

        keyword = clean_text(keyword)

        if len(keyword) < 2:
            continue

        output.append({

            "entity": keyword,

            "type": "KEYPHRASE",

            "score": round(

                1 - score,

                4

            ),

            "source": "yake"

        })

    return output


###########################################################
# REMOVE NOISE
###########################################################

def remove_noise(entities):

    cleaned = []

    for entity in entities:

        name = entity["entity"].strip()

        if len(name) < 2:
            continue

        lower = name.lower()

        if lower in STOP_ENTITIES:
            continue

        if lower.isdigit():
            continue

        if len(lower.split()) > 8:
            continue

        cleaned.append(entity)

    return cleaned


###########################################################
# MERGE SIMILAR
###########################################################

def merge_entities(entities):

    merged = {}

    for entity in entities:

        name = entity["entity"]

        matched = None

        for existing in merged:

            similarity = fuzz.token_sort_ratio(

                existing.lower(),

                name.lower()

            )

            if similarity >= 90:

                matched = existing

                break

        if matched:

            merged[matched]["mentions"] += 1

            merged[matched]["sources"].add(

                entity["source"]

            )

            if "score" in entity:

                merged[matched]["score"] += entity["score"]

        else:

            merged[name] = {

                "entity": name,

                "type": entity["type"],

                "mentions": 1,

                "score": entity.get(

                    "score",

                    1

                ),

                "sources": {

                    entity["source"]

                }

            }

    return merged


###########################################################
# INITIAL IMPORTANCE
###########################################################

def calculate_importance(merged):

    results = []

    for entity in merged.values():

        source_bonus = len(

            entity["sources"]

        ) * 5

        importance = (

            entity["mentions"] * 3 +

            source_bonus +

            entity["score"]

        )

        entity["importance"] = round(

            importance,

            2

        )

        entity["sources"] = list(

            entity["sources"]

        )

        results.append(entity)

    results.sort(

        key=lambda x: x["importance"],

        reverse=True

    )

    return results


###########################################################
# PIPELINE
###########################################################

def extract_entities(text):

    all_entities = []

    try:

        all_entities.extend(

            spacy_entities(text)

        )

    except Exception:

        pass

    try:

        all_entities.extend(

            gliner_entities(text)

        )

    except Exception:

        pass

    try:

        all_entities.extend(

            keybert_entities(text)

        )

    except Exception:

        pass

    try:

        all_entities.extend(

            yake_entities(text)

        )

    except Exception:

        pass

    all_entities = remove_noise(

        all_entities

    )

    merged = merge_entities(

        all_entities

    )

    return calculate_importance(

        merged

    )

###########################################################
# COMPETITOR ENTITY ENGINE
###########################################################

def build_entity_dashboard(competitors):

    entity_map = {}

    total_competitors = len(competitors)

    for competitor in competitors:

        entities = extract_entities(
            competitor.get("text", "")
        )

        competitor["entities"] = entities

        seen = set()

        for entity in entities:

            name = entity["entity"]

            if name not in entity_map:

                entity_map[name] = {

                    "entity": name,

                    "type": entity["type"],

                    "mentions": 0,

                    "competitors_using": 0,

                    "importance": 0,

                    "sources": Counter(),

                    "urls": [],

                    "positions": [],

                    "co_occurrence": Counter()

                }

            entity_map[name]["mentions"] += entity["mentions"]

            entity_map[name]["importance"] += entity["importance"]

            for source in entity["sources"]:

                entity_map[name]["sources"][source] += 1

            if name not in seen:

                entity_map[name]["competitors_using"] += 1

                entity_map[name]["urls"].append(
                    competitor["url"]
                )

                entity_map[name]["positions"].append(
                    competitor["position"]
                )

                seen.add(name)

        entity_names = [

            x["entity"]

            for x in entities

        ]

        for entity in entity_names:

            for other in entity_names:

                if entity == other:
                    continue

                entity_map[entity]["co_occurrence"][other] += 1

    dashboard = []

    for entity in entity_map.values():

        coverage = round(

            entity["competitors_using"] /

            total_competitors *

            100,

            2

        )

        avg_mentions = round(

            entity["mentions"] /

            entity["competitors_using"],

            2

        )

        dashboard.append({

            "entity": entity["entity"],

            "type": entity["type"],

            "coverage": coverage,

            "competitors": entity["competitors_using"],

            "mentions": entity["mentions"],

            "avg_mentions": avg_mentions,

            "importance": round(

                entity["importance"],

                2

            ),

            "sources": sorted(

                entity["sources"].keys()

            ),

            "urls": entity["urls"],

            "positions": entity["positions"],

            "related_entities": [

                x

                for x, _ in

                entity["co_occurrence"].most_common(20)

            ]

        })

    dashboard.sort(

        key=lambda x: (

            x["coverage"],

            x["importance"],

            x["mentions"]

        ),

        reverse=True

    )

    return dashboard


###########################################################
# FULL INTELLIGENCE ENGINE
###########################################################

def run_intelligence_engine(competitors):

    for competitor in competitors:

        competitor["statistics"] = get_statistics(

            competitor["text"]

        )

        competitor["structure"] = extract_structure(

            competitor["text"]

        )

        competitor["metadata"] = clean_metadata(

            competitor["metadata"]

        )

    dashboard = build_entity_dashboard(

        competitors

    )

    return {

        "competitors": competitors,

        "entities": dashboard

    }