from collections import Counter


def calculate_coverage(all_headings):

    counter = Counter()

    total_competitors = len(all_headings)

    for competitor_headings in all_headings:

        unique_topics = set()

        for heading in competitor_headings:

            heading = heading.strip()

            if heading:
                unique_topics.add(heading)

        counter.update(unique_topics)

    results = []

    for topic, count in counter.items():

        coverage = round(
            (count / total_competitors) * 100,
            1
        )

        results.append(
            {
                "topic": topic,
                "count": count,
                "coverage": coverage
            }
        )

    results.sort(
        key=lambda x: x["coverage"],
        reverse=True
    )

    return results