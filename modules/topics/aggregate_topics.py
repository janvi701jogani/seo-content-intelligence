from collections import defaultdict


def aggregate_topics(
    competitor_topics
):

    topic_map = defaultdict(list)

    for competitor in competitor_topics:

        for topic_data in competitor:

            topic = topic_data["topic"]

            topic_map[topic].append(
                topic_data
            )

    results = []

    for topic, entries in topic_map.items():

        results.append(
            {
                "topic": topic,
                "coverage_count": len(entries),
                "entries": entries
            }
        )

    results.sort(
        key=lambda x: x["coverage_count"],
        reverse=True
    )

    return results