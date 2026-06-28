def expand_topics(
    client,
    content,
    topics
):

    if not client:
        return "OpenAI client not initialized."

    content = content[:12000]

    topic_list = "\n".join(topics)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are an SEO content analyst.

For each topic:

1. Explain what was covered.
2. Mention examples or evidence used.
3. Stay grounded in the article.
4. Do not invent information.
"""
            },
            {
                "role": "user",
                "content": f"""
ARTICLE:

{content}

TOPICS:

{topic_list}

For EACH topic return:

TOPIC: <topic>

COVERED:
- point
- point

EVIDENCE:
- evidence
- evidence

Only include information actually present in the article.
"""
            }
        ]
    )

    return response.choices[0].message.content