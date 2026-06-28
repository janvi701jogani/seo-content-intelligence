def extract_topics(
    client,
    content
):

    if not client:
        return []

    content = content[:12000]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are an SEO topical analysis expert.

Your job is to identify meaningful conceptual subtopics
covered within an article.

Do NOT return:
- keywords
- entities
- brand names
- headings copied verbatim

Return conceptual topics only.
"""
            },
            {
                "role": "user",
                "content": f"""
Analyze this article content.

ARTICLE:

{content}

Return:

One topic per line.

Examples:

Mutual Fund Basics
Diversification
Risk Management
Expense Ratios
Taxation
Fund Selection
Portfolio Allocation

Return ONLY the topic list.
"""
            }
        ]
    )

    result = response.choices[0].message.content

    topics = [
        topic.strip()
        for topic in result.split("\n")
        if topic.strip()
    ]

    return topics