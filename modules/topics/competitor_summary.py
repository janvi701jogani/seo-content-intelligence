def generate_competitor_summary(
    client,
    content
):

    if not client:
        return "OpenAI client not initialized."

    content = content[:12000]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are an elite SEO content analyst.

Analyze the competitor article and
return structured competitive intelligence.
"""
            },

            {
                "role": "user",
                "content": f"""
Analyze this article.

ARTICLE:

{content}

Return exactly in this format:

MAIN THEMES
- ...

KEY POINTS
- ...

UNIQUE INSIGHTS

STATISTICS / EVIDENCE
- ...

TARGET AUDIENCE
- ...

INFORMATION GAIN
- ...

DEPTH SCORE
x/10

Return only the analysis.
"""
            }
        ]
    )

    return response.choices[0].message.content