def generate_brief(
    client,
    keyword,
    geography,
    serp_summary
):

    if not client:
        return "OpenAI client not initialized."

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You are an advanced SEO content strategist."
            },
            {
                "role": "user",
                "content": f"""
Create a comprehensive SEO content brief.

Keyword:
{keyword}

Geography:
{geography}

SERP DATA:
{serp_summary}

Include:
- search intent
- recommended H1
- detailed H2/H3 structure
- topical coverage
- semantic entities
- content flow
- formatting suggestions
- AI Overview optimization
- user psychology considerations
- content differentiation
- opportunities competitors missed
"""
            }
        ]
    )

    return response.choices[0].message.content