def generate_eeat(
    client,
    keyword,
    serp_summary
):

    if not client:
        return "OpenAI client not initialized."

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an EEAT optimization expert."
            },
            {
                "role": "user",
                "content": f"""
Suggest EEAT improvements for:

{keyword}

Based on these SERP results:

{serp_summary}

Include:

1. Expertise Signals
- expert contributions
- credentials to highlight
- author profile recommendations

2. Experience Signals
- first-hand experience opportunities
- case studies
- practical examples

3. Authoritativeness Signals
- citations to add
- statistics opportunities
- industry sources to reference
- government sources
- academic sources

4. Trust Signals
- transparency recommendations
- disclaimer opportunities
- trust-building sections
- credibility enhancements

5. YMYL Considerations
- factual verification needs
- risk statements
- compliance considerations

6. AI Overview Readiness
- trust-enhancing answer blocks
- fact-heavy sections
- citation-friendly content

7. Competitor Weaknesses
- trust gaps competitors may have
- authority opportunities competitors missed
"""
            }
        ]
    )

    return response.choices[0].message.content