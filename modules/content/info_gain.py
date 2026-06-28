def generate_info_gain(
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
                "content": "You are an information gain and content differentiation expert."
            },
            {
                "role": "user",
                "content": f"""
Analyze the keyword:

{keyword}

And these SERP results:

{serp_summary}

Identify:

1. Missing Angles
- topics competitors ignore
- uncommon perspectives
- overlooked user needs

2. Research Opportunities
- studies that could strengthen content
- statistics opportunities
- data-backed insights

3. Original Content Ideas
- frameworks
- methodologies
- checklists
- decision models

4. Community Insights
- likely Reddit discussions
- likely Quora discussions
- pain points
- misconceptions

5. Comparison Opportunities
- side-by-side comparisons
- alternatives
- trade-offs

6. Brand Differentiation
- ways a brand can stand out
- unique value propositions

7. Information Gain Opportunities
- sections that would make the article meaningfully different from competitors
"""
            }
        ]
    )

    return response.choices[0].message.content