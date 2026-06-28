def generate_faqs(
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
                "content": "You are an SEO FAQ strategist."
            },
            {
                "role": "user",
                "content": f"""
Generate advanced FAQs for:

{keyword}

Based on these SERP results:

{serp_summary}

Include:
- informational FAQs
- transactional FAQs
- comparison FAQs
- objection handling FAQs
- AI Overview friendly FAQs
- trust/reassurance FAQs
"""
            }
        ]
    )

    return response.choices[0].message.content	