from modules.topics.topic_parser import parse_topics

sample = """
TOPIC: Mutual Fund Fundamentals

COVERED:
- Explanation of mutual funds as pooled investments
- Portfolio diversification
- Professional management

EVIDENCE:
- Mutual funds pool money from investors
- Managed by investment professionals

TOPIC: Expense Ratios

COVERED:
- Definition of expense ratios
- Impact on returns

EVIDENCE:
- 1% expense ratio example
"""

parsed = parse_topics(sample)

print(parsed)