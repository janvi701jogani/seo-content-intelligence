from modules.topics.coverage import calculate_coverage

sample = [
    ["SIP", "Taxation", "Risk"],
    ["SIP", "Taxation"],
    ["SIP", "Expense Ratio"]
]

print(calculate_coverage(sample))