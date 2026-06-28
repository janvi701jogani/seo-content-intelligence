import re

def parse_topics(text):

    results = []

    topic_blocks = re.split(
        r"TOPIC:\s*",
        text
    )

    for block in topic_blocks:

        block = block.strip()

        if not block:
            continue

        lines = block.splitlines()

        topic = lines[0].strip()

        covered = []
        evidence = []

        current_section = None

        for line in lines[1:]:

            line = line.strip()

            if not line:
                continue

            if line.startswith("COVERED:"):
                current_section = "covered"
                continue

            if line.startswith("EVIDENCE:"):
                current_section = "evidence"
                continue

            if line.startswith("-"):

                item = line.lstrip("-").strip()

                if current_section == "covered":
                    covered.append(item)

                elif current_section == "evidence":
                    evidence.append(item)

        results.append(
            {
                "topic": topic,
                "covered": covered,
                "evidence": evidence
            }
        )

    return results