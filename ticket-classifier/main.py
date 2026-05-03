import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CATEGORIES = ["engineering", "regulatory", "qa", "general"]
EXAMPLES_PER_CATEGORY = 25

def generate_examples(category: str, count: int) -> list:
    prompt = f"""Generate {count} realistic internal company support tickets 
that belong to the category: {category}

Context: This is a medical device company (like a CPAP or cardiac monitor manufacturer).
The four possible categories are:
- engineering: code, bugs, architecture, build pipelines, technical implementation
- regulatory: FDA compliance, 510k submissions, audit trails, documentation approvals  
- qa: test cases, defects, validation, regression testing, quality checks
- general: anything else — meetings, admin, office, HR, general questions

Generate exactly {count} examples for the "{category}" category.
Each example should be a realistic, natural-sounding ticket a real employee might write.
Vary the length and style — some short, some longer, some formal, some casual.

Return ONLY a JSON array, no other text, no markdown, no explanation.
Format:
[
  {{"text": "ticket text here", "label": "{category}"}},
  {{"text": "ticket text here", "label": "{category}"}}
]"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # strip markdown code blocks if Claude wrapped it
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # debug — see exactly what Claude returned
    print(f"  Raw response preview: {raw[:200]}")

    examples = json.loads(raw)
    
    return examples


def main():
    all_examples = []

    for category in CATEGORIES:
        print(f"Generating {EXAMPLES_PER_CATEGORY} examples for: {category}...")
        examples = generate_examples(category, EXAMPLES_PER_CATEGORY)
        all_examples.extend(examples)
        print(f"  ✓ got {len(examples)} examples")

    # shuffle so categories are mixed
    import random
    random.shuffle(all_examples)

    # save as JSONL
    with open("eval_set.jsonl", "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    print(f"\n✓ Saved {len(all_examples)} examples to eval_set.jsonl")

    # quick sanity check
    counts = {}
    for ex in all_examples:
        counts[ex["label"]] = counts.get(ex["label"], 0) + 1
    print("Distribution:", counts)


if __name__ == "__main__":
    main()