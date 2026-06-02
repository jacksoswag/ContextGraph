from __future__ import annotations

import json
from pathlib import Path

DOMAINS = (
    ("science", ("bees", "produce", "honey"), (("cows", "produce", "milk"), ("lab", "produce", "data"))),
    ("programming", ("python", "used for", "automation"), (("java", "used for", "android apps"), ("python", "is a", "snake"))),
    ("finance", ("invoice", "used for", "requesting payment"), (("receipt", "used for", "proof of purchase"), ("payment", "causes", "cash flow"))),
    ("law", ("contract", "has", "obligations"), (("contract", "at location", "office"), ("statute", "has", "sections"))),
    ("medicine", ("infection", "causes", "fever"), (("exercise", "causes", "sweat"), ("fever", "has property", "dangerous"))),
    ("history", ("renaissance", "at location", "europe"), (("revolution", "at location", "france"), ("europe", "has", "cities"))),
    ("planning", ("checklist", "used for", "tracking tasks"), (("calendar", "used for", "scheduling"), ("task", "has property", "urgent"))),
    ("sports", ("striker", "capable of", "score goals"), (("goalkeeper", "capable of", "block shots"), ("coach", "has", "strategy"))),
    ("writing", ("outline", "used for", "organizing essays"), (("draft", "used for", "revision"), ("essay", "has", "paragraphs"))),
    ("objects", ("wheel", "part of", "vehicle"), (("handle", "part of", "door"), ("vehicle", "used for", "transport"))),
)


def build_rows() -> tuple[dict[str, object], ...]:
    rows = []
    for index, (domain, target, distractors) in enumerate(DOMAINS, start=1):
        source, relation, expected_target = target
        rows.append(
            {
                "id": f"stress.generated.{domain}.{index:03d}",
                "category": "stress",
                "prompt": f"What is {source} {relation}?",
                "memory_facts": [list(target), *[list(item) for item in distractors]],
                "expected_facts": [list(target)],
                "expected_targets": [expected_target],
                "forbidden_facts": [list(item) for item in distractors],
                "forbidden_terms": [item[2] for item in distractors],
            }
        )
    return tuple(rows)


def main() -> int:
    target = Path(__file__).resolve().parent / "fixtures" / "stress_cases.jsonl"
    target.write_text("\n".join(json.dumps(row, sort_keys=True) for row in build_rows()) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
