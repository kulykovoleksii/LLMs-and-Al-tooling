"""Compare a classifier with the fixed synthetic golden set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Protocol
from urllib.error import URLError

from .classifier import Classification, RuleBasedClassifier
from .ollama_classifier import OllamaClassifier, OllamaError


GOLDEN_SET = Path(__file__).parent / "data" / "golden.jsonl"


class Classifier(Protocol):
    def classify(self, text: str) -> Classification: ...


def evaluate(classifier: Classifier) -> dict[str, object]:
    examples = [
        json.loads(line)
        for line in GOLDEN_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return evaluate_examples(classifier, examples)


def evaluate_examples(
    classifier: Classifier, examples: list[dict[str, object]]
) -> dict[str, object]:
    mismatches: list[dict[str, object]] = []
    exact_matches = 0

    for example in examples:
        expected = {
            "intent": example["expected_intent"],
            "needs_clarification": example["needs_clarification"],
        }
        try:
            actual = classifier.classify(str(example["input"]))
        except (OllamaError, URLError, TimeoutError) as error:
            mismatches.append(
                {"id": example["id"], "expected": expected, "error": str(error)}
            )
            continue

        actual_values = {
            "intent": actual.intent.value,
            "needs_clarification": actual.needs_clarification,
        }
        if actual_values == expected:
            exact_matches += 1
        else:
            mismatches.append(
                {
                    "id": example["id"],
                    "expected": expected,
                    "actual": actual_values,
                }
            )

    total = len(examples)
    return {
        "total": total,
        "exact_matches": exact_matches,
        "accuracy": exact_matches / total if total else 0.0,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a classifier on the golden set.")
    parser.add_argument("--provider", choices=("rules", "ollama"), default="rules")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL"))
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.provider == "ollama":
        if not args.model:
            parser.error("--model or OLLAMA_MODEL is required for the ollama provider")
        classifier: Classifier = OllamaClassifier(
            args.model, args.ollama_url, args.timeout
        )
    else:
        classifier = RuleBasedClassifier()

    report = evaluate(classifier)
    report["provider"] = args.provider
    if args.provider == "ollama":
        report["model"] = args.model
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
