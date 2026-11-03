"""Command-line entry point for the baseline classifier."""

import argparse
import json
import os
from urllib.error import URLError

from .classifier import RuleBasedClassifier
from .ollama_classifier import OllamaClassifier, OllamaError


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a knowledge-assistant request.")
    parser.add_argument("request", help="Request text to classify")
    parser.add_argument(
        "--provider",
        choices=("rules", "ollama"),
        default="rules",
        help="Use the offline rules baseline or a local Ollama model",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL"),
        help="Local Ollama model name (or set OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
        help="Ollama server URL (or set OLLAMA_URL)",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.provider == "ollama":
        if not args.model:
            parser.error("--model or OLLAMA_MODEL is required for the ollama provider")
        classifier = OllamaClassifier(args.model, args.ollama_url, args.timeout)
    else:
        classifier = RuleBasedClassifier()

    try:
        result = classifier.classify(args.request)
    except (OllamaError, URLError, TimeoutError) as error:
        parser.error(str(error))

    print(json.dumps(result.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
