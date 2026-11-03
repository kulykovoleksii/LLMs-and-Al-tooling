import io
import json
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from capstone.classifier import Classification, Intent, RuleBasedClassifier
from capstone.evaluate import evaluate_examples
from capstone.ollama_classifier import OllamaClassifier, OllamaError


GOLDEN_SET = Path(__file__).parents[1] / "data" / "golden.jsonl"


class RuleBasedClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = RuleBasedClassifier()

    def test_golden_set(self) -> None:
        with GOLDEN_SET.open(encoding="utf-8") as examples:
            for line in examples:
                with self.subTest(example=json.loads(line)["id"]):
                    example = json.loads(line)
                    result = self.classifier.classify(example["input"])
                    self.assertEqual(result.intent.value, example["expected_intent"])
                    self.assertEqual(
                        result.needs_clarification, example["needs_clarification"]
                    )

    def test_mixed_intents_require_clarification(self) -> None:
        result = self.classifier.classify("Explain this and draft a reply.")
        self.assertEqual(result.intent, Intent.UNCLEAR)
        self.assertTrue(result.needs_clarification)
        self.assertEqual(result.reason, "conflicting_rules")

    def test_empty_input_is_rejected(self) -> None:
        for value in ("", "  ", "\n"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.classifier.classify(value)

    def test_non_string_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.classifier.classify(123)  # type: ignore[arg-type]

    def test_classification_round_trips_through_schema(self) -> None:
        expected = self.classifier.classify("Що таке retrieval?")
        result = Classification.from_dict(expected.to_dict())
        self.assertEqual(result, expected)

    def test_schema_rejects_unknown_fields_and_inconsistent_flags(self) -> None:
        with self.assertRaises(ValueError):
            Classification.from_dict(
                {
                    "intent": "knowledge_query",
                    "needs_clarification": False,
                    "reason": "matched_knowledge_rule",
                    "extra": "not allowed",
                }
            )

    def test_schema_rejects_non_object(self) -> None:
        with self.assertRaises(ValueError):
            Classification.from_dict([])  # type: ignore[arg-type]

    def test_schema_rejects_inconsistent_flags(self) -> None:
        with self.assertRaises(ValueError):
            Classification.from_dict(
                {
                    "intent": "unclear",
                    "needs_clarification": False,
                    "reason": "no_matching_rule",
                }
            )


class EvaluationTests(unittest.TestCase):
    def test_reports_mismatches_and_provider_errors(self) -> None:
        class FakeClassifier:
            def classify(self, text: str) -> Classification:
                if text == "error":
                    raise OllamaError("invalid model output")
                return Classification(Intent.UNCLEAR, True, "test")

        report = evaluate_examples(
            FakeClassifier(),
            [
                {
                    "id": "match",
                    "input": "request",
                    "expected_intent": "unclear",
                    "needs_clarification": True,
                },
                {
                    "id": "mismatch",
                    "input": "request",
                    "expected_intent": "draft_request",
                    "needs_clarification": False,
                },
                {
                    "id": "error",
                    "input": "error",
                    "expected_intent": "knowledge_query",
                    "needs_clarification": False,
                },
            ],
        )
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["exact_matches"], 1)
        self.assertEqual(report["accuracy"], 1 / 3)
        self.assertEqual([item["id"] for item in report["mismatches"]], ["mismatch", "error"])


class OllamaClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = OllamaClassifier("local-model", timeout=4)

    @staticmethod
    def ollama_response(content: str) -> io.BytesIO:
        envelope = {"message": {"content": content}}
        return io.BytesIO(json.dumps(envelope).encode("utf-8"))

    def test_sends_schema_and_validates_model_output(self) -> None:
        content = json.dumps(
            {
                "intent": "knowledge_query",
                "needs_clarification": False,
                "reason": "ollama_classification",
            }
        )
        with patch(
            "capstone.ollama_classifier.urlopen",
            return_value=self.ollama_response(content),
        ) as open_url:
            result = self.classifier.classify("What is retrieval?")

        self.assertEqual(result.intent, Intent.KNOWLEDGE_QUERY)
        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["model"], "local-model")
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"]["additionalProperties"], False)
        self.assertEqual(open_url.call_args.kwargs["timeout"], 4)

    def test_rejects_malformed_model_output(self) -> None:
        for content in (
            "not json",
            json.dumps(
                {
                    "intent": "unknown",
                    "needs_clarification": False,
                    "reason": "ollama_classification",
                }
            ),
            json.dumps(
                {
                    "intent": "knowledge_query",
                    "needs_clarification": False,
                    "reason": "invented",
                }
            ),
        ):
            with self.subTest(content=content), patch(
                "capstone.ollama_classifier.urlopen",
                return_value=self.ollama_response(content),
            ), self.assertRaises(OllamaError):
                self.classifier.classify("What is retrieval?")

    def test_surfaces_http_failures(self) -> None:
        error = HTTPError(
            "http://127.0.0.1:11434/api/chat", 404, "not found", {}, None
        )
        with patch("capstone.ollama_classifier.urlopen", side_effect=error):
            with self.assertRaisesRegex(OllamaError, "HTTP 404"):
                self.classifier.classify("What is retrieval?")

    def test_rejects_invalid_configuration_and_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            OllamaClassifier("")
        with self.assertRaises(ValueError):
            OllamaClassifier("local-model", timeout=0)
        with self.assertRaises(ValueError):
            self.classifier.classify(" ")


if __name__ == "__main__":
    unittest.main()
