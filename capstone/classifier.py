"""Deterministic baseline for routing supported assistant requests."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class Intent(str, Enum):
    KNOWLEDGE_QUERY = "knowledge_query"
    DRAFT_REQUEST = "draft_request"
    UNCLEAR = "unclear"


@dataclass(frozen=True, slots=True)
class Classification:
    intent: Intent
    needs_clarification: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.intent, Intent):
            raise ValueError("intent must be a supported Intent")
        if not isinstance(self.needs_clarification, bool):
            raise ValueError("needs_clarification must be a boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if (self.intent is Intent.UNCLEAR) != self.needs_clarification:
            raise ValueError("unclear intent and clarification flag must agree")

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "intent": self.intent.value,
            "needs_clarification": self.needs_clarification,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Classification:
        if not isinstance(value, Mapping):
            raise ValueError("classification must be an object")

        expected_keys = {"intent", "needs_clarification", "reason"}
        if set(value) != expected_keys:
            raise ValueError(f"classification must contain exactly {sorted(expected_keys)}")

        try:
            intent = Intent(value["intent"])
        except (TypeError, ValueError) as error:
            raise ValueError("intent must be a supported intent value") from error

        needs_clarification = value["needs_clarification"]
        reason = value["reason"]
        if not isinstance(needs_clarification, bool):
            raise ValueError("needs_clarification must be a boolean")
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")

        return cls(
            intent=intent,
            needs_clarification=needs_clarification,
            reason=reason,
        )


class RuleBasedClassifier:
    """Classify explicit knowledge or draft requests; defer other inputs."""

    _KNOWLEDGE_PHRASES = (
        "what is",
        "explain",
        "how does",
        "how do",
        "tell me about",
        "where can i find",
        "summarize",
        "що таке",
        "поясни",
        "як працює",
        "як працюють",
        "розкажи про",
        "де знайти",
        "підсумуй",
        "чи можеш пояснити",
    )
    _DRAFT_PHRASES = (
        "draft",
        "write an email",
        "write a message",
        "compose",
        "create a draft",
        "please write",
        "напиши лист",
        "напиши короткий лист",
        "напиши повідомлення",
        "створи чернетку",
        "підготуй лист",
        "склади повідомлення",
        "сформулюй",
    )

    def __init__(self) -> None:
        self._knowledge_patterns = self._compile(self._KNOWLEDGE_PHRASES)
        self._draft_patterns = self._compile(self._DRAFT_PHRASES)

    def classify(self, text: str) -> Classification:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        normalized = unicodedata.normalize("NFKC", text).casefold().strip()
        if not normalized:
            raise ValueError("text must not be empty")

        knowledge_match = self._matches_any(normalized, self._knowledge_patterns)
        draft_match = self._matches_any(normalized, self._draft_patterns)

        if knowledge_match and not draft_match:
            return Classification(
                Intent.KNOWLEDGE_QUERY, False, "matched_knowledge_rule"
            )
        if draft_match and not knowledge_match:
            return Classification(Intent.DRAFT_REQUEST, False, "matched_draft_rule")
        if knowledge_match and draft_match:
            return Classification(Intent.UNCLEAR, True, "conflicting_rules")
        return Classification(Intent.UNCLEAR, True, "no_matching_rule")

    @staticmethod
    def _compile(phrases: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
        return tuple(
            re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)")
            for phrase in phrases
        )

    @staticmethod
    def _matches_any(
        text: str, patterns: tuple[re.Pattern[str], ...]
    ) -> bool:
        return any(pattern.search(text) for pattern in patterns)
