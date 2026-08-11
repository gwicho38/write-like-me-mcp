"""Tests for language detection and per-language rule sets.

These tests prove:
- Short, ordinary sentences in each supported language are detected correctly.
- Text with no recognizable function words reports ``unknown`` rather than
  guessing, so downstream metrics can be skipped instead of computed wrongly.
- Every supported language exposes a complete rule set, so no metric silently
  falls back to English tables for non-English text.
- Detection ignores case and accents-as-written (it never mutates its input).

All fixtures are synthetic sentences written for the test, not corpus content.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from write_like_me_mcp.language import (  # noqa: E402
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    UNKNOWN_LANGUAGE,
    detect_language,
    rules_for,
)

SAMPLES: dict[str, str] = {
    "en": "I think this is the part that we should have written down before the meeting.",
    "fr": "Je pense que nous devons écrire cette partie avant la réunion de demain.",
    "es": "Creo que nosotros debemos escribir esta parte antes de la reunión de mañana.",
    "pt": "Eu acho que nós devemos escrever esta parte antes da reunião de amanhã.",
    "it": "Penso che noi dobbiamo scrivere questa parte prima della riunione di domani.",
}


class TestDetectLanguage:
    """``detect_language`` distinguishes the supported languages."""

    @pytest.mark.parametrize("code", sorted(SAMPLES))
    def test_detects_each_supported_language(self, code: str) -> None:
        assert detect_language(SAMPLES[code]) == code

    def test_empty_text_is_unknown(self) -> None:
        assert detect_language("") == UNKNOWN_LANGUAGE

    def test_text_without_function_words_is_unknown(self) -> None:
        """A bare list of proper nouns carries no language signal."""
        assert detect_language("Holguín 1952 Mayarí 1961 Valenzuela") == UNKNOWN_LANGUAGE

    def test_detection_is_case_insensitive(self) -> None:
        assert detect_language(SAMPLES["fr"].upper()) == "fr"

    def test_detection_does_not_mutate_input(self) -> None:
        text = SAMPLES["es"]
        detect_language(text)
        assert text == SAMPLES["es"]

    def test_dominant_language_wins_in_mixed_text(self) -> None:
        """A stray English clause must not flip a mostly-Spanish document."""
        mixed = SAMPLES["es"] + " " + SAMPLES["es"] + " The meeting is over."
        assert detect_language(mixed) == "es"


class TestRulesFor:
    """Every supported language exposes a complete, non-English-defaulted rule set."""

    @pytest.mark.parametrize("code", sorted(SUPPORTED_LANGUAGES))
    def test_every_supported_language_has_rules(self, code: str) -> None:
        rules = rules_for(code)
        assert rules.code == code
        assert rules.be_verbs, f"{code} has no passive auxiliaries"
        assert rules.first_person, f"{code} has no first-person pronouns"

    def test_unknown_language_falls_back_to_default(self) -> None:
        assert rules_for(UNKNOWN_LANGUAGE).code == DEFAULT_LANGUAGE

    def test_language_rule_sets_are_distinct(self) -> None:
        """A copied-from-English table would silently produce English metrics."""
        assert rules_for("fr").be_verbs != rules_for("en").be_verbs
        assert rules_for("es").first_person != rules_for("en").first_person

    def test_mandatory_elisions_are_declared_for_elision_languages(self) -> None:
        """French/Italian elisions are grammar, not a stylistic contraction habit."""
        assert rules_for("fr").mandatory_elisions
        assert rules_for("it").mandatory_elisions
        assert not rules_for("en").mandatory_elisions


class TestDetectionConfidence:
    """Detection abstains rather than guessing on thin evidence."""

    def test_very_short_fragment_is_unknown(self) -> None:
        """A handful of tokens cannot separate close Romance languages."""
        assert detect_language("de la casa y el trabajo") == UNKNOWN_LANGUAGE

    def test_ambiguous_shared_vocabulary_is_unknown(self) -> None:
        """Words common to Spanish and Portuguese must not pick a winner."""
        ambiguous = " ".join(["de", "a", "o", "que", "em", "por", "para"] * 6)
        assert detect_language(ambiguous) == UNKNOWN_LANGUAGE

    def test_clear_winner_still_detected_at_moderate_length(self) -> None:
        assert detect_language(SAMPLES["pt"] + " " + SAMPLES["pt"]) == "pt"
