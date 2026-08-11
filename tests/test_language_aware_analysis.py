"""Tests for language-aware corpus analysis.

These tests prove:
- Tokenization keeps accented words whole, so non-English corpora are not
  measured on truncated tokens.
- Passive voice is detected with the auxiliaries of the document's own
  language, instead of being diluted to zero by English-only tables.
- Mandatory French/Italian elisions do not inflate the contraction rate.
- A multilingual corpus reports a per-language metric breakdown, and the
  top-level metrics describe the dominant language rather than a blend.
- A single-language corpus is unaffected (backwards compatibility).

All fixtures are synthetic sentences written for the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from write_like_me_mcp import style_analyzer  # noqa: E402

EN_PARAGRAPH = (
    "I think the report was written by the team before the deadline. "
    "We should have shipped it earlier. It was reviewed twice and it was "
    "approved by everyone who read it."
)
FR_PARAGRAPH = (
    "Je pense que le rapport a été écrit par l'équipe avant la date limite. "
    "Nous aurions dû le publier plus tôt. Il a été relu deux fois et il a "
    "été approuvé par tous ceux qui l'ont lu."
)
ES_PARAGRAPH = (
    "Creo que el informe fue escrito por el equipo antes de la fecha límite. "
    "Nosotros deberíamos haberlo publicado antes. Fue revisado dos veces y "
    "fue aprobado por todos los que lo leyeron."
)


class TestUnicodeTokenization:
    """Accented words survive tokenization intact."""

    def test_accented_words_are_not_truncated(self) -> None:
        tokens = style_analyzer._tokenize_words("café niño réunion süß")
        assert "café" in tokens
        assert "niño" in tokens
        assert "réunion" in tokens

    def test_ascii_tokenization_is_unchanged(self) -> None:
        assert style_analyzer._tokenize_words("don't stop the count") == [
            "don't",
            "stop",
            "the",
            "count",
        ]

    def test_accented_word_counts_as_one_token(self) -> None:
        """Truncation used to split one word into a stem plus nothing,
        deflating total_words and inflating lexical diversity."""
        assert len(style_analyzer._tokenize_words("réunion")) == 1


class TestPerLanguagePassiveVoice:
    """Passive voice uses the auxiliaries of the text's own language."""

    def test_french_passive_is_detected(self) -> None:
        profile = style_analyzer.analyze_text(FR_PARAGRAPH, profile_name="t")
        assert profile.passive_voice_rate > 0

    def test_spanish_passive_is_detected(self) -> None:
        profile = style_analyzer.analyze_text(ES_PARAGRAPH, profile_name="t")
        assert profile.passive_voice_rate > 0

    def test_english_passive_still_detected(self) -> None:
        profile = style_analyzer.analyze_text(EN_PARAGRAPH, profile_name="t")
        assert profile.passive_voice_rate > 0


class TestContractionsVersusElisions:
    """An apostrophe is a style choice in English and grammar in French."""

    def test_french_mandatory_elisions_do_not_count_as_contractions(self) -> None:
        profile = style_analyzer.analyze_text(
            "L'équipe a écrit l'article et j'ai relu l'introduction avant la "
            "réunion de l'après-midi avec l'équipe.",
            profile_name="t",
        )
        assert profile.contraction_rate == 0.0

    def test_english_contractions_still_counted(self) -> None:
        profile = style_analyzer.analyze_text(
            "I don't think we'll ship it today. It isn't ready and we can't "
            "rush it. They won't agree.",
            profile_name="t",
        )
        assert profile.contraction_rate > 0


class TestLanguageBreakdown:
    """A multilingual corpus is reported per language, not blended."""

    def test_multilingual_corpus_reports_each_language(self) -> None:
        profile = style_analyzer.analyze_documents(
            [EN_PARAGRAPH, FR_PARAGRAPH, ES_PARAGRAPH], profile_name="t"
        )
        assert set(profile.languages) >= {"en", "fr", "es"}

    def test_each_language_entry_carries_its_own_metrics(self) -> None:
        profile = style_analyzer.analyze_documents(
            [EN_PARAGRAPH, FR_PARAGRAPH], profile_name="t"
        )
        for code in ("en", "fr"):
            entry = profile.languages[code]
            assert entry["doc_count"] >= 1
            assert entry["total_words"] > 0
            assert "passive_voice_rate" in entry
            assert "contraction_rate" in entry

    def test_dominant_language_is_recorded(self) -> None:
        profile = style_analyzer.analyze_documents(
            [FR_PARAGRAPH, FR_PARAGRAPH, FR_PARAGRAPH, EN_PARAGRAPH],
            profile_name="t",
        )
        assert profile.metadata["dominant_language"] == "fr"

    def test_top_level_metrics_describe_the_dominant_language(self) -> None:
        """Blending would report a passive rate diluted by the other language."""
        profile = style_analyzer.analyze_documents(
            [FR_PARAGRAPH, FR_PARAGRAPH, FR_PARAGRAPH, EN_PARAGRAPH],
            profile_name="t",
        )
        assert profile.passive_voice_rate == profile.languages["fr"]["passive_voice_rate"]

    def test_single_language_corpus_is_unaffected(self) -> None:
        profile = style_analyzer.analyze_documents(
            [EN_PARAGRAPH, EN_PARAGRAPH], profile_name="t"
        )
        assert profile.metadata["dominant_language"] == "en"
        assert profile.passive_voice_rate == profile.languages["en"]["passive_voice_rate"]

    def test_style_guide_names_the_dominant_language(self) -> None:
        profile = style_analyzer.analyze_documents(
            [ES_PARAGRAPH, ES_PARAGRAPH], profile_name="t"
        )
        assert "Spanish" in profile.style_guide_md

    def test_undetectable_documents_do_not_create_a_language_entry(self) -> None:
        """A list of names and dates carries no language signal."""
        profile = style_analyzer.analyze_documents(
            [EN_PARAGRAPH, "Holguín 1952 Mayarí 1961 Valenzuela Almira 1990"],
            profile_name="t",
        )
        assert "unknown" not in profile.languages


class TestDraftLanguageMatching:
    """A draft is compared against the author's metrics *in the draft's language*."""

    def _corpus(self) -> object:
        return style_analyzer.analyze_documents(
            [EN_PARAGRAPH, EN_PARAGRAPH, EN_PARAGRAPH, FR_PARAGRAPH, FR_PARAGRAPH],
            profile_name="t",
        )

    def test_french_draft_targets_french_metrics(self) -> None:
        from write_like_me_mcp.server import _target_metrics_for

        profile = self._corpus()
        draft = style_analyzer.analyze_text(FR_PARAGRAPH, profile_name="d")
        targets, language = _target_metrics_for(profile, draft)

        assert language == "fr"
        assert targets["passive_voice_rate"] == profile.languages["fr"]["passive_voice_rate"]

    def test_english_draft_targets_dominant_metrics(self) -> None:
        from write_like_me_mcp.server import _target_metrics_for

        profile = self._corpus()
        draft = style_analyzer.analyze_text(EN_PARAGRAPH, profile_name="d")
        targets, language = _target_metrics_for(profile, draft)

        assert language == "en"
        assert targets["passive_voice_rate"] == profile.passive_voice_rate

    def test_draft_in_unrepresented_language_falls_back_to_top_level(self) -> None:
        """The author has no Italian corpus, so the dominant metrics are used."""
        from write_like_me_mcp.server import _target_metrics_for

        profile = self._corpus()
        draft = style_analyzer.analyze_text(
            "Penso che noi dobbiamo scrivere questa parte prima della riunione "
            "di domani con tutti i colleghi del gruppo.",
            profile_name="d",
        )
        targets, language = _target_metrics_for(profile, draft)

        assert language is None
        assert targets["passive_voice_rate"] == profile.passive_voice_rate
