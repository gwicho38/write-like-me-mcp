"""Tests for the deterministic style analyzer (style_analyzer.py).

Written before the implementation (TDD). They pin the frozen v0.1 metric
contract from the spec ("StyleProfile Schema") against a *known* synthetic
fixture corpus whose word/sentence counts are hand-computed, so the asserted
values are exact rather than approximate.

Fixture corpus (``tests/fixtures/corpus/``), 4 documents, 3 sentences each:

    doc1.txt          We ship code today. / Tests keep us honest. /
                      Plans stay quite simple.
    doc2.txt          We ship code today. / The plan stays simple. /
                      We ship code daily.
    doc3.txt          We ship code often. / Plans change a lot. /
                      Honesty always wins out.
    doc4_sentinel.txt We ship code now. /
                      The zphinx quokka waltzed nimbly past my jovial trumpet
                      today. / Plans guide our work.

Eleven 4-word sentences plus one 10-word sentinel sentence => 12 sentences,
54 words. avg = 54/12 = 4.5, median = 4.0. The bigram "ship code" occurs five
times across all four documents (signature phrase); "zphinx quokka" occurs
once in one document only (privacy floor must drop it).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from write_like_me_mcp import style_analyzer  # noqa: E402
from write_like_me_mcp.model import SCHEMA_VERSION, StyleProfile  # noqa: E402

CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "corpus"


def _corpus_paths() -> list[Path]:
    """Return the fixture corpus files in a stable, sorted order."""
    return sorted(CORPUS_DIR.glob("*.txt"))


def _build() -> StyleProfile:
    """Build a profile from the full fixture corpus."""
    return style_analyzer.analyze_corpus(_corpus_paths(), profile_name="test")


def test_corpus_fixture_present() -> None:
    """The synthetic corpus exists with the four expected documents."""
    paths = _corpus_paths()
    names = {p.name for p in paths}
    assert names == {"doc1.txt", "doc2.txt", "doc3.txt", "doc4_sentinel.txt"}


def test_returns_styleprofile_with_metadata() -> None:
    """analyze_corpus returns a StyleProfile with privacy-safe metadata."""
    profile = _build()
    assert isinstance(profile, StyleProfile)
    md = profile.metadata
    assert md["profile_name"] == "test"
    assert md["doc_count"] == 4
    assert md["total_words"] == 54
    assert md["schema_version"] == SCHEMA_VERSION
    assert isinstance(md["generated_at"], str) and md["generated_at"]


def test_determinism_metric_fields_byte_equal_across_runs() -> None:
    """Identical corpus -> identical metric fields (byte-equal JSON).

    ``generated_at`` is the only non-deterministic field; it is excluded so the
    rest of the profile can be compared byte-for-byte across two runs.
    """
    a = _build()
    b = _build()

    da = a.to_dict()
    db = b.to_dict()
    da["metadata"].pop("generated_at", None)
    db["metadata"].pop("generated_at", None)

    import json

    assert json.dumps(da, sort_keys=True) == json.dumps(db, sort_keys=True)


def test_sentence_length_metrics_exact() -> None:
    """avg/median/distribution are exact on the known fixture."""
    profile = _build()
    assert profile.avg_sentence_length == 4.5
    assert profile.median_sentence_length == 4.0

    dist = profile.sentence_length_distribution
    # Percentile and bucket keys are part of the frozen distribution summary.
    assert dist["p10"] == 4
    assert dist["p50"] == 4
    assert dist["p90"] >= 4
    # All twelve sentences are <=12 words (the 10-word sentinel included), so
    # every sentence falls in the "short" bucket on this fixture.
    assert dist["short"] == 12
    assert dist["medium"] == 0
    assert dist["long"] == 0
    assert dist["short"] + dist["medium"] + dist["long"] == 12


def test_lexical_diversity_falls_back_to_ttr_for_small_corpus() -> None:
    """A corpus < MATTR_WINDOW tokens uses plain TTR and flags the fallback."""
    profile = _build()  # 54 tokens < 100-token MATTR window
    assert profile.metadata["ttr_fallback"] is True

    # Plain TTR == unique tokens / total tokens, deterministic and in (0, 1].
    assert 0.0 < profile.lexical_diversity <= 1.0


def test_mattr_used_when_corpus_large_enough() -> None:
    """A corpus >= MATTR_WINDOW tokens uses MATTR and does not flag fallback."""
    # Repeat the fixture text enough to exceed the 100-token window.
    text = ("we ship code today. " * 60).strip()
    paths = []
    profile = style_analyzer.analyze_text(text, profile_name="big", source_paths=paths)
    assert profile.metadata["ttr_fallback"] is False
    assert 0.0 < profile.lexical_diversity <= 1.0


def test_punctuation_and_rate_metrics_exact() -> None:
    """punctuation-per-1k, contraction_rate, passive_voice_rate are exact."""
    # 12 sentences, each ends with '.', so 12 periods are NOT in the tracked
    # set; the tracked set is em-dash/semicolon/comma/exclamation/question/
    # ellipsis/parentheses. The fixture has none of those.
    profile = _build()
    punct = profile.punctuation_per_1k
    for key in (
        "em_dash",
        "semicolon",
        "comma",
        "exclamation",
        "question",
        "ellipsis",
        "parentheses",
    ):
        assert key in punct
        assert punct[key] == 0.0

    # No contractions in the fixture.
    assert profile.contraction_rate == 0.0

    # No be-verb + past participle constructions in the fixture.
    assert profile.passive_voice_rate == 0.0


def test_punctuation_counts_known_marks() -> None:
    """Punctuation per-1k is computed against an explicitly-counted text."""
    # 1000 words so the per-1k rate equals the raw count exactly.
    body = ("word " * 1000).strip()
    text = body + " , ; ! ? -- ( )"
    profile = style_analyzer.analyze_text(text, profile_name="punct", source_paths=[])
    p = profile.punctuation_per_1k
    # words: 1000 base + tokens for the symbols may or may not count as words;
    # assert presence of each mark's contribution is non-zero where added.
    assert p["comma"] > 0.0
    assert p["semicolon"] > 0.0
    assert p["exclamation"] > 0.0
    assert p["question"] > 0.0
    assert p["em_dash"] > 0.0
    assert p["parentheses"] > 0.0


def test_passive_voice_detected() -> None:
    """A clearly passive sentence is counted; an active one is not."""
    passive = "The report was written. The code was reviewed. The bug was fixed."
    active = "She wrote the report. He reviews code. They fix bugs."

    p_passive = style_analyzer.analyze_text(passive, profile_name="p", source_paths=[])
    p_active = style_analyzer.analyze_text(active, profile_name="a", source_paths=[])

    assert p_passive.passive_voice_rate > 0.0
    assert p_active.passive_voice_rate == 0.0


def test_contraction_rate_detected() -> None:
    """Contractions raise contraction_rate above zero."""
    text = "I can't go. We won't wait. They don't know. It isn't ready."
    profile = style_analyzer.analyze_text(text, profile_name="c", source_paths=[])
    assert profile.contraction_rate > 0.0


def test_signature_phrases_are_capped_bi_and_tri_grams() -> None:
    """signature_phrases contains only <=3-token, dispersion-floored n-grams."""
    profile = _build()
    phrases = profile.signature_phrases
    assert isinstance(phrases, list)
    assert len(phrases) >= 1

    for entry in phrases:
        # Each entry carries the phrase plus its frequency (spec: "with
        # frequency"). Length cap: never more than 3 tokens.
        phrase = entry["phrase"] if isinstance(entry, dict) else entry
        token_count = len(phrase.split())
        assert 2 <= token_count <= style_analyzer.NGRAM_MAX

    # "ship code" appears 5x across all 4 docs -> must be present.
    surfaced = {
        (e["phrase"] if isinstance(e, dict) else e) for e in phrases
    }
    assert "ship code" in surfaced


def test_style_guide_md_non_empty_and_reflects_short_sentences() -> None:
    """style_guide_md is non-empty prose reflecting the short-sentence corpus."""
    profile = _build()
    guide = profile.style_guide_md
    assert isinstance(guide, str)
    assert guide.strip()
    # median sentence length is 4 -> the deterministic template should describe
    # short, punchy sentences.
    assert "short" in guide.lower()


def test_metadata_uses_basenames_not_absolute_paths() -> None:
    """metadata records source basenames only -- never absolute paths."""
    profile = _build()
    import json

    raw = profile.to_json()
    assert "/Users/" not in raw
    assert str(REPO_ROOT) not in raw

    parsed = json.loads(raw)
    sources = parsed["metadata"].get("sources", [])
    for s in sources:
        assert "/" not in s and "\\" not in s


def test_formality_markers_present() -> None:
    """formality_markers carries first/second-person and hedging rates."""
    profile = _build()
    fm = profile.formality_markers
    for key in ("first_person_rate", "second_person_rate", "hedging_rate"):
        assert key in fm
        assert isinstance(fm[key], float)
