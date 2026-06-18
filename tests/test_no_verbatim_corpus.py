"""OSS-safety criterion 6: no verbatim corpus text persisted in the profile.

This is the C1 privacy invariant. It builds a StyleProfile from the fixture
corpus -- which contains a unique sentinel sentence and a unique
single-occurrence phrase -- and asserts the persisted, potentially shareable
``profile.json`` leaks neither:

  (a) the sentinel sentence appears NOWHERE in the serialized profile JSON;
  (b) the unique single-occurrence phrase is NOT in ``signature_phrases``
      (it fails the document-dispersion floor);
  (c) no signature phrase exceeds 3 tokens (so no sentence-length span is
      ever stored).

See spec "No verbatim corpus text in the persisted profile" and OSS-safety
criterion 6.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from write_like_me_mcp import style_analyzer  # noqa: E402

CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "corpus"

# The unique sentinel sentence and unique single-occurrence phrase planted in
# doc4_sentinel.txt. Neither must ever surface in the persisted profile.
SENTINEL_SENTENCE = (
    "The zphinx quokka waltzed nimbly past my jovial trumpet today."
)
SINGLE_OCCURRENCE_PHRASE = "zphinx quokka"


def _build_profile_json() -> str:
    paths = sorted(CORPUS_DIR.glob("*.txt"))
    profile = style_analyzer.analyze_corpus(paths, profile_name="privacy")
    return profile.to_json()


def test_sentinel_sentence_absent_from_serialized_profile() -> None:
    """(a) The sentinel sentence appears nowhere in profile.json."""
    raw = _build_profile_json()
    assert SENTINEL_SENTENCE not in raw
    # Even a substantial fragment of it must not be present verbatim.
    assert "waltzed nimbly past" not in raw
    assert "jovial trumpet" not in raw


def test_single_occurrence_phrase_not_in_signature_phrases() -> None:
    """(b) A once-occurring phrase fails the dispersion floor and is dropped."""
    paths = sorted(CORPUS_DIR.glob("*.txt"))
    profile = style_analyzer.analyze_corpus(paths, profile_name="privacy")

    surfaced = {
        (e["phrase"] if isinstance(e, dict) else e)
        for e in profile.signature_phrases
    }
    assert SINGLE_OCCURRENCE_PHRASE not in surfaced
    # No surfaced phrase should even contain the unique sentinel tokens.
    for phrase in surfaced:
        assert "zphinx" not in phrase
        assert "quokka" not in phrase


def test_no_signature_phrase_exceeds_three_tokens() -> None:
    """(c) Every signature phrase is at most 3 tokens long."""
    paths = sorted(CORPUS_DIR.glob("*.txt"))
    profile = style_analyzer.analyze_corpus(paths, profile_name="privacy")

    for entry in profile.signature_phrases:
        phrase = entry["phrase"] if isinstance(entry, dict) else entry
        assert len(phrase.split()) <= style_analyzer.NGRAM_MAX <= 3


def test_no_full_sentence_appears_in_profile() -> None:
    """No multi-word fixture sentence is stored verbatim in the profile."""
    raw = _build_profile_json()
    # Sample 4+ word spans from the fixture that must not appear verbatim.
    for span in (
        "Tests keep us honest",
        "Plans stay quite simple",
        "Plans change a lot",
        "Plans guide our work",
    ):
        assert span not in raw
