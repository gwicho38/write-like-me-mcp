"""Tests for the StyleProfile model and its JSON persistence (model.py).

These tests are written before the implementation (TDD). They assert the
StyleProfile dataclass contract from the spec's frozen schema:

- lossless save -> load round-trip (all frozen fields, including nested objects)
- schema_version and ISO8601 generated_at present in metadata
- profile.json is human-readable (indented JSON)
- loading a missing profile.json returns None (drives a ``not_built`` status)
- the model never serializes absolute paths (privacy invariant)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from write_like_me_mcp.model import (  # noqa: E402
    SCHEMA_VERSION,
    StyleProfile,
)


def _sample_metadata() -> dict:
    """Return a representative, privacy-safe metadata dict."""
    return {
        "profile_name": "default",
        "source_count": 2,
        "doc_count": 5,
        "total_words": 12345,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
    }


def _sample_profile() -> StyleProfile:
    """Construct a fully-populated StyleProfile with nested objects/arrays."""
    return StyleProfile(
        avg_sentence_length=18.4,
        median_sentence_length=17.0,
        sentence_length_distribution={"short": 0.2, "medium": 0.5, "long": 0.3},
        lexical_diversity=0.62,
        avg_paragraph_length=4.1,
        punctuation_per_1k={"comma": 55.2, "semicolon": 3.1, "emdash": 1.7},
        contraction_rate=0.21,
        passive_voice_rate=0.08,
        signature_phrases=["to be clear", "in practice", "the upshot is"],
        formality_markers={"formal": 0.4, "informal": 0.6},
        style_guide_md="# Style Guide\n\n- Prefer active voice.\n- Use em-dashes sparingly.\n",
        metadata=_sample_metadata(),
    )


def test_save_load_roundtrip_is_lossless(tmp_path: Path) -> None:
    """A saved profile loads back with every frozen field preserved exactly."""
    original = _sample_profile()
    path = tmp_path / "profile.json"

    original.save(path)
    restored = StyleProfile.load(path)

    assert restored == original
    # Nested structures must survive the round-trip with correct types.
    assert restored.sentence_length_distribution == original.sentence_length_distribution
    assert restored.punctuation_per_1k == original.punctuation_per_1k
    assert restored.signature_phrases == original.signature_phrases
    assert restored.formality_markers == original.formality_markers
    assert restored.metadata == original.metadata
    assert isinstance(restored.signature_phrases, list)
    assert isinstance(restored.sentence_length_distribution, dict)


def test_metadata_has_schema_version_and_iso8601_generated_at(tmp_path: Path) -> None:
    """metadata carries schema_version and an ISO8601-parseable generated_at."""
    profile = _sample_profile()
    path = tmp_path / "profile.json"
    profile.save(path)

    restored = StyleProfile.load(path)
    assert restored is not None

    assert restored.metadata["schema_version"] == SCHEMA_VERSION
    assert SCHEMA_VERSION == 1

    generated_at = restored.metadata["generated_at"]
    # Must be a string that parses as ISO8601 (raises ValueError otherwise).
    parsed = datetime.fromisoformat(generated_at)
    assert isinstance(parsed, datetime)


def test_profile_json_is_human_readable(tmp_path: Path) -> None:
    """profile.json is written as indented, multi-line JSON."""
    profile = _sample_profile()
    path = tmp_path / "profile.json"
    profile.save(path)

    raw = path.read_text(encoding="utf-8")
    # Indented JSON spans multiple lines and contains indentation whitespace.
    assert "\n" in raw
    assert "\n  " in raw or "\n    " in raw
    # It must be valid JSON.
    parsed = json.loads(raw)
    assert parsed["avg_sentence_length"] == profile.avg_sentence_length


def test_load_missing_path_returns_none(tmp_path: Path) -> None:
    """Loading a non-existent profile.json returns None rather than raising."""
    missing = tmp_path / "does_not_exist" / "profile.json"
    assert StyleProfile.load(missing) is None


def test_to_json_and_from_json_roundtrip() -> None:
    """to_json/from_json round-trip in-memory without touching the filesystem."""
    original = _sample_profile()
    payload = original.to_json()
    assert isinstance(payload, str)

    restored = StyleProfile.from_json(payload)
    assert restored == original


def test_save_does_not_serialize_absolute_paths(tmp_path: Path) -> None:
    """The serialized profile contains no absolute filesystem paths (privacy)."""
    profile = _sample_profile()
    path = tmp_path / "profile.json"
    profile.save(path)

    raw = path.read_text(encoding="utf-8")
    # No POSIX-absolute or Windows-absolute path markers should appear.
    assert "/Users/" not in raw
    assert "/home/" not in raw
    assert ":\\" not in raw
    # Sanity: the dataclass itself does not inject any absolute path on save.
    parsed = json.loads(raw)
    for value in parsed.get("metadata", {}).values():
        if isinstance(value, str):
            assert not value.startswith("/"), f"absolute path leaked: {value}"


def test_non_ascii_preserved(tmp_path: Path) -> None:
    """ensure_ascii=False keeps non-ASCII characters readable and lossless."""
    profile = StyleProfile(
        avg_sentence_length=10.0,
        median_sentence_length=9.0,
        sentence_length_distribution={},
        lexical_diversity=0.5,
        avg_paragraph_length=3.0,
        punctuation_per_1k={},
        contraction_rate=0.1,
        passive_voice_rate=0.05,
        signature_phrases=["café déjà vu", "naïve résumé"],
        formality_markers={},
        style_guide_md="Prefer “smart quotes” and em—dashes.",
        metadata=_sample_metadata(),
    )
    path = tmp_path / "profile.json"
    profile.save(path)

    raw = path.read_text(encoding="utf-8")
    assert "café déjà vu" in raw
    assert "“smart quotes”" in raw

    restored = StyleProfile.load(path)
    assert restored is not None
    assert restored.signature_phrases == profile.signature_phrases
    assert restored.style_guide_md == profile.style_guide_md
