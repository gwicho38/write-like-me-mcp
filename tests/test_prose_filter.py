"""The corpus admission gate: only human prose the analyzer can learn from.

Regression suite for the corrupted-profile bug. The corpus walker admitted any
file with a .md/.txt suffix, so a DRM'd ebook renamed .md, a syslog, a router
topology dump and eight copies of Moby Dick all fed the style profile. The
resulting profile reported an average sentence of 62 words against a median of
12, and offered "x x", "x x x" and "0 0" as the author's characteristic phrases.
"""

from __future__ import annotations

import pytest

from write_like_me_mcp.prose_filter import (
    MIN_LETTER_RATIO,
    MIN_PROSE_WORDS,
    is_prose,
)

PROSE = (
    "I spoke to the landlord this morning and left a voicemail. He has not "
    "called back yet. The buyer sent three years of financials last week, so "
    "the delay is not on our side. I will chase him again on Wednesday, and if "
    "he stays quiet we send the notice by courier and start the clock in "
    "writing. That is the part that matters. Everything else is conversation."
)


def test_ordinary_prose_is_admitted():
    verdict = is_prose(PROSE)
    assert verdict.ok, verdict.reason
    assert verdict.reason is None


def test_binary_masquerading_as_markdown_is_rejected():
    """A Kindle .azw renamed .md — the real file that poisoned the profile."""
    blob = "\x00\x01��\x1b" * 500 + "X-X-X-X-X-X-X-X-X-X-" * 200
    verdict = is_prose(blob)
    assert not verdict.ok
    assert verdict.reason in {"binary", "not-prose"}


def test_replacement_characters_are_rejected():
    verdict = is_prose("�" * 200 + PROSE)
    assert not verdict.ok
    assert verdict.reason == "binary"


def test_data_dump_without_letters_is_rejected():
    """Router topology / matrix dumps: digits and delimiters, no language."""
    verdict = is_prose("\n".join("0 0 1 0 1 1 0 0 1 0" for _ in range(200)))
    assert not verdict.ok
    assert verdict.reason == "not-prose"


def test_unsegmented_text_is_rejected():
    """No terminal punctuation means no sentences to measure."""
    verdict = is_prose(" ".join(["word"] * 4000))
    assert not verdict.ok
    assert verdict.reason == "unsegmented"


def test_base64_blob_is_rejected():
    verdict = is_prose("\n".join(["QWxhZGRpbjpvcGVuIHNlc2FtZQ" * 4] * 100))
    assert not verdict.ok
    assert verdict.reason in {"not-prose", "encoded", "unsegmented"}


def test_too_short_to_characterize_is_rejected():
    verdict = is_prose("Thanks. Will do.")
    assert not verdict.ok
    assert verdict.reason == "too-short"


def test_minimum_length_boundary_is_inclusive():
    text = ("The quick brown fox jumps over the lazy dog. " * 20).strip()
    assert len(text.split()) >= MIN_PROSE_WORDS
    assert is_prose(text).ok


def test_markdown_with_code_and_tables_still_counts_as_prose():
    """A working note is not a data dump; the gate must not eat real writing."""
    note = (
        "# Landlord call\n\n"
        "I called Scott and left a voicemail. He has the financials already.\n\n"
        "| Party | Role |\n|---|---|\n| Kelly | landlord |\n\n"
        "```\ngit log --oneline -5\n```\n\n"
        "The next step is the courier letter. I will draft it tomorrow and send "
        "it once David signs. Nothing goes out before that.\n"
    )
    verdict = is_prose(note)
    assert verdict.ok, verdict.reason


def test_letter_ratio_threshold_is_documented_and_sane():
    assert 0.4 < MIN_LETTER_RATIO < 0.7


def test_verdict_is_falsy_when_rejected_and_truthy_when_admitted():
    assert is_prose(PROSE)
    assert not is_prose("0 0 0 0 " * 200)


@pytest.mark.parametrize("empty", ["", "   ", "\n\n\t"])
def test_empty_input_is_rejected_without_raising(empty):
    verdict = is_prose(empty)
    assert not verdict.ok
    assert verdict.reason == "too-short"
