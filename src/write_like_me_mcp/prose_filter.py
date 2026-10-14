"""Corpus admission gate: is this file human prose the analyzer can learn from?

Why this exists
---------------
The corpus walker admits any file whose suffix is in ``file_extensions``. A
suffix is not a guarantee of content. In the field this let four kinds of
non-writing into a profile built from ~7,300 files:

* a DRM'd Kindle book renamed ``.azw.md`` — raw binary, which surfaced as the
  author's characteristic phrases ``"x x"`` and ``"x x x"``;
* router-topology and grading fixtures of bare digits — ``"0 0"``;
* syslogs and word lists with no terminal punctuation, each read as a single
  4,000-word "sentence", which pulled the reported average sentence length to
  62 words against a median of 12;
* published books and other people's course notes, which are prose, but not
  *this author's* prose.

The first three are detectable from the text itself and are rejected here. The
fourth is an authorship question no heuristic can settle — it belongs in the
profile's ``exclude`` list, and :func:`is_prose` says nothing about it.

Every check is a deterministic, stdlib-only function of the text, in keeping
with the analyzer: same input, same verdict, no network, no model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Fewest words a document needs before it can characterize anything. A stub
#: ("Thanks. Will do.") is not evidence of a writing style.
MIN_PROSE_WORDS: int = 15

#: Share of non-whitespace characters that must be alphabetic. Measured over a
#: 500-file sample of the real corpus, ordinary writing sits at 0.83 (median);
#: data dumps, config files and topology fixtures sit below 0.15.
MIN_LETTER_RATIO: float = 0.55

#: Share of characters that may be control characters or U+FFFD before the
#: file is treated as binary. Legitimate text has none; the 95th percentile of
#: the sampled corpus is exactly 0.
MAX_BINARY_RATIO: float = 0.005

#: Words per sentence above which the text has no usable sentence structure.
#: The sampled corpus reaches 50 at the 95th percentile; a syslog reaches 794.
MAX_WORDS_PER_SENTENCE: float = 80.0

#: A token this long is not a word.
LONG_TOKEN_CHARS: int = 30

#: Share of tokens that may exceed :data:`LONG_TOKEN_CHARS` before the text is
#: treated as encoded (base64, hashes, minified data).
MAX_LONG_TOKEN_RATIO: float = 0.05

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

#: Control characters that never appear in prose (tab, newline, CR excepted).
_ALLOWED_CONTROL = frozenset("\n\r\t")


@dataclass(frozen=True)
class ProseVerdict:
    """The gate's decision, with the reason a rejected file was rejected.

    ``reason`` is one of ``too-short``, ``binary``, ``not-prose``, ``encoded``,
    ``unsegmented``, or ``None`` when the document is admitted. It is carried
    into the build summary so a corrupted corpus is visible rather than silent.
    """

    ok: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.ok


_ADMITTED = ProseVerdict(True, None)


def is_prose(text: str) -> ProseVerdict:
    """Return whether ``text`` is prose worth learning a writing style from.

    The checks run cheapest-and-most-certain first, so the reason attached to a
    rejection names the strongest thing wrong with the file.

    Args:
        text: The already-extracted document text.

    Returns:
        A :class:`ProseVerdict`. Truthy when the document is admitted.
    """
    if not text or not text.strip():
        return ProseVerdict(False, "too-short")

    words = _WORD_RE.findall(text)
    if len(words) < MIN_PROSE_WORDS:
        return ProseVerdict(False, "too-short")

    binary_chars = sum(
        1 for c in text if c == "�" or (ord(c) < 32 and c not in _ALLOWED_CONTROL)
    )
    if binary_chars / len(text) > MAX_BINARY_RATIO:
        return ProseVerdict(False, "binary")

    non_space = [c for c in text if not c.isspace()]
    if non_space:
        letters = sum(1 for c in non_space if c.isalpha())
        if letters / len(non_space) < MIN_LETTER_RATIO:
            return ProseVerdict(False, "not-prose")

    long_tokens = sum(1 for w in words if len(w) > LONG_TOKEN_CHARS)
    if long_tokens / len(words) > MAX_LONG_TOKEN_RATIO:
        return ProseVerdict(False, "encoded")

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences or len(words) / len(sentences) > MAX_WORDS_PER_SENTENCE:
        return ProseVerdict(False, "unsegmented")

    return _ADMITTED
