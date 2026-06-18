"""Deterministic, stdlib-only writing-style analyzer for write-like-me-mcp.

This module computes a :class:`~write_like_me_mcp.model.StyleProfile` from a
user's document corpus using **only the standard library and regular
expressions** — no NLP frameworks, no network, no LLM. Every metric is a
deterministic, pure function of the input text: identical corpora yield
byte-identical metric fields (the lone exception is ``metadata.generated_at``,
a timestamp).

Privacy invariants (C1 / OSS-safety criterion 6)
------------------------------------------------
The persisted profile stores **no verbatim corpus text**. It contains only:

* aggregate statistics (sentence/paragraph lengths, punctuation rates, etc.),
* capped, dispersion-floored signature phrases — bi- and tri-grams only
  (``<= NGRAM_MAX`` tokens), each appearing in ``>= MIN_DOC_DISPERSION``
  distinct documents with a total count ``>= MIN_PHRASE_COUNT``, so a unique
  once-occurring (potentially sensitive) phrase can never be surfaced, and
* deterministically templated prose (``style_guide_md``).

It also stores **no absolute filesystem paths**: ``metadata.sources`` holds
basenames only.

Public entry points
-------------------
* :func:`analyze_corpus` — read & extract a list of files, then build a profile.
* :func:`analyze_text` — build a profile directly from an in-memory string
  (used by tests and by callers that already hold extracted text).

The heuristic metrics (``passive_voice_rate``, ``contraction_rate``) are regex
approximations, **not** linguistic parsers; their docstrings say so.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .extractors import extract_text
from .model import SCHEMA_VERSION, StyleProfile

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Frozen module constants (no inline magic numbers).
# --------------------------------------------------------------------------- #

#: Fixed window size (in tokens) for the moving-average type-token ratio.
MATTR_WINDOW: int = 100

#: Default minimum number of *distinct documents* a phrase must appear in to be
#: eligible as a signature phrase (the C1 privacy dispersion floor).
MIN_DOC_DISPERSION: int = 3

#: Relaxed dispersion floor used when the corpus has fewer than this many docs.
SMALL_CORPUS_DOC_THRESHOLD: int = 3

#: Dispersion floor applied to small corpora (< ``SMALL_CORPUS_DOC_THRESHOLD``).
SMALL_CORPUS_DISPERSION: int = 2

#: Minimum total occurrences (across the corpus) for a signature phrase.
MIN_PHRASE_COUNT: int = 3

#: Maximum n-gram length (in tokens) for signature phrases. Capped at 3 so no
#: sentence-length verbatim span is ever stored.
NGRAM_MAX: int = 3

#: Number of top signature phrases retained in the profile.
MAX_SIGNATURE_PHRASES: int = 25

#: Word-count boundaries for the sentence-length distribution buckets.
SHORT_SENTENCE_MAX_WORDS: int = 12
MEDIUM_SENTENCE_MAX_WORDS: int = 25

#: Median-sentence-length threshold below which the style guide calls the
#: voice "short, punchy".
SHORT_VOICE_MEDIAN_THRESHOLD: float = 12.0
#: Median above which the style guide calls the voice "long, flowing".
LONG_VOICE_MEDIAN_THRESHOLD: float = 25.0

#: Be-verbs used by the passive-voice heuristic.
BE_VERBS: frozenset[str] = frozenset(
    {"am", "is", "are", "was", "were", "be", "been", "being"}
)

#: A small irregular past-participle list, since the regex ``\w+(ed|en)\b``
#: cannot catch participles that do not end in -ed/-en.
IRREGULAR_PARTICIPLES: frozenset[str] = frozenset(
    {
        "done",
        "made",
        "said",
        "told",
        "kept",
        "sent",
        "built",
        "bought",
        "brought",
        "caught",
        "taught",
        "found",
        "held",
        "left",
        "lost",
        "meant",
        "paid",
        "put",
        "read",
        "set",
        "shown",
        "sung",
        "thrown",
        "won",
        "written",
        "given",
        "taken",
        "drawn",
        "known",
        "grown",
        "begun",
        "chosen",
        "broken",
        "spoken",
    }
)

#: Stopwords filtered out of signature-phrase candidates. A phrase is dropped
#: if it is composed *entirely* of stopwords.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
        "of", "to", "in", "on", "at", "by", "for", "with", "about", "as",
        "into", "from", "up", "down", "out", "over", "under", "again",
        "is", "am", "are", "was", "were", "be", "been", "being", "do",
        "does", "did", "have", "has", "had", "will", "would", "shall",
        "should", "can", "could", "may", "might", "must", "i", "you",
        "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "this", "that",
        "these", "those", "there", "here", "what", "which", "who", "whom",
        "not", "no", "yes", "so", "too", "very", "just", "than", "such",
    }
)

#: First-person pronouns for the formality markers.
FIRST_PERSON: frozenset[str] = frozenset(
    {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
)

#: Second-person pronouns for the formality markers.
SECOND_PERSON: frozenset[str] = frozenset(
    {"you", "your", "yours", "yourself", "yourselves"}
)

#: Hedging words for the formality markers.
HEDGING_WORDS: frozenset[str] = frozenset(
    {
        "maybe", "perhaps", "possibly", "probably", "likely", "seems",
        "seem", "appears", "appear", "somewhat", "rather", "fairly",
        "arguably", "presumably", "apparently", "roughly", "approximately",
        "kind", "sort", "guess", "suppose", "think", "believe", "might",
        "could", "may", "tend", "generally", "usually", "often",
    }
)

#: Percentile points reported in the sentence-length distribution summary.
_PERCENTILES: tuple[int, ...] = (10, 50, 90)

#: Scale factor for "per 1,000" rate metrics.
PER_1K: int = 1000

# --------------------------------------------------------------------------- #
# Regex patterns (compiled once).
# --------------------------------------------------------------------------- #

#: Word token: runs of letters/digits, allowing internal apostrophes so that
#: contractions ("can't", "won't") survive as single tokens.
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

#: Sentence terminators. Sentences are split on ., !, ? (one or more), keeping
#: this deterministic and dependency-free.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

#: Paragraph separator: one or more blank lines.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

#: Past-participle approximation from the spec: ``\w+(ed|en)\b``.
_PARTICIPLE_RE = re.compile(r"^\w+(ed|en)$", re.IGNORECASE)

#: Contraction detection: an apostrophe between letters (don't, we'll, it's).
_CONTRACTION_RE = re.compile(r"\b[A-Za-z]+'[A-Za-z]+\b")

#: Punctuation marks tracked for the per-1k rates. Order is deterministic.
#: Em-dash matches a literal em-dash or a double hyphen "--".
_PUNCTUATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("em_dash", r"—|--"),
    ("semicolon", r";"),
    ("comma", r","),
    ("exclamation", r"!"),
    ("question", r"\?"),
    ("ellipsis", r"…|\.\.\."),
    ("parentheses", r"[()]"),
)


# --------------------------------------------------------------------------- #
# Tokenization / segmentation.
# --------------------------------------------------------------------------- #


def _tokenize_words(text: str) -> list[str]:
    """Return lowercased word tokens (contractions preserved)."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into non-empty, stripped sentences.

    Splitting is purely on terminal punctuation (``.!?``) and is deterministic.
    Empty fragments (e.g. trailing whitespace) are discarded.
    """
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split ``text`` into non-empty paragraphs on blank lines."""
    parts = _PARAGRAPH_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _percentile(sorted_values: list[int], pct: int) -> float:
    """Return the ``pct``-th percentile of ``sorted_values`` (nearest-rank).

    Uses the nearest-rank method for full determinism. ``sorted_values`` must
    be sorted ascending and non-empty.
    """
    if not sorted_values:
        return 0.0
    rank = max(1, -(-pct * len(sorted_values) // 100))  # ceil(pct/100 * n)
    return float(sorted_values[rank - 1])


# --------------------------------------------------------------------------- #
# Individual metric computations.
# --------------------------------------------------------------------------- #


def _sentence_length_metrics(sentence_word_counts: list[int]) -> tuple[float, float, dict]:
    """Compute (avg, median, distribution) from per-sentence word counts."""
    if not sentence_word_counts:
        return 0.0, 0.0, {
            "p10": 0, "p50": 0, "p90": 0,
            "short": 0, "medium": 0, "long": 0,
        }

    counts = sorted(sentence_word_counts)
    n = len(counts)
    avg = round(sum(counts) / n, 4)

    mid = n // 2
    if n % 2 == 1:
        median = float(counts[mid])
    else:
        median = round((counts[mid - 1] + counts[mid]) / 2, 4)

    short = sum(1 for c in counts if c <= SHORT_SENTENCE_MAX_WORDS)
    medium = sum(
        1 for c in counts
        if SHORT_SENTENCE_MAX_WORDS < c <= MEDIUM_SENTENCE_MAX_WORDS
    )
    long_ = sum(1 for c in counts if c > MEDIUM_SENTENCE_MAX_WORDS)

    distribution = {
        "p10": _percentile(counts, 10),
        "p50": _percentile(counts, 50),
        "p90": _percentile(counts, 90),
        "short": short,
        "medium": medium,
        "long": long_,
    }
    return avg, median, distribution


def _lexical_diversity(tokens: list[str]) -> tuple[float, bool]:
    """Return (diversity, ttr_fallback).

    Uses MATTR (moving-average type-token ratio) over a fixed
    ``MATTR_WINDOW``-token window. If the corpus has fewer than
    ``MATTR_WINDOW`` tokens, falls back to plain TTR and returns
    ``ttr_fallback=True``.
    """
    n = len(tokens)
    if n == 0:
        return 0.0, True
    if n < MATTR_WINDOW:
        ttr = len(set(tokens)) / n
        return round(ttr, 6), True

    window_ttrs: list[float] = []
    for start in range(0, n - MATTR_WINDOW + 1):
        window = tokens[start:start + MATTR_WINDOW]
        window_ttrs.append(len(set(window)) / MATTR_WINDOW)
    mattr = sum(window_ttrs) / len(window_ttrs)
    return round(mattr, 6), False


def _punctuation_per_1k(text: str, total_words: int) -> dict:
    """Return per-1k-word rates for the tracked punctuation marks."""
    rates: dict[str, float] = {}
    denom = total_words if total_words > 0 else 1
    for name, pattern in _PUNCTUATION_PATTERNS:
        count = len(re.findall(pattern, text))
        rates[name] = round(count / denom * PER_1K, 4)
    return rates


def _contraction_rate(tokens: list[str], total_words: int) -> float:
    """Return contractions per 1k words (regex approximation, not a parser)."""
    denom = total_words if total_words > 0 else 1
    contractions = sum(1 for t in tokens if "'" in t)
    return round(contractions / denom * PER_1K, 4)


def _is_participle(token: str) -> bool:
    """Heuristic: does ``token`` look like a past participle?

    Approximation per spec: matches ``\\w+(ed|en)`` or a known irregular
    participle. Not a linguistic parser.
    """
    low = token.lower()
    if low in IRREGULAR_PARTICIPLES:
        return True
    return bool(_PARTICIPLE_RE.match(low))


def _passive_voice_rate(sentences: list[str]) -> float:
    """Return heuristic passive constructions per 1k sentences.

    Heuristic (approximation, not a parser): a sentence is counted as passive
    if it contains a be-verb immediately (or near-immediately, allowing one
    intervening adverb) followed by a token that looks like a past participle.
    Expressed per 1,000 sentences.
    """
    if not sentences:
        return 0.0

    passive_hits = 0
    for sentence in sentences:
        words = _tokenize_words(sentence)
        for i, word in enumerate(words):
            if word not in BE_VERBS:
                continue
            # be-verb directly followed by a participle, or with one
            # intervening adverb-like token.
            for offset in (1, 2):
                j = i + offset
                if j < len(words) and _is_participle(words[j]):
                    passive_hits += 1
                    break
            else:
                continue
            break
    return round(passive_hits / len(sentences) * PER_1K, 4)


def _formality_markers(tokens: list[str], total_words: int) -> dict:
    """Return first/second-person and hedging rates (per 1k words)."""
    denom = total_words if total_words > 0 else 1
    first = sum(1 for t in tokens if t in FIRST_PERSON)
    second = sum(1 for t in tokens if t in SECOND_PERSON)
    hedging = sum(1 for t in tokens if t in HEDGING_WORDS)
    return {
        "first_person_rate": round(first / denom * PER_1K, 4),
        "second_person_rate": round(second / denom * PER_1K, 4),
        "hedging_rate": round(hedging / denom * PER_1K, 4),
    }


# --------------------------------------------------------------------------- #
# Signature phrases (the C1 privacy-bounded extractor).
# --------------------------------------------------------------------------- #


def _phrase_is_all_stopwords(tokens: tuple[str, ...]) -> bool:
    """Return True if every token in the phrase is a stopword."""
    return all(t in STOPWORDS for t in tokens)


def _signature_phrases(per_doc_tokens: list[list[str]]) -> list[dict]:
    """Extract dispersion-floored, <=3-token signature phrases.

    For each document we generate bi- and tri-grams (``2..NGRAM_MAX`` tokens),
    drop phrases composed entirely of stopwords, and track both total
    corpus frequency and the number of *distinct documents* a phrase appears
    in. A phrase qualifies only if:

    * it appears in ``>= dispersion_floor`` distinct documents, **and**
    * its total count ``>= MIN_PHRASE_COUNT``.

    Qualifying phrases are ranked by ``frequency * document_dispersion``
    (ties broken alphabetically for determinism) and the top
    ``MAX_SIGNATURE_PHRASES`` are returned as ``{"phrase", "frequency"}``
    dicts. The length cap (``NGRAM_MAX``) guarantees no sentence-length
    verbatim span is ever stored.
    """
    doc_count = len(per_doc_tokens)
    dispersion_floor = (
        SMALL_CORPUS_DISPERSION
        if doc_count < SMALL_CORPUS_DOC_THRESHOLD
        else MIN_DOC_DISPERSION
    )

    total_counts: Counter[tuple[str, ...]] = Counter()
    doc_dispersion: Counter[tuple[str, ...]] = Counter()

    for tokens in per_doc_tokens:
        seen_in_doc: set[tuple[str, ...]] = set()
        for size in range(2, NGRAM_MAX + 1):
            for start in range(0, len(tokens) - size + 1):
                gram = tuple(tokens[start:start + size])
                if _phrase_is_all_stopwords(gram):
                    continue
                total_counts[gram] += 1
                seen_in_doc.add(gram)
        for gram in seen_in_doc:
            doc_dispersion[gram] += 1

    qualifying: list[tuple[tuple[str, ...], int, int]] = []
    for gram, freq in total_counts.items():
        docs = doc_dispersion[gram]
        if docs >= dispersion_floor and freq >= MIN_PHRASE_COUNT:
            qualifying.append((gram, freq, docs))

    # Rank by frequency * dispersion (desc), then alphabetically for stability.
    qualifying.sort(key=lambda item: (-(item[1] * item[2]), " ".join(item[0])))

    return [
        {"phrase": " ".join(gram), "frequency": freq}
        for gram, freq, _docs in qualifying[:MAX_SIGNATURE_PHRASES]
    ]


# --------------------------------------------------------------------------- #
# Style-guide templating (deterministic; no LLM).
# --------------------------------------------------------------------------- #


def _build_style_guide(
    avg_sentence_length: float,
    median_sentence_length: float,
    lexical_diversity: float,
    contraction_rate: float,
    passive_voice_rate: float,
    formality_markers: dict,
    signature_phrases: list[dict],
) -> str:
    """Distill the metrics into deterministic Markdown prose (no LLM).

    Thresholds map to fixed phrasings so identical metrics always produce
    identical guides. Contains only aggregate descriptions and the already
    capped/floored signature phrases — never verbatim corpus sentences.
    """
    lines: list[str] = ["# Writing Style Guide", ""]

    # Sentence rhythm.
    if median_sentence_length < SHORT_VOICE_MEDIAN_THRESHOLD:
        rhythm = (
            "Prefer **short, punchy sentences**. The author's typical sentence "
            f"runs about {median_sentence_length:g} words."
        )
    elif median_sentence_length > LONG_VOICE_MEDIAN_THRESHOLD:
        rhythm = (
            "Favor **long, flowing sentences**. The author's typical sentence "
            f"runs about {median_sentence_length:g} words."
        )
    else:
        rhythm = (
            "Use **medium-length sentences**. The author's typical sentence "
            f"runs about {median_sentence_length:g} words."
        )
    lines.append("## Sentence rhythm")
    lines.append(rhythm)
    lines.append(
        f"Average sentence length is {avg_sentence_length:g} words."
    )
    lines.append("")

    # Voice & formality.
    lines.append("## Voice")
    if contraction_rate > 0:
        lines.append(
            "Contractions are welcome — the voice is conversational."
        )
    else:
        lines.append("Avoid contractions — the voice is formal.")
    if passive_voice_rate > 0:
        lines.append(
            "Some passive constructions appear; prefer active voice but it is "
            "acceptable in moderation."
        )
    else:
        lines.append("Strongly prefer the **active voice**.")
    if formality_markers.get("first_person_rate", 0.0) > 0:
        lines.append("First-person voice is part of the author's style.")
    if formality_markers.get("hedging_rate", 0.0) > 0:
        lines.append("Light hedging language is characteristic.")
    lines.append("")

    # Lexical diversity.
    lines.append("## Vocabulary")
    lines.append(
        f"Lexical diversity (TTR/MATTR) is {lexical_diversity:g}."
    )
    lines.append("")

    # Signature phrases (already <=3 tokens, dispersion-floored).
    if signature_phrases:
        lines.append("## Characteristic phrases")
        for entry in signature_phrases[:10]:
            lines.append(f"- \"{entry['phrase']}\"")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Public assembly.
# --------------------------------------------------------------------------- #


def _assemble_profile(
    per_doc_texts: list[str],
    profile_name: str,
    source_basenames: list[str],
    source_count: int,
) -> StyleProfile:
    """Compute every metric from per-document texts and assemble a profile.

    No verbatim corpus text reaches the returned profile: only aggregates,
    capped/floored signature phrases, and templated prose.
    """
    full_text = "\n\n".join(per_doc_texts)

    all_tokens = _tokenize_words(full_text)
    total_words = len(all_tokens)

    sentences = _split_sentences(full_text)
    sentence_word_counts = [len(_tokenize_words(s)) for s in sentences]
    sentence_word_counts = [c for c in sentence_word_counts if c > 0]

    paragraphs = _split_paragraphs(full_text)
    if paragraphs:
        sentences_per_paragraph = [
            max(1, len(_split_sentences(p))) for p in paragraphs
        ]
        avg_paragraph_length = round(
            sum(sentences_per_paragraph) / len(sentences_per_paragraph), 4
        )
    else:
        avg_paragraph_length = 0.0

    avg_sentence, median_sentence, distribution = _sentence_length_metrics(
        sentence_word_counts
    )
    lexical_diversity, ttr_fallback = _lexical_diversity(all_tokens)
    punctuation = _punctuation_per_1k(full_text, total_words)
    contraction_rate = _contraction_rate(all_tokens, total_words)
    passive_voice_rate = _passive_voice_rate(sentences)
    formality_markers = _formality_markers(all_tokens, total_words)

    per_doc_tokens = [_tokenize_words(t) for t in per_doc_texts]
    signature_phrases = _signature_phrases(per_doc_tokens)

    style_guide_md = _build_style_guide(
        avg_sentence_length=avg_sentence,
        median_sentence_length=median_sentence,
        lexical_diversity=lexical_diversity,
        contraction_rate=contraction_rate,
        passive_voice_rate=passive_voice_rate,
        formality_markers=formality_markers,
        signature_phrases=signature_phrases,
    )

    metadata = {
        "profile_name": profile_name,
        "source_count": source_count,
        "doc_count": len(per_doc_texts),
        "total_words": total_words,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "ttr_fallback": ttr_fallback,
        "sources": source_basenames,
    }

    return StyleProfile(
        avg_sentence_length=avg_sentence,
        median_sentence_length=median_sentence,
        sentence_length_distribution=distribution,
        lexical_diversity=lexical_diversity,
        avg_paragraph_length=avg_paragraph_length,
        punctuation_per_1k=punctuation,
        contraction_rate=contraction_rate,
        passive_voice_rate=passive_voice_rate,
        signature_phrases=signature_phrases,
        formality_markers=formality_markers,
        style_guide_md=style_guide_md,
        metadata=metadata,
    )


def analyze_text(
    text: str,
    profile_name: str,
    source_paths: list[Path] | None = None,
) -> StyleProfile:
    """Build a :class:`StyleProfile` from a single in-memory text string.

    The whole string is treated as **one document** for dispersion purposes.
    Useful for tests and for callers that already hold extracted text.

    Args:
        text: The corpus text to analyze.
        profile_name: Name recorded in ``metadata.profile_name``.
        source_paths: Optional source paths; only their **basenames** are
            recorded (privacy invariant). May be empty/``None``.

    Returns:
        A computed :class:`StyleProfile` (no verbatim text persisted).
    """
    paths = source_paths or []
    basenames = [Path(p).name for p in paths]
    return _assemble_profile(
        per_doc_texts=[text],
        profile_name=profile_name,
        source_basenames=basenames,
        source_count=len(basenames),
    )


def analyze_corpus(file_paths: list[Path], profile_name: str) -> StyleProfile:
    """Extract text from ``file_paths`` and build a :class:`StyleProfile`.

    This is the primary entry point fed by the extractors: the server
    (``analyze_writing_style`` tool) and the file watcher call it to (re)build
    a corpus-global profile. Each readable file becomes one document for the
    signature-phrase dispersion floor.

    Files that fail extraction (``extract_text`` returns ``None``) or yield no
    text are skipped and logged to stderr; they still count toward
    ``source_count`` but not toward ``doc_count``.

    Args:
        file_paths: Corpus files to analyze (any extractor-supported format).
        profile_name: Name recorded in ``metadata.profile_name``.

    Returns:
        A computed :class:`StyleProfile`. The persisted profile contains only
        aggregate statistics, capped/floored signature phrases, and templated
        prose — never verbatim corpus text and never absolute paths.
    """
    per_doc_texts: list[str] = []
    basenames: list[str] = []

    for path in file_paths:
        path = Path(path)
        text = extract_text(path)
        if not text or not text.strip():
            logger.warning("No text extracted from %s; skipping", path.name)
            continue
        per_doc_texts.append(text)
        basenames.append(path.name)

    if not per_doc_texts:
        logger.warning(
            "analyze_corpus: no readable documents in %d source(s)",
            len(file_paths),
        )

    return _assemble_profile(
        per_doc_texts=per_doc_texts,
        profile_name=profile_name,
        source_basenames=basenames,
        source_count=len(file_paths),
    )
