"""Tests for write_like_me_mcp.indexer — the whole-excerpt FTS5 retrieval index.

These tests exercise the public surface of ``ExcerptIndex``:
    - index build + FTS5 MATCH + BM25 returns expected excerpts for a query
    - excerpts are coherent whole paragraphs (NOT 1500-char search chunks)
    - per-source diversity cap (results not dominated by one document)
    - result shape: ``{excerpt, source_filename, score}`` — basenames, never abs paths
    - ``max_results`` clamped to bounds (min 1 / max 20)
    - incremental refresh via ``file_meta`` change-detection skips unchanged files

The indexer accepts already-extracted ``(path, text)`` pairs (extraction lives in
``extractors.py``); it never reads files itself, so these tests pass plain strings.
"""

from __future__ import annotations

import os

import pytest

from write_like_me_mcp.indexer import (
    MAX_EXCERPTS_PER_SOURCE,
    MAX_RESULTS_CEILING,
    MIN_RESULTS_FLOOR,
    ExcerptIndex,
)


# ---------------------------------------------------------------------------
# Fixtures (small inline corpus — no shared fixture files with sibling groups)
# ---------------------------------------------------------------------------

# Each "document" is several blank-line-separated paragraphs. The paragraphs are
# long enough that a 1500-char overlapping chunker would NOT split them, and we
# assert excerpts come back as whole coherent paragraphs.

DOC_GARDEN = """\
The morning frost clung to every blade of grass in the quiet garden.

Tomatoes ripen best when the soil stays warm and the watering is steady, so I
mulch heavily in early summer to hold the moisture through the long dry weeks.

Basil and rosemary share a bed near the south wall where the afternoon light
lingers, and the bees find them before I do almost every single morning.
"""

DOC_OCEAN = """\
The tide pulled back at dawn and left the dark sand glittering with shells.

Sailing close to the wind demands patience and a steady hand on the tiller,
because the smallest gust can spin an inattentive boat right around.

Salt air and the cry of gulls follow every harbor I have ever loved, and they
follow me home long after the voyage is over.
"""

DOC_CODE = """\
Refactoring is the disciplined art of changing structure without changing
behavior, and the tests are the only safety net that makes it honest work.

A good function reads like a sentence and hides the machinery behind a name
that tells the reader exactly what to expect before they ever look inside.

Premature optimization scatters complexity across a codebase long before any
profiler has confirmed where the real time is actually being spent.
"""


@pytest.fixture
def corpus() -> list[tuple[str, str]]:
    """A small multi-document corpus as ``(path, text)`` pairs.

    Paths are absolute-looking so we can assert outputs strip them to basenames.
    """
    base = "/private/authors/jane/writing"
    return [
        (f"{base}/garden.md", DOC_GARDEN),
        (f"{base}/ocean.md", DOC_OCEAN),
        (f"{base}/code.md", DOC_CODE),
    ]


@pytest.fixture
def index(tmp_path, corpus) -> ExcerptIndex:
    """A built ``ExcerptIndex`` over the inline corpus."""
    db_path = tmp_path / "examples.db"
    idx = ExcerptIndex(db_path)
    idx.index_files(corpus)
    yield idx
    idx.close()


# ---------------------------------------------------------------------------
# 1. Build + FTS5 MATCH + BM25 returns expected excerpts
# ---------------------------------------------------------------------------


def test_search_returns_relevant_excerpts(index):
    """A query for gardening terms surfaces the garden document's excerpts."""
    results = index.search("tomatoes soil watering", max_results=5)

    assert results, "expected at least one matching excerpt"
    top = results[0]
    assert "tomatoes" in top["excerpt"].lower()
    assert top["source_filename"] == "garden.md"


def test_search_ranks_by_relevance(index):
    """The most on-topic document outranks unrelated ones (BM25 ordering)."""
    results = index.search("sailing wind tiller boat", max_results=5)

    assert results
    assert results[0]["source_filename"] == "ocean.md"
    # Scores are non-increasing (sorted best-first).
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_query_returns_empty(index):
    """An empty / punctuation-only query yields no results, not an error."""
    assert index.search("", max_results=5) == []
    assert index.search("   !!! ???  ", max_results=5) == []


# ---------------------------------------------------------------------------
# 2. Excerpts are whole paragraphs, not 1500-char windows
# ---------------------------------------------------------------------------


def test_excerpts_are_whole_paragraphs(index):
    """Each excerpt is a coherent paragraph from the source, not a sliced window.

    The matching paragraph about tomatoes must come back intact (same leading and
    trailing words), proving paragraph-boundary indexing rather than char-window
    chunking that would clip mid-sentence.
    """
    results = index.search("tomatoes mulch moisture", max_results=5)
    assert results
    excerpt = results[0]["excerpt"]

    # Whole paragraph: starts at the paragraph's first word and ends on its last.
    assert excerpt.startswith("Tomatoes ripen best")
    assert excerpt.rstrip().endswith("long dry weeks.")
    # No overlap artifacts: the unrelated frost paragraph isn't glued on.
    assert "morning frost" not in excerpt


# ---------------------------------------------------------------------------
# 3. Per-source diversity cap
# ---------------------------------------------------------------------------


def test_diversity_cap_limits_excerpts_per_source(tmp_path):
    """No single source dominates results beyond MAX_EXCERPTS_PER_SOURCE."""
    # One document with many paragraphs that ALL strongly match the query.
    paragraphs = [
        f"Coffee brewing is a daily ritual and this is paragraph number {i} "
        f"about coffee brewing and coffee beans and coffee water temperature."
        for i in range(10)
    ]
    text = "\n\n".join(paragraphs)

    db_path = tmp_path / "examples.db"
    idx = ExcerptIndex(db_path)
    try:
        idx.index_files([("/authors/x/coffee.md", text)])
        results = idx.search("coffee brewing beans", max_results=20)

        from_coffee = [r for r in results if r["source_filename"] == "coffee.md"]
        assert len(from_coffee) <= MAX_EXCERPTS_PER_SOURCE
    finally:
        idx.close()


# ---------------------------------------------------------------------------
# 4. Result shape + no absolute paths
# ---------------------------------------------------------------------------


def test_result_shape_and_no_absolute_paths(index):
    """Results carry exactly excerpt/source_filename/score; never absolute paths."""
    results = index.search("garden basil rosemary bees", max_results=5)
    assert results

    for r in results:
        assert set(r.keys()) == {"excerpt", "source_filename", "score"}
        assert isinstance(r["excerpt"], str) and r["excerpt"]
        assert isinstance(r["score"], float)

        fname = r["source_filename"]
        # Basename only — no directory separators, no absolute prefixes.
        assert not os.path.isabs(fname)
        assert "/" not in fname
        assert "\\" not in fname
        assert fname in {"garden.md", "ocean.md", "code.md"}


# ---------------------------------------------------------------------------
# 5. max_results clamping (min 1 / max 20)
# ---------------------------------------------------------------------------


def test_max_results_clamped_to_floor(index):
    """A non-positive max_results is clamped up to MIN_RESULTS_FLOOR."""
    results = index.search("the", max_results=0)
    # "the" is a common token; with a floor of 1 we get at most 1 back here only
    # if matches exist. The contract is: never more than the clamped floor.
    assert len(results) <= MIN_RESULTS_FLOOR
    assert MIN_RESULTS_FLOOR == 1


def test_max_results_clamped_to_ceiling(tmp_path):
    """A huge max_results is clamped down to MAX_RESULTS_CEILING."""
    # Build a corpus with many sources so diversity cap doesn't mask the ceiling.
    corpus = []
    for i in range(40):
        text = (
            f"Document {i} discusses rivers and mountains and forests in detail. "
            f"Rivers carve valleys and mountains rise above the forests slowly."
        )
        corpus.append((f"/authors/x/doc_{i}.md", text))

    db_path = tmp_path / "examples.db"
    idx = ExcerptIndex(db_path)
    try:
        idx.index_files(corpus)
        results = idx.search("rivers mountains forests", max_results=9999)
        assert len(results) <= MAX_RESULTS_CEILING
        assert MAX_RESULTS_CEILING == 20
    finally:
        idx.close()


# ---------------------------------------------------------------------------
# 6. Incremental refresh via file_meta change-detection
# ---------------------------------------------------------------------------


def test_unchanged_file_is_skipped_on_reindex(tmp_path):
    """Re-indexing identical (path, text) leaves the index unchanged (skip)."""
    db_path = tmp_path / "examples.db"
    idx = ExcerptIndex(db_path)
    try:
        idx.index_files([("/authors/x/a.md", DOC_GARDEN)])
        before = idx.excerpt_count

        # Same content again -> change detection should skip, count unchanged.
        report = idx.index_files([("/authors/x/a.md", DOC_GARDEN)])
        after = idx.excerpt_count

        assert after == before
        assert report["skipped"] >= 1
        assert report["indexed"] == 0
    finally:
        idx.close()


def test_changed_file_is_reindexed(tmp_path):
    """Changing a file's text replaces its excerpts on the next index pass."""
    db_path = tmp_path / "examples.db"
    idx = ExcerptIndex(db_path)
    try:
        idx.update_file("/authors/x/a.md", "Old paragraph about apples.\n")
        assert idx.search("apples", max_results=5)
        assert not idx.search("zebras", max_results=5)

        idx.update_file("/authors/x/a.md", "New paragraph about zebras.\n")
        # Old content gone, new content present.
        assert not idx.search("apples", max_results=5)
        assert idx.search("zebras", max_results=5)
    finally:
        idx.close()


def test_index_persists_across_reopen(tmp_path):
    """The SQLite DB survives close/reopen (persistence + WAL lifecycle)."""
    db_path = tmp_path / "examples.db"
    idx = ExcerptIndex(db_path)
    idx.index_files([("/authors/x/garden.md", DOC_GARDEN)])
    idx.close()

    reopened = ExcerptIndex(db_path)
    try:
        results = reopened.search("tomatoes soil", max_results=5)
        assert results
        assert results[0]["source_filename"] == "garden.md"
    finally:
        reopened.close()
