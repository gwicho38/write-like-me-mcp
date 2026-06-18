"""Whole-excerpt index and search engine for ``find_writing_examples``.

ADAPTED from the ``legal-workspace-mcp`` indexer: this module reuses that
template's proven SQLite **FTS5 + BM25** ranking, ``file_meta`` change-detection
table, and WAL pragmas, but **drops the search-tuned 1500-char overlapping
chunking**. Style retrieval wants *coherent real excerpts*, so the corpus is
indexed at **paragraph boundaries** (blank-line separated spans), with an
optional sentence-boundary cap for pathologically long paragraphs — never a
sliding character window.

Privacy / output contract:
    * Excerpts are served live from a local-only ``examples.db``.
    * Search results carry only the source **basename** — never an absolute path.

Extraction lives in :mod:`write_like_me_mcp.extractors`; this index never reads
files itself. Callers supply already-extracted ``(path, text)`` pairs so the
extraction strategy stays in exactly one place.

All diagnostic logging goes to **stderr** (the module logger); stdout is
reserved for the MCP protocol channel.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional, TypedDict, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants (no magic literals)
# ---------------------------------------------------------------------------

#: SQLite schema version, for future migrations.
SCHEMA_VERSION: int = 1

#: Minimum value ``max_results`` is clamped up to (inclusive floor).
MIN_RESULTS_FLOOR: int = 1

#: Maximum value ``max_results`` is clamped down to (inclusive ceiling).
MAX_RESULTS_CEILING: int = 20

#: Per-source diversity cap: at most this many excerpts from any one document.
MAX_EXCERPTS_PER_SOURCE: int = 2

#: Multiplier for over-fetching candidate rows before diversity filtering.
DIVERSITY_OVERFETCH_FACTOR: int = 6

#: Paragraphs shorter than this (chars) are dropped as non-representative noise
#: (headings, stray lines) rather than indexed as standalone excerpts.
MIN_PARAGRAPH_CHARS: int = 40

#: Paragraphs longer than this (chars) are split at sentence boundaries so a
#: single excerpt stays digestible for few-shot prompting. This is a coherence
#: cap, NOT a sliding window — splits land on sentence ends only.
MAX_PARAGRAPH_CHARS: int = 1200

#: Cap on excerpts indexed per source file, to bound DB growth on huge docs.
MAX_EXCERPTS_PER_FILE_INDEXED: int = 500


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


class ExcerptResult(TypedDict):
    """A single search hit: a coherent excerpt with provenance and BM25 score."""

    excerpt: str
    source_filename: str
    score: float


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class ExcerptIndex:
    """SQLite FTS5 index of whole-paragraph corpus excerpts with BM25 ranking.

    Lifecycle::

        idx = ExcerptIndex(db_path)
        idx.index_files([(path, text), ...])   # batch (re)index w/ change-detection
        idx.update_file(path, text)            # single-file incremental update
        idx.remove_file(path)                  # drop a file's excerpts
        results = idx.search(query, max_results)
        idx.close()

    The index accepts already-extracted ``(path, text)`` pairs; it never reads
    files from disk. Change detection uses an MD5 of the extracted text stored in
    a ``file_meta`` table, so unchanged content is skipped on re-index.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        """Open (creating if needed) the SQLite excerpt index at ``db_path``.

        Args:
            db_path: Filesystem path to the SQLite database file. The parent
                directory is created if it does not exist.
        """
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # -- lifecycle ----------------------------------------------------------

    def _init_db(self) -> None:
        """Create the connection, apply WAL pragmas, and ensure the schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Performance / durability pragmas (template-derived).
        self._conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-2000;
            PRAGMA temp_store=MEMORY;
            """
        )

        # Change-detection + bookkeeping tables.
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS file_meta(
                file_path TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                excerpt_count INTEGER NOT NULL,
                indexed_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS schema_info(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

        # FTS5 excerpt table: porter unicode61 tokenizer, BM25 ranking (default).
        # file_path/file_name are UNINDEXED metadata; only `excerpt` is searched.
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS excerpts USING fts5(
                file_path UNINDEXED,
                file_name UNINDEXED,
                excerpt,
                tokenize='porter unicode61'
            );
            """
        )

        self._conn.execute(
            "INSERT OR REPLACE INTO schema_info(key, value) VALUES('version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def close(self) -> None:
        """Optimize and close the database connection (idempotent)."""
        if self._conn is not None:
            try:
                self._conn.execute("PRAGMA optimize")
                self._conn.close()
            except sqlite3.Error as exc:
                logger.error("Error closing excerpt index: %s", exc)
            finally:
                self._conn = None

    # -- introspection ------------------------------------------------------

    @property
    def excerpt_count(self) -> int:
        """Total number of indexed excerpts across all sources."""
        if self._conn is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM excerpts").fetchone()
        return int(row[0]) if row else 0

    @property
    def source_count(self) -> int:
        """Number of distinct source documents in the index."""
        if self._conn is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM file_meta").fetchone()
        return int(row[0]) if row else 0

    # -- indexing -----------------------------------------------------------

    def index_files(self, files: list[tuple[str, str]]) -> dict[str, int]:
        """Batch (re)index ``(path, text)`` pairs with change-detection.

        Unchanged content (matching stored MD5) is skipped. Changed or new
        content replaces any prior excerpts for that path.

        Args:
            files: List of ``(source_path, extracted_text)`` tuples.

        Returns:
            A summary dict with keys ``indexed``, ``skipped``, ``removed``,
            and ``excerpts`` (total excerpts now in the index).
        """
        indexed = 0
        skipped = 0
        for path, text in files:
            if self.update_file(path, text):
                indexed += 1
            else:
                skipped += 1

        return {
            "indexed": indexed,
            "skipped": skipped,
            "removed": 0,
            "excerpts": self.excerpt_count,
        }

    def update_file(self, path: str, text: Optional[str]) -> bool:
        """Index or refresh a single source's excerpts (incremental).

        Args:
            path: Source path (used only to derive the stored basename; never
                emitted as an absolute path in results).
            text: The already-extracted text. ``None``/empty removes the source.

        Returns:
            ``True`` if the index was modified (new or changed content),
            ``False`` if the content was unchanged and skipped.
        """
        if self._conn is None:
            return False

        if not text or not text.strip():
            self.remove_file(path)
            return False

        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

        # Skip unchanged content.
        row = self._conn.execute(
            "SELECT content_hash FROM file_meta WHERE file_path = ?",
            (path,),
        ).fetchone()
        if row is not None and row[0] == content_hash:
            return False

        excerpts = self._split_excerpts(text)
        if not excerpts:
            self.remove_file(path)
            return False

        file_name = Path(path).name
        self._remove_rows(path)

        self._conn.executemany(
            "INSERT INTO excerpts(file_path, file_name, excerpt) VALUES(?, ?, ?)",
            [(path, file_name, excerpt) for excerpt in excerpts],
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO file_meta(
                file_path, file_name, content_hash, excerpt_count, indexed_at
            ) VALUES(?, ?, ?, ?, ?)
            """,
            (path, file_name, content_hash, len(excerpts), time.time()),
        )
        self._conn.commit()
        logger.info("Indexed %s (%d excerpts)", file_name, len(excerpts))
        return True

    def remove_file(self, path: str) -> None:
        """Remove a source and all of its excerpts from the index.

        Args:
            path: The source path previously passed to :meth:`update_file`.
        """
        if self._conn is None:
            return
        existed = self._conn.execute(
            "SELECT 1 FROM file_meta WHERE file_path = ?", (path,)
        ).fetchone()
        self._remove_rows(path)
        self._conn.commit()
        if existed:
            logger.info("Removed %s from excerpt index", Path(path).name)

    def _remove_rows(self, path: str) -> None:
        """Delete a path's excerpt rows and metadata (no commit)."""
        if self._conn is None:
            return
        self._conn.execute("DELETE FROM excerpts WHERE file_path = ?", (path,))
        self._conn.execute("DELETE FROM file_meta WHERE file_path = ?", (path,))

    def clear(self) -> None:
        """Drop every excerpt and all file metadata, leaving an empty index.

        Public entry point for a full rebuild (used by ``analyze_writing_style``
        with ``force_reindex=True``), so callers never reach into the private
        connection. The schema/``schema_info`` row is preserved.
        """
        if self._conn is None:
            return
        self._conn.execute("DELETE FROM excerpts")
        self._conn.execute("DELETE FROM file_meta")
        self._conn.commit()
        logger.info("Excerpt index cleared")

    def indexed_paths(self) -> list[str]:
        """Return the source paths currently present in the index."""
        if self._conn is None:
            return []
        rows = self._conn.execute("SELECT file_path FROM file_meta").fetchall()
        return [row[0] for row in rows]

    # -- search -------------------------------------------------------------

    def search(self, query: str, max_results: int = 5) -> list[ExcerptResult]:
        """Search excerpts via FTS5 MATCH, ranked by BM25, diversity-capped.

        Args:
            query: User query string. Empty/punctuation-only queries return ``[]``.
            max_results: Desired result count, clamped to
                ``[MIN_RESULTS_FLOOR, MAX_RESULTS_CEILING]``.

        Returns:
            A list of :class:`ExcerptResult` dicts sorted best-first. Each carries
            the source **basename** only — never an absolute path.
        """
        if self._conn is None:
            return []

        limit = self._clamp_max_results(max_results)
        fts_query = self._prepare_fts_query(query)
        if not fts_query:
            return []

        # Over-fetch so the per-source diversity cap still leaves enough results.
        fetch_limit = limit * DIVERSITY_OVERFETCH_FACTOR
        try:
            rows = self._conn.execute(
                """
                SELECT file_name, excerpt, rank
                FROM excerpts
                WHERE excerpts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, fetch_limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 query failed for %r: %s", query, exc)
            return []

        results: list[ExcerptResult] = []
        per_source: dict[str, int] = {}

        for row in rows:
            if len(results) >= limit:
                break
            file_name = row["file_name"]

            # Per-source diversity cap.
            if per_source.get(file_name, 0) >= MAX_EXCERPTS_PER_SOURCE:
                continue
            per_source[file_name] = per_source.get(file_name, 0) + 1

            # BM25 `rank` is negative (more-negative == better); flip to positive.
            score = -float(row["rank"])
            results.append(
                ExcerptResult(
                    excerpt=row["excerpt"],
                    source_filename=file_name,
                    score=score,
                )
            )

        return results

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _clamp_max_results(max_results: int) -> int:
        """Clamp ``max_results`` into ``[MIN_RESULTS_FLOOR, MAX_RESULTS_CEILING]``."""
        try:
            value = int(max_results)
        except (TypeError, ValueError):
            value = MIN_RESULTS_FLOOR
        return max(MIN_RESULTS_FLOOR, min(value, MAX_RESULTS_CEILING))

    @staticmethod
    def _prepare_fts_query(query: str) -> str:
        """Convert a free-text query into a safe FTS5 MATCH expression.

        Strips FTS5 special characters and quotes each term, joining with ``OR``
        so excerpts matching *any* query term are candidates; BM25 then ranks
        excerpts matching more (and rarer) terms higher. OR semantics suit
        few-shot excerpt retrieval, where a strict AND of every term would too
        often exclude otherwise-representative spans. Returns ``""`` for empty
        input.
        """
        if not query:
            return ""
        cleaned = re.sub(r"[^\w\s]", " ", query)
        terms = [t for t in cleaned.split() if t]
        if not terms:
            return ""
        return " OR ".join(f'"{term}"' for term in terms)

    @staticmethod
    def _split_excerpts(text: str) -> list[str]:
        """Split text into coherent whole-paragraph excerpts.

        Paragraphs are blank-line-separated spans. Very short fragments are
        dropped; very long paragraphs are split at sentence boundaries (a
        coherence cap, NOT a sliding character window). Internal newlines within
        a paragraph are collapsed to single spaces so each excerpt is a clean,
        few-shot-ready span.

        Args:
            text: Extracted source text.

        Returns:
            Up to ``MAX_EXCERPTS_PER_FILE_INDEXED`` coherent excerpts.
        """
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        raw_paragraphs = re.split(r"\n\s*\n", normalized)

        excerpts: list[str] = []
        for para in raw_paragraphs:
            collapsed = re.sub(r"\s+", " ", para).strip()
            if len(collapsed) < MIN_PARAGRAPH_CHARS:
                continue

            if len(collapsed) <= MAX_PARAGRAPH_CHARS:
                excerpts.append(collapsed)
            else:
                excerpts.extend(ExcerptIndex._split_long_paragraph(collapsed))

            if len(excerpts) >= MAX_EXCERPTS_PER_FILE_INDEXED:
                return excerpts[:MAX_EXCERPTS_PER_FILE_INDEXED]

        # Fallback: a short single-paragraph source (every span below the
        # min-length floor) is still indexed as one whole excerpt rather than
        # dropped entirely, so small documents remain retrievable.
        if not excerpts:
            collapsed_all = re.sub(r"\s+", " ", normalized).strip()
            if collapsed_all:
                return [collapsed_all[:MAX_PARAGRAPH_CHARS]]

        return excerpts

    @staticmethod
    def _split_long_paragraph(paragraph: str) -> list[str]:
        """Split an over-long paragraph at sentence boundaries into ≤cap spans.

        Sentences are accumulated until adding the next would exceed
        ``MAX_PARAGRAPH_CHARS``; a single oversized sentence is emitted whole
        rather than cut mid-word.
        """
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        spans: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and current_len + len(sentence) + 1 > MAX_PARAGRAPH_CHARS:
                spans.append(" ".join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence) + 1

        if current:
            spans.append(" ".join(current))

        return spans
