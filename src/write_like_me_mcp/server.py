"""Write-Like-Me MCP Server.

Learns a user's personal writing style from a *local* document corpus and serves
self-updating style context to Claude over the Model Context Protocol — so that
generated prose matches how the author actually writes. Everything runs
on-device: there is **no network and no LLM code path** in v0.1.

Tools (tools only — no resources):
    - analyze_writing_style: (Re)scan the corpus, build profile.json + examples.db.
    - get_style_profile:      Return the full persisted StyleProfile (call this
                              before writing prose to match the author's voice).
    - apply_style:            Compare a draft to the profile, return a rewrite brief.
    - find_writing_examples:  FTS5/BM25 search for representative real excerpts.
    - style_status:           Report profile/index health and readiness.

Server lifespan (ADAPTED from legal-workspace-mcp/server.py):
    1. ``load_config`` resolves the active profile via the discovery chain.
    2. An ``ExcerptIndex`` is opened and any persisted ``profile.json`` is loaded
       for immediate availability.
    3. A background ``asyncio.create_task`` (re)builds the profile and excerpt
       index so the MCP stdio handshake returns immediately.
    4. The build sets the ``_profile_ready`` :class:`asyncio.Event`; tools
       ``await`` it with a ~30s timeout and otherwise return a structured
       ``"building"`` status (the protocol is never blocked indefinitely).
    5. A debounced :class:`WorkspaceWatcher` triggers a corpus-global rebuild on
       edits to any source.
    6. On shutdown the build task is cancelled, the watcher stopped, the DB closed.

All diagnostic logging goes to **stderr** — stdout is the MCP protocol channel.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is POSIX-only (macOS/Linux target)
    fcntl = None  # type: ignore[assignment]

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from .config import (
    ConfigError,
    StyleConfig,
    is_cloud_only,
    is_excluded,
    load_config,
)
from .extractors import extract_text
from .indexer import ExcerptIndex
from .model import StyleProfile
from .prose_filter import is_prose
from .style_analyzer import analyze_documents, analyze_text
from .watcher import WorkspaceWatcher

# ---------------------------------------------------------------------------
# Logging to stderr (stdout is reserved for the MCP protocol channel)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named constants (no magic literals)
# ---------------------------------------------------------------------------

#: Seconds a tool will ``await`` the readiness event before returning a
#: structured ``"building"`` status. Module-level so tests can patch it short.
READINESS_TIMEOUT_SECONDS: float = 30.0

#: When more than this share of the collected corpus files cannot be read, a
#: rebuild does not overwrite an existing profile built from more documents.
UNREADABLE_KEEP_PROFILE_RATIO: float = 0.5

#: Readiness state strings returned to callers.
READINESS_READY: str = "ready"
READINESS_BUILDING: str = "building"

#: Difference thresholds for ``apply_style`` draft-vs-profile observations.
#: A sentence-length gap (in words) below this is treated as "close enough".
SENTENCE_LENGTH_TOLERANCE_WORDS: float = 4.0
#: Contraction-rate gap (per-1k words) below this is treated as "close enough".
CONTRACTION_RATE_TOLERANCE: float = 2.0
#: Em-dash rate (per-1k words) above this in the profile counts as "frequent".
EM_DASH_FREQUENT_THRESHOLD: float = 1.0


# ---------------------------------------------------------------------------
# Module-level server state (initialized in the lifespan)
# ---------------------------------------------------------------------------

_config: Optional[StyleConfig] = None
_index: Optional[ExcerptIndex] = None
_watcher: Optional[WorkspaceWatcher] = None
_profile: Optional[StyleProfile] = None
_build_task: Optional[asyncio.Task] = None

#: Config file path supplied as a CLI argument (discovery Priority 1), set by
#: :func:`main` before the server starts and read by the lifespan. ``None`` falls
#: back to the ``WRITE_LIKE_ME_CONFIG`` env var, then ``~/.write-like-me.json``.
_cli_config_path: Optional[str] = None

#: Serializes the synchronous corpus build across ALL callers — the startup
#: background task, explicit ``analyze_writing_style`` calls (both via
#: ``asyncio.to_thread``), and the watcher's debounced rebuild (a timer thread).
#: Without this, two builds could write the same ``examples.db`` connection
#: concurrently (the connection uses ``check_same_thread=False`` and does not
#: serialize writers). A plain ``threading.Lock`` covers every path, including
#: the non-async watcher thread that an ``asyncio.Lock`` could not.
_build_thread_lock: threading.Lock = threading.Lock()

#: Set when the background corpus build completes; tools await this.
_profile_ready: asyncio.Event = asyncio.Event()

#: In-process cache of extracted text, keyed by resolved file path, storing
#: ``(mtime_ns, size, text)``. A corpus build reuses a cache hit instead of
#: re-running ``extract_text`` for files unchanged since they were last read —
#: unavoidable on first sight of a file, but every subsequent rebuild triggered
#: by the watcher (one full corpus rebuild per changed file, per its own
#: docstring) otherwise re-extracts *every* file in the corpus, not just the
#: one that changed. Process-lifetime only: never persisted, so this does not
#: touch the "no verbatim corpus text is ever persisted" privacy invariant —
#: the text is already transiently in memory during any build regardless.
_extract_cache: dict[Path, tuple[int, int, str]] = {}

#: Filename of the cross-process advisory lock file within ``profile_dir``.
#: All Claude Code sessions on a machine share the same ``data_dir`` (it is
#: keyed by profile name, not by session/PID), so independent server processes
#: started by separate sessions all read/write the same ``profile.json`` and
#: ``examples.db``. Without this lock, N concurrently-running sessions each
#: redundantly re-walk and re-analyze the whole corpus on every corpus change.
_REBUILD_LOCK_FILENAME: str = "build.lock"

#: If ``profile.json`` was written less than this many seconds ago, treat a
#: failed lock acquisition as "another process just finished building" rather
#: than "building never happened" — safe to reuse its output instead of
#: queuing behind the lock or redoing the work ourselves.
_FRESH_REBUILD_STALENESS_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# Corpus collection + build (synchronous core, run off the event loop)
# ---------------------------------------------------------------------------


def _collect_corpus_files(config: StyleConfig) -> list[Path]:
    """Collect analyzable corpus files from the configured sources.

    Directory sources are walked recursively; file sources are included directly.
    Only files whose suffix is in ``config.file_extensions`` are kept, and hidden
    / Office-temp files and excluded path components are skipped. Exclude
    entries may be literal components or globs (``*-master``, ``*.azw.md``). The result is
    de-duplicated and sorted for deterministic ordering.

    Args:
        config: The resolved active-profile configuration.

    Returns:
        A sorted, de-duplicated list of corpus file paths.
    """
    extensions = {e.lower() for e in config.file_extensions}

    def _is_relevant(path: Path) -> bool:
        name = path.name
        if name.startswith(".") or name.startswith("~$"):
            return False
        if path.suffix.lower() not in extensions:
            return False
        return not is_excluded(path, config.exclude)

    found: set[Path] = set()
    for source in config.resolved_sources:
        if source.is_dir():
            for candidate in source.rglob("*"):
                if candidate.is_file() and _is_relevant(candidate):
                    found.add(candidate)
        elif source.is_file():
            if _is_relevant(source):
                found.add(source)
        else:
            logger.warning("Source does not exist; skipping: %s", source)

    return sorted(found)


def _extract_text_cached(path: Path) -> Optional[str]:
    """Extract text from ``path``, reusing ``_extract_cache`` when unchanged.

    A cache hit requires both the file's mtime and size to match what was
    recorded at the previous extraction; either changing invalidates the entry
    and forces a fresh ``extract_text`` call. This is the single extraction
    point for a corpus build — both the style profile and the excerpt index
    consume its result — so each changed file is read and parsed once per
    rebuild, and an unchanged file is not re-parsed at all.

    Args:
        path: Corpus file to extract.

    Returns:
        The extracted text, or ``None`` if the file is unreadable/unstattable
        or extraction yields nothing.
    """
    try:
        stat = path.stat()
    except OSError:
        return None

    cached = _extract_cache.get(path)
    if (
        cached is not None
        and cached[0] == stat.st_mtime_ns
        and cached[1] == stat.st_size
    ):
        return cached[2]

    text = extract_text(path)
    if text:
        _extract_cache[path] = (stat.st_mtime_ns, stat.st_size, text)
    else:
        _extract_cache.pop(path, None)
    return text


def _profile_is_fresh(config: StyleConfig) -> bool:
    """Whether ``profile.json`` was written within the last few seconds.

    Args:
        config: The resolved active-profile configuration.

    Returns:
        ``True`` if the file exists and its mtime is within
        :data:`_FRESH_REBUILD_STALENESS_SECONDS` of now.
    """
    try:
        mtime = config.profile_path.stat().st_mtime
    except OSError:
        return False
    return (time.time() - mtime) < _FRESH_REBUILD_STALENESS_SECONDS


def _try_acquire_build_lock(config: StyleConfig) -> Optional[IO[str]]:
    """Try to acquire the cross-process build lock; ``None`` if unavailable.

    Uses a non-blocking ``flock`` on a lock file inside ``profile_dir`` so that
    when several independent server processes (one per Claude Code session,
    all sharing the same ``data_dir``) each want to rebuild the same corpus at
    once, only one actually does the work. On platforms without ``fcntl``
    (non-POSIX), locking is skipped entirely and every caller proceeds to build
    — the pre-existing, non-regressed behavior.

    Args:
        config: The resolved active-profile configuration.

    Returns:
        The open, locked file handle (caller must close it, which releases the
        lock) on success; ``None`` if another process already holds it or
        locking is unavailable on this platform.
    """
    if fcntl is None:  # pragma: no cover - exercised only on non-POSIX platforms
        return None

    config.profile_dir.mkdir(parents=True, exist_ok=True)
    lock_path = config.profile_dir / _REBUILD_LOCK_FILENAME
    handle = open(lock_path, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _release_build_lock(handle: Optional[IO[str]]) -> None:
    """Release and close a lock handle from :func:`_try_acquire_build_lock`."""
    if handle is None:
        return
    if fcntl is not None:  # pragma: no branch - fcntl is None only on non-POSIX
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:  # pragma: no cover - best-effort unlock
            pass
    handle.close()


def _build_corpus_sync(force_reindex: bool = False) -> dict:
    """Build the StyleProfile and excerpt index from the configured corpus.

    This is the synchronous core invoked from a worker thread. It collects the
    corpus, extracts each file's text exactly once (via the in-process
    :data:`_extract_cache`, so unchanged files since the last build in this
    process are not re-parsed), computes the profile from that already-extracted
    text, persists ``profile.json``, updates the live ``_profile`` global, and
    (re)indexes excerpts into ``examples.db``. With ``force_reindex=True`` the
    excerpt index is cleared and rebuilt from scratch; otherwise the index's own
    MD5 change-detection skips unchanged files.

    Independent server processes (one per Claude Code session) share the same
    ``data_dir``/``profile.json``/``examples.db``. Before doing the work, this
    tries a non-blocking cross-process lock (:func:`_try_acquire_build_lock`);
    if another process already holds it, this reuses the on-disk
    ``profile.json`` that process just wrote (or is about to) rather than
    redundantly re-walking and re-analyzing the same corpus.

    Args:
        force_reindex: Rebuild the excerpt index from scratch when ``True``.

    Returns:
        A summary dict (``doc_count``, ``total_words``, ``source_count``,
        ``generated_at``, ``profile_path`` (relative), headline metrics). When
        another process's build was reused instead, ``status`` is
        ``"reused_existing"`` and ``excerpts_indexed`` is omitted (the shared
        index was not touched by this process).

    Raises:
        RuntimeError: If the server config/index are not initialized.
    """
    global _profile

    if _config is None or _index is None:
        raise RuntimeError("Server state not initialized; cannot build corpus.")

    config = _config

    # Serialize the whole build so concurrent callers *within this process*
    # (startup task, an explicit analyze_writing_style, the watcher thread)
    # never write the index at once.
    with _build_thread_lock:
        lock_handle = _try_acquire_build_lock(config)
        if lock_handle is None and _profile_is_fresh(config):
            reloaded = StyleProfile.load(config.profile_path)
            if reloaded is not None:
                _profile = reloaded
                logger.info(
                    "Corpus build already in progress in another process; "
                    "reused its profile.json instead of rebuilding."
                )
                meta = reloaded.metadata
                return {
                    "status": "reused_existing",
                    "doc_count": meta.get("doc_count", 0),
                    "total_words": meta.get("total_words", 0),
                    "source_count": meta.get("source_count", 0),
                    "generated_at": meta.get("generated_at"),
                    "profile_path": str(
                        Path(config.profile_name) / config.profile_path.name
                    ),
                    "avg_sentence_length": reloaded.avg_sentence_length,
                    "median_sentence_length": reloaded.median_sentence_length,
                    "lexical_diversity": reloaded.lexical_diversity,
                }
            # Lock unavailable and nothing usable on disk yet (e.g. the very
            # first build racing across sessions): fall through and build.

        try:
            files = _collect_corpus_files(config)

            # 0) Extract once, then admit. A file that yields no text is
            #    unreadable: most often a cloud-only placeholder (OneDrive /
            #    Google Drive "Files On-Demand") whose read times out because the
            #    sync client did not download it. Count these instead of dropping
            #    them silently.
            #
            #    A .md suffix is not a promise of prose: a renamed ebook, a syslog
            #    or a topology dump poisons every metric in the profile, and
            #    duplicated copies of one document multiply whatever they contain.
            #    Reject both here, once, so the profile and the excerpt index see
            #    the same admitted set.
            per_doc_texts: list[str] = []
            basenames: list[str] = []
            pairs: list[tuple[str, str]] = []
            skipped_unreadable = 0
            skipped_cloud_only = 0
            skipped_reasons: Counter[str] = Counter()
            seen_hashes: set[str] = set()
            duplicates = 0
            for path in files:
                # Never open a cloud-only placeholder: the read makes the sync
                # client download it, and one build per session over thousands
                # of placeholders fills the disk (issue #8).
                if is_cloud_only(path):
                    skipped_cloud_only += 1
                    continue
                text = _extract_text_cached(path)
                if not text or not text.strip():
                    skipped_unreadable += 1
                    continue
                verdict = is_prose(text)
                if not verdict.ok:
                    skipped_reasons[verdict.reason or "not-prose"] += 1
                    logger.debug("Skipping %s (%s)", path.name, verdict.reason)
                    continue
                digest = hashlib.md5(text.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    duplicates += 1
                    logger.debug("Skipping duplicate content: %s", path.name)
                    continue
                seen_hashes.add(digest)
                per_doc_texts.append(text)
                basenames.append(path.name)
                pairs.append((str(path), text))

            if skipped_cloud_only:
                logger.info(
                    "%d of %d corpus files are cloud-only placeholders and were not "
                    "read (mark them 'Always keep on this device' to include them)",
                    skipped_cloud_only,
                    len(files),
                )
            if skipped_unreadable:
                logger.warning(
                    "%d of %d corpus files could not be read (cloud-only placeholders "
                    "time out until the sync client downloads them)",
                    skipped_unreadable,
                    len(files),
                )
            if skipped_reasons or duplicates:
                logger.info(
                    "Corpus admission: %d of %d files admitted (non-prose: %s; duplicates: %d)",
                    len(per_doc_texts),
                    len(files),
                    dict(skipped_reasons) or "none",
                    duplicates,
                )

            # 1) Compute the StyleProfile. Do not let a build that could read
            #    only a small part of the corpus replace a fuller profile on disk.
            profile = analyze_documents(
                per_doc_texts,
                config.profile_name,
                source_paths=[Path(name) for name in basenames],
                source_count=len(files),
            )
            kept = (
                _profile
                if _profile is not None
                else StyleProfile.load(config.profile_path)
            )
            not_read = skipped_unreadable + skipped_cloud_only
            if (
                files
                and not_read / len(files) > UNREADABLE_KEEP_PROFILE_RATIO
                and kept is not None
                and kept.metadata.get("doc_count", 0)
                > profile.metadata.get("doc_count", 0)
            ):
                logger.warning(
                    "Keeping the existing profile: this build read only %d of %d files",
                    len(files) - not_read,
                    len(files),
                )
                _profile = kept
                return {
                    "status": "degraded",
                    "reason": (
                        "Most corpus files could not be read, so the existing, fuller "
                        "profile was kept. Download cloud-only files locally and rebuild."
                    ),
                    "doc_count": kept.metadata.get("doc_count", 0),
                    "source_count": len(files),
                    "skipped_unreadable": skipped_unreadable,
                    "skipped_cloud_only": skipped_cloud_only,
                    "generated_at": kept.metadata.get("generated_at"),
                }
            profile.save(config.profile_path)
            _profile = profile

            # 2) (Re)build the excerpt index from the same admitted pairs.
            #    force_reindex drops everything via the index's public clear()
            #    and rebuilds from scratch.
            if force_reindex:
                _index.clear()
            index_summary = _index.index_files(pairs)
            logger.info("Excerpt index updated: %s", index_summary)
        finally:
            _release_build_lock(lock_handle)

    meta = profile.metadata
    return {
        "status": "built",
        "doc_count": meta.get("doc_count", 0),
        "total_words": meta.get("total_words", 0),
        "source_count": meta.get("source_count", 0),
        "generated_at": meta.get("generated_at"),
        # Relative form only — never an absolute author path.
        "profile_path": str(Path(config.profile_name) / config.profile_path.name),
        "avg_sentence_length": profile.avg_sentence_length,
        "median_sentence_length": profile.median_sentence_length,
        "lexical_diversity": profile.lexical_diversity,
        "excerpts_indexed": index_summary.get("excerpts", 0),
        "skipped_unreadable": skipped_unreadable,
        "skipped_cloud_only": skipped_cloud_only,
        # What the admission gate threw away, so a corrupted corpus is visible
        # in the build result instead of only in the logs.
        "skipped_non_prose": sum(skipped_reasons.values()),
        "skipped_reasons": dict(skipped_reasons),
        "skipped_duplicates": duplicates,
    }


async def _build_corpus_async(force_reindex: bool = False) -> dict:
    """Run :func:`_build_corpus_sync` off the event loop and mark readiness.

    Args:
        force_reindex: Forwarded to :func:`_build_corpus_sync`.

    Returns:
        The build summary dict.
    """
    summary = await asyncio.to_thread(_build_corpus_sync, force_reindex)
    _profile_ready.set()
    return summary


# ---------------------------------------------------------------------------
# Readiness helper
# ---------------------------------------------------------------------------


def _build_in_flight() -> bool:
    """Whether a background corpus build is currently running."""
    return _build_task is not None and not _build_task.done()


async def _await_ready() -> bool:
    """Wait up to ``READINESS_TIMEOUT_SECONDS`` for the profile to be ready.

    Only waits while a background build is actually in flight; if no build is
    running the call returns immediately so tools can distinguish a genuine
    ``"building"`` state from a never-built (``"not_built"``) one.

    Returns:
        ``True`` if the profile is ready (event set), ``False`` if a build is
        still in flight after the timeout (the caller should return a structured
        building status). When nothing is building, returns whether the event is
        already set.
    """
    if _profile_ready.is_set():
        return True
    if not _build_in_flight():
        # Nothing is building; don't block. The caller handles the resulting
        # state (e.g. not_built) based on whether a profile exists.
        return _profile_ready.is_set()
    try:
        await asyncio.wait_for(_profile_ready.wait(), timeout=READINESS_TIMEOUT_SECONDS)
        return True
    except asyncio.TimeoutError:
        return False


def _building_payload() -> dict:
    """Structured payload returned by tools while the build is in progress."""
    return {
        "status": READINESS_BUILDING,
        "readiness": READINESS_BUILDING,
        "message": (
            "The writing-style profile is still being built from your corpus. "
            "Try again in a few seconds."
        ),
    }


def _profile_mismatch_payload(requested: Optional[str]) -> Optional[dict]:
    """Return an error payload if ``requested`` names a non-active profile.

    v0.1 builds and serves only the active profile (the schema is multi-ready,
    but profile *switching* is deferred). Rather than silently serving the active
    profile when a caller asks for a different one, every tool returns this clear
    error so the API never lies about which profile it operated on.

    Args:
        requested: The ``profile`` argument supplied to a tool, if any.

    Returns:
        An error dict when ``requested`` is set and differs from the active
        profile; otherwise ``None`` (the call may proceed on the active profile).
    """
    if requested and _config is not None and requested != _config.profile_name:
        return {
            "status": "error",
            "error": (
                f"Profile {requested!r} is not the active profile. v0.1 serves "
                f"only the active profile {_config.profile_name!r}. To switch, set "
                f"'active_profile' in your write-like-me.json."
            ),
        }
    return None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def server_lifespan(mcp_server):
    """Initialize config, excerpt index, background build, and watcher.

    Mirrors the legal-workspace template's lifespan: the index opens
    synchronously, the corpus build runs in a background task so the MCP
    handshake returns immediately, and a debounced watcher refreshes the corpus
    on edits. Cleanly tears everything down on exit.

    Raises:
        ConfigError: Re-raised with a clear message if the configuration is
            missing or invalid.
    """
    global _config, _index, _watcher, _build_task

    # Reset the readiness gate for this server instance.
    _profile_ready.clear()

    try:
        # Priority 1: explicit CLI-argument config path (set by main()); falls
        # back inside load_config to WRITE_LIKE_ME_CONFIG, then ~/.write-like-me.json.
        _config = load_config(_cli_config_path)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        raise

    logger.info(
        "Active profile %r; %d source(s)",
        _config.profile_name,
        len(_config.resolved_sources),
    )

    # Open the excerpt index (creates the DB if needed).
    _index = ExcerptIndex(_config.examples_db_path)

    # Load any persisted profile for immediate availability before the rebuild.
    global _profile
    try:
        _profile = StyleProfile.load(_config.profile_path)
        if _profile is not None:
            logger.info("Loaded existing profile from %s", _config.profile_path.name)
            _profile_ready.set()
    except Exception:  # pragma: no cover - corrupt profile is non-fatal at startup
        logger.exception("Failed to load existing profile; will rebuild")
        _profile = None

    # Background build so the stdio handshake is not blocked.
    _build_task = asyncio.create_task(_build_corpus_async(force_reindex=False))

    # Debounced corpus-global watcher; the on_change closure re-runs the build.
    def _on_change() -> None:
        logger.info("Corpus changed; rebuilding profile + excerpt index")
        try:
            _build_corpus_sync(force_reindex=False)
            _profile_ready.set()
        except Exception:  # pragma: no cover - never let a watch thread die
            logger.exception("Corpus rebuild failed")

    _watcher = WorkspaceWatcher(_config, _on_change)
    _watcher.start()

    try:
        yield {"config": _config, "index": _index, "watcher": _watcher}
    finally:
        if _build_task is not None and not _build_task.done():
            _build_task.cancel()
            try:
                await _build_task
            except asyncio.CancelledError:
                pass
        if _watcher is not None:
            _watcher.stop()
        if _index is not None:
            _index.close()


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "write_like_me_mcp",
    lifespan=server_lifespan,
    instructions=(
        "This server knows how the *user* personally writes, learned from their "
        "own local document corpus. To make any prose you produce sound like the "
        "user — emails, essays, posts, replies, documentation, messages — do the "
        "following BEFORE drafting:\n"
        "1. Call get_style_profile to load the user's voice (sentence rhythm, "
        "   vocabulary, contractions, punctuation habits, signature phrases, and a "
        "   ready-to-use style guide).\n"
        "2. Optionally call find_writing_examples with a topic to retrieve real "
        "   excerpts of the user's writing for few-shot grounding.\n"
        "3. Match the user's voice by DEFAULT when writing on their behalf. This is "
        "   guidance, not a hard constraint: honor any explicit instruction to use a "
        "   different tone or register.\n"
        "After drafting, you may call apply_style with your draft to get a concrete "
        "rewrite brief (draft-vs-author metric gaps) and tighten the result. "
        "Everything is computed locally; no document text leaves the machine."
    ),
)


# ---------------------------------------------------------------------------
# Pydantic input models (str_strip_whitespace=True, extra="forbid")
# ---------------------------------------------------------------------------


class AnalyzeInput(BaseModel):
    """Input for ``analyze_writing_style``."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    profile: Optional[str] = Field(
        default=None,
        description="Profile name to (re)build. Defaults to the active profile.",
    )
    force_reindex: bool = Field(
        default=False,
        description=(
            "Rebuild the excerpt index from scratch instead of using "
            "change-detection to skip unchanged files."
        ),
    )


class GetProfileInput(BaseModel):
    """Input for ``get_style_profile``."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    profile: Optional[str] = Field(
        default=None,
        description="Profile name to read. Defaults to the active profile.",
    )


class ApplyStyleInput(BaseModel):
    """Input for ``apply_style``."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    draft_text: str = Field(
        ...,
        description="The draft prose to compare against the author's style.",
        min_length=1,
    )
    profile: Optional[str] = Field(
        default=None,
        description="Profile name to compare against. Defaults to the active profile.",
    )


class FindExamplesInput(BaseModel):
    """Input for ``find_writing_examples``."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Topic/keywords to retrieve representative author excerpts for.",
        min_length=1,
        max_length=500,
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of excerpts to return (1-20).",
        ge=1,
        le=20,
    )
    profile: Optional[str] = Field(
        default=None,
        description="Profile name to search. Defaults to the active profile.",
    )


class StatusInput(BaseModel):
    """Input for ``style_status``."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    profile: Optional[str] = Field(
        default=None,
        description="Profile name to report on. Defaults to the active profile.",
    )


# ---------------------------------------------------------------------------
# apply_style observation helper (local-only; no LLM)
# ---------------------------------------------------------------------------


def _draft_observations(draft: StyleProfile, author: StyleProfile) -> list[str]:
    """Produce concrete draft-vs-author observations (deterministic, local).

    Compares a handful of headline metrics and emits human-readable guidance the
    calling LLM can act on. No verbatim text is involved.
    """
    observations: list[str] = []

    d_sent = draft.avg_sentence_length
    a_sent = author.avg_sentence_length
    if abs(d_sent - a_sent) > SENTENCE_LENGTH_TOLERANCE_WORDS:
        direction = "shorten" if d_sent > a_sent else "lengthen"
        observations.append(
            f"Your draft averages {d_sent:g} words/sentence; the author averages "
            f"{a_sent:g} — {direction} your sentences."
        )

    d_contr = draft.contraction_rate
    a_contr = author.contraction_rate
    if a_contr > d_contr + CONTRACTION_RATE_TOLERANCE:
        observations.append(
            f"Your draft uses few contractions ({d_contr:g}/1k words); the author "
            f"contracts ~{a_contr:g}/1k — use more contractions for their voice."
        )
    elif d_contr > a_contr + CONTRACTION_RATE_TOLERANCE:
        observations.append(
            f"Your draft uses many contractions ({d_contr:g}/1k words); the author "
            f"uses ~{a_contr:g}/1k — use fewer for their voice."
        )

    a_em = author.punctuation_per_1k.get("em_dash", 0.0)
    d_em = draft.punctuation_per_1k.get("em_dash", 0.0)
    if a_em >= EM_DASH_FREQUENT_THRESHOLD and d_em == 0.0:
        observations.append(
            "Em-dashes are absent from your draft; the author uses them "
            f"frequently (~{a_em:g}/1k words) — work some in."
        )

    if author.passive_voice_rate < draft.passive_voice_rate:
        observations.append(
            "Your draft uses more passive voice than the author — prefer the "
            "active voice."
        )

    a_first = author.formality_markers.get("first_person_rate", 0.0)
    d_first = draft.formality_markers.get("first_person_rate", 0.0)
    if a_first > 0 and d_first == 0.0:
        observations.append(
            "The author writes in the first person; your draft does not — "
            "consider first-person framing."
        )

    if not observations:
        observations.append(
            "Your draft already aligns closely with the author's headline "
            "metrics; follow the style guide for finer details."
        )
    return observations


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="analyze_writing_style",
    annotations=ToolAnnotations(
        title="Analyze Writing Style",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def analyze_writing_style(params: AnalyzeInput) -> str:
    """(Re)build the writing-style profile and excerpt index from the corpus.

    Scans the configured sources, extracts text, computes the StyleProfile,
    writes ``profile.json``, and (re)builds the excerpt index in ``examples.db``.
    Use ``force_reindex=True`` to rebuild the excerpt index from scratch.

    Args:
        params: ``profile`` (optional; defaults to active) and ``force_reindex``.

    Returns:
        JSON summary with ``status: "built"``, ``doc_count``, ``total_words``,
        ``source_count``, ``generated_at``, ``profile_path`` (relative), and
        headline metrics.
    """
    logger.info("TOOL analyze_writing_style | force_reindex=%s", params.force_reindex)
    if _config is None or _index is None:
        return json.dumps({"status": "error", "error": "Server not initialized."})

    mismatch = _profile_mismatch_payload(params.profile)
    if mismatch is not None:
        return json.dumps(mismatch)

    try:
        summary = await _build_corpus_async(force_reindex=params.force_reindex)
    except Exception as exc:  # pragma: no cover - surfaced to the caller as JSON
        logger.exception("analyze_writing_style failed")
        return json.dumps({"status": "error", "error": str(exc)})

    return json.dumps(summary, indent=2, ensure_ascii=False)


@mcp.tool(
    name="get_style_profile",
    annotations=ToolAnnotations(
        title="Get Style Profile",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def get_style_profile(params: GetProfileInput) -> str:
    """Return the full persisted StyleProfile — call before writing prose.

    Args:
        params: ``profile`` (optional; defaults to the active profile).

    Returns:
        The full StyleProfile JSON, or a ``status: "not_built"`` payload guiding
        the caller to run ``analyze_writing_style`` first.
    """
    logger.info("TOOL get_style_profile")
    mismatch = _profile_mismatch_payload(params.profile)
    if mismatch is not None:
        return json.dumps(mismatch)
    if not await _await_ready() and _build_in_flight():
        return json.dumps(_building_payload())

    if _profile is None:
        return json.dumps(
            {
                "status": "not_built",
                "message": (
                    "No style profile has been built yet. Run analyze_writing_style "
                    "to learn the author's voice from the configured corpus."
                ),
            }
        )

    return json.dumps(_profile.to_dict(), indent=2, ensure_ascii=False)


def _target_metrics_for(
    profile: StyleProfile, draft_profile: StyleProfile
) -> tuple[dict, str | None]:
    """Return the author metrics a draft should be compared against.

    Passive voice and contraction habits only mean something within one
    grammar, so a draft is measured against the author's metrics *in the
    draft's own language* when the corpus contains that language. Otherwise the
    top-level (dominant-language) metrics are used.

    Args:
        profile: The author's built style profile.
        draft_profile: The profile computed for the draft text.

    Returns:
        A ``(metrics, matched_language)`` pair. ``matched_language`` is the ISO
        code whose metrics were used, or ``None`` when the draft's language is
        not represented in the corpus and the dominant metrics were used.
    """
    metrics = {
        "avg_sentence_length": profile.avg_sentence_length,
        "median_sentence_length": profile.median_sentence_length,
        "lexical_diversity": profile.lexical_diversity,
        "contraction_rate": profile.contraction_rate,
        "passive_voice_rate": profile.passive_voice_rate,
        "punctuation_per_1k": profile.punctuation_per_1k,
        "formality_markers": profile.formality_markers,
    }

    draft_language = draft_profile.metadata.get("dominant_language")
    entry = profile.languages.get(draft_language) if draft_language else None
    if entry is None:
        return metrics, None

    metrics["contraction_rate"] = entry["contraction_rate"]
    metrics["passive_voice_rate"] = entry["passive_voice_rate"]
    metrics["formality_markers"] = entry["formality_markers"]
    return metrics, draft_language


@mcp.tool(
    name="apply_style",
    annotations=ToolAnnotations(
        title="Apply Style (Rewrite Brief)",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def apply_style(params: ApplyStyleInput) -> str:
    """Compare a draft to the author's profile and return a rewrite brief.

    Computes lightweight metrics on ``draft_text`` (locally, no network/LLM),
    compares them to the stored profile, and returns the author's style guide
    plus concrete draft-vs-author observations. The calling LLM performs the
    actual rewrite.

    Args:
        params: ``draft_text`` (required, non-empty) and optional ``profile``.

    Returns:
        JSON ``{ style_guide_md, observations[], target_metrics, draft_metrics,
        mode: "brief" }``, or a building / not_built status if no profile exists.
    """
    logger.info("TOOL apply_style | draft_chars=%d", len(params.draft_text))
    mismatch = _profile_mismatch_payload(params.profile)
    if mismatch is not None:
        return json.dumps(mismatch)
    if not await _await_ready() and _build_in_flight():
        return json.dumps(_building_payload())

    if _profile is None:
        return json.dumps(
            {
                "status": "not_built",
                "message": (
                    "No style profile available. Run analyze_writing_style before "
                    "requesting a rewrite brief."
                ),
            }
        )

    profile_name = params.profile or (_config.profile_name if _config else "draft")
    draft_profile = analyze_text(params.draft_text, f"{profile_name}:draft")

    target_metrics, matched_language = _target_metrics_for(_profile, draft_profile)
    draft_metrics = {
        "avg_sentence_length": draft_profile.avg_sentence_length,
        "median_sentence_length": draft_profile.median_sentence_length,
        "lexical_diversity": draft_profile.lexical_diversity,
        "contraction_rate": draft_profile.contraction_rate,
        "passive_voice_rate": draft_profile.passive_voice_rate,
        "punctuation_per_1k": draft_profile.punctuation_per_1k,
        "formality_markers": draft_profile.formality_markers,
    }

    payload = {
        "mode": "brief",
        "style_guide_md": _profile.style_guide_md,
        "observations": _draft_observations(draft_profile, _profile),
        "target_metrics": target_metrics,
        "draft_metrics": draft_metrics,
        "matched_language": matched_language,
    }
    # A short draft (< MATTR window) computes lexical_diversity as plain TTR while
    # the author corpus uses MATTR — the two numbers are not directly comparable.
    # Surface that so a consumer does not naively diff them.
    if draft_profile.metadata.get("ttr_fallback") and not _profile.metadata.get(
        "ttr_fallback"
    ):
        payload["metric_notes"] = (
            "draft_metrics.lexical_diversity is plain TTR (draft shorter than the "
            "MATTR window); target_metrics.lexical_diversity is MATTR — do not "
            "compare these two values directly."
        )
    return json.dumps(payload, indent=2, ensure_ascii=False)


@mcp.tool(
    name="find_writing_examples",
    annotations=ToolAnnotations(
        title="Find Writing Examples",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def find_writing_examples(params: FindExamplesInput) -> str:
    """Search the indexed corpus for representative real excerpts (few-shot).

    Args:
        params: ``query`` (required, non-empty), ``max_results`` (1-20), and
            optional ``profile``.

    Returns:
        JSON ``{ status, query, result_count, results: [{excerpt,
        source_filename, score}] }`` with basenames only (no absolute paths).
    """
    logger.info(
        "TOOL find_writing_examples | query=%r max_results=%d",
        params.query,
        params.max_results,
    )
    mismatch = _profile_mismatch_payload(params.profile)
    if mismatch is not None:
        return json.dumps(mismatch)
    if not await _await_ready() and _build_in_flight():
        return json.dumps(_building_payload())

    if _index is None:
        return json.dumps({"status": "error", "error": "Index not initialized."})

    results = _index.search(params.query, max_results=params.max_results)
    payload = {
        "status": READINESS_READY,
        "query": params.query,
        "result_count": len(results),
        "results": [
            {
                "excerpt": r["excerpt"],
                "source_filename": r["source_filename"],
                "score": round(r["score"], 4),
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


@mcp.tool(
    name="style_status",
    annotations=ToolAnnotations(
        title="Style Profile Status",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def style_status(params: StatusInput) -> str:
    """Report profile/index health and readiness.

    Args:
        params: ``profile`` (optional; defaults to the active profile).

    Returns:
        JSON with ``built`` (bool), ``doc_count``, ``last_built`` (ISO8601),
        ``sources`` (basenames), ``readiness`` (ready/building), ``llm_enabled``,
        and the active profile name. This tool never blocks on readiness.
    """
    logger.info("TOOL style_status")
    mismatch = _profile_mismatch_payload(params.profile)
    if mismatch is not None:
        return json.dumps(mismatch)
    readiness = READINESS_READY if _profile_ready.is_set() else READINESS_BUILDING
    built = _profile is not None
    meta = _profile.metadata if _profile is not None else {}

    payload = {
        "built": built,
        "readiness": readiness,
        "profile": _config.profile_name if _config else (params.profile or "default"),
        "doc_count": meta.get("doc_count", 0),
        "last_built": meta.get("generated_at"),
        "sources": meta.get("sources", []),
        "llm_enabled": _config.llm_enabled if _config else False,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Test-support helpers (not part of the MCP surface)
# ---------------------------------------------------------------------------


def _reset_state_for_tests() -> None:
    """Reset all module-level server state. Used by the test suite for isolation."""
    global _config, _index, _watcher, _profile, _build_task
    if _watcher is not None:
        try:
            _watcher.stop()
        except Exception:
            pass
    if _index is not None:
        try:
            _index.close()
        except Exception:
            pass
    _config = None
    _index = None
    _watcher = None
    _profile = None
    _build_task = None
    # Tests use distinct tmp_path corpora, so a stale cache entry from a prior
    # test (same basename, different tmp dir) must never leak across tests.
    _extract_cache.clear()
    # Recreate the readiness event so it is not bound to a closed event loop
    # from a previous test; a fresh Event binds lazily to the current loop.
    global _profile_ready
    _profile_ready = asyncio.Event()


def _install_config_for_tests(config: StyleConfig) -> None:
    """Install a config + fresh excerpt index without starting a build/watcher.

    Lets tests exercise the pre-build ("building"/"not_built") paths and then
    opt into a build via :func:`_build_corpus_async`.
    """
    global _config, _index, _profile
    _config = config
    _index = ExcerptIndex(config.examples_db_path)
    _profile = StyleProfile.load(config.profile_path)
    if _profile is not None:
        _profile_ready.set()
    else:
        _profile_ready.clear()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _config_path_from_argv(argv: list[str]) -> Optional[str]:
    """Extract the optional config-file path from CLI arguments.

    The server is launched as ``write-like-me-mcp [CONFIG_PATH]`` (the README's
    primary form, and what ``claude mcp add ... -- write-like-me-mcp <path>`` /
    Claude Desktop ``args`` produce). The first positional argument, if present,
    is the config-file path (discovery Priority 1).

    Args:
        argv: Argument vector including the program name at index 0.

    Returns:
        The config path, or ``None`` when no positional argument was given.
    """
    positionals = [a for a in argv[1:] if not a.startswith("-")]
    return positionals[0] if positionals else None


def main() -> None:
    """Console entry point: run the MCP server over stdio.

    Parses an optional config-file path positional argument and threads it into
    the lifespan's ``load_config`` as discovery Priority 1.
    """
    global _cli_config_path
    _cli_config_path = _config_path_from_argv(sys.argv)
    if _cli_config_path:
        logger.info("Using config path from CLI argument: %s", _cli_config_path)
    mcp.run()


if __name__ == "__main__":
    main()
