"""Filesystem watcher for the Write-Like-Me MCP server.

Monitors every configured writing source root and, after a debounce quiet
period, triggers a single **corpus-global rebuild** so edits to the author's
corpus are reflected in the served style profile and excerpt index.

Design (ADAPTED from ``legal-workspace-mcp/watcher.py``):

* **Multi-source.** The legal-workspace template watches one directory. A
  Write-Like-Me profile may declare several ``sources`` that are dirs *or*
  individual files. We schedule one :class:`~watchdog.observers.Observer`
  watch per resolved source root: directories are watched directly
  (recursively); a file source is watched via its parent directory.

* **Corpus-global, NOT per-file (M1 audit fix).** The template mutated the
  index per changed file (``update_file`` / ``remove_file``). Style metrics
  (TTR, n-grams, sentence-length distributions) and BM25 corpus statistics are
  inherently corpus-global, so a per-file mutation would produce an
  inconsistent profile. Instead, any relevant change in any source debounces
  into a single zero-argument ``on_change`` call.

* **Decoupled.** This module does NOT import the analyzer or indexer. The
  caller (the server, G8) supplies an ``on_change: Callable[[], None]`` hook
  that re-runs ``analyze_corpus`` and refreshes the ``ExcerptIndex``. The
  watcher only knows *that* the corpus changed, never *how* to rebuild it.

Threading: debouncing uses a cancellable :class:`threading.Timer` guarded by a
:class:`threading.Lock`, exactly as in the template. All diagnostic logging
goes to **stderr** (stdout is reserved for the MCP protocol channel).

Contract summary:

    watcher = WorkspaceWatcher(config, on_change)   # on_change() rebuilds the corpus
    watcher.start()   # schedules one Observer watch per resolved source root
    ...               # edits to any source debounce into a single on_change()
    watcher.stop()    # clean shutdown for lifespan integration
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .config import StyleConfig

# ---------------------------------------------------------------------------
# Named constants (no magic literals)
# ---------------------------------------------------------------------------

#: Default debounce quiet period, in seconds, before a corpus rebuild fires.
#: Matches the template's 2s window; injectable via the constructor for tests.
DEFAULT_DEBOUNCE_SECONDS: float = 2.0

#: Filesystem timeout, in seconds, when joining the observer thread on stop.
_OBSERVER_JOIN_TIMEOUT: float = 5.0


# ---------------------------------------------------------------------------
# Logging to stderr (stdout is the MCP protocol channel)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(stream=sys.stderr)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.propagate = False


# ---------------------------------------------------------------------------
# Debounced, corpus-global event handler
# ---------------------------------------------------------------------------


class _DebouncedHandler(FileSystemEventHandler):
    """Collapses a burst of relevant filesystem events into one rebuild.

    Unlike the template (which tracked per-file pending updates/removes), this
    handler tracks only *whether* a relevant change occurred. After the
    debounce quiet period it invokes the corpus-global ``on_change`` hook once.

    Args:
        config: Resolved profile config; used to decide relevance (extensions,
            excluded path substrings, hidden/temp-file filtering).
        on_change: Zero-argument corpus-global rebuild hook supplied by the
            caller. Invoked from the debounce timer thread.
        debounce_seconds: Quiet period before ``on_change`` fires. Injectable
            for tests; defaults to :data:`DEFAULT_DEBOUNCE_SECONDS`.
    """

    def __init__(
        self,
        config: StyleConfig,
        on_change: Callable[[], None],
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ) -> None:
        super().__init__()
        self._config = config
        self._on_change = on_change
        self._debounce_seconds = debounce_seconds
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._pending = False

    # -- watchdog event hooks ------------------------------------------------

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._note_change(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._note_change(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._note_change(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._note_change(event.src_path)
            dest = getattr(event, "dest_path", None)
            if dest is not None:
                self._note_change(dest)

    # -- debounce machinery --------------------------------------------------

    def _note_change(self, path: str | bytes) -> None:
        """Record a relevant change and (re)arm the debounce timer."""
        # watchdog event paths are ``str | bytes`` depending on how the watch
        # was scheduled; ``os.fsdecode`` normalizes both to ``str``.
        if not self._is_relevant(Path(os.fsdecode(path))):
            return
        with self._lock:
            self._pending = True
            self._reset_timer()

    def _reset_timer(self) -> None:
        """(Re)start the debounce timer. Caller must hold ``self._lock``."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce_seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        """Invoke the corpus-global rebuild hook once for the pending burst."""
        with self._lock:
            if not self._pending:
                return
            self._pending = False

        logger.info("Corpus change detected; triggering global rebuild")
        try:
            self._on_change()
        except Exception:  # pragma: no cover - defensive; never kill the timer thread
            logger.exception("Corpus rebuild hook raised an exception")

    # -- relevance filtering -------------------------------------------------

    def _is_relevant(self, file_path: Path) -> bool:
        """Return whether a changed path should trigger a corpus rebuild."""
        name = file_path.name
        # Skip hidden files and Office temp/lock files (~$document.docx).
        if name.startswith(".") or name.startswith("~$"):
            return False
        # Only configured analyzed extensions are relevant.
        if file_path.suffix.lower() not in self._config.file_extensions:
            return False
        # Skip excluded path substrings (matched against path components).
        parts = file_path.parts
        for excluded in self._config.exclude:
            if excluded and excluded in parts:
                return False
        return True


# ---------------------------------------------------------------------------
# Public watcher
# ---------------------------------------------------------------------------


class WorkspaceWatcher:
    """Watches every configured source root and fires a corpus-global rebuild.

    Args:
        config: Resolved :class:`~write_like_me_mcp.config.StyleConfig` whose
            ``resolved_sources`` (dirs and/or files) are watched.
        on_change: Zero-argument corpus-global rebuild hook. Supplied by the
            server (G8); typically re-runs ``analyze_corpus`` and refreshes the
            ``ExcerptIndex``. The watcher never imports those itself.
        debounce_seconds: Quiet period before a rebuild fires. Defaults to
            :data:`DEFAULT_DEBOUNCE_SECONDS` (2s); injectable for tests.

    Usage:
        watcher = WorkspaceWatcher(config, on_change)
        watcher.start()
        # ... server runs ...
        watcher.stop()
    """

    def __init__(
        self,
        config: StyleConfig,
        on_change: Callable[[], None],
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ) -> None:
        self._config = config
        self._on_change = on_change
        self._debounce_seconds = debounce_seconds
        self._observer: BaseObserver | None = None
        self._running = False

    def _watch_roots(self) -> list[Path]:
        """Resolve each source to the directory that should be watched.

        Directory sources are watched directly; file sources are watched via
        their parent directory (watchdog observes directories, not files).
        Duplicate roots (e.g. several files in one dir) are de-duplicated while
        preserving order.
        """
        roots: list[Path] = []
        seen: set[Path] = set()
        for source in self._config.resolved_sources:
            root = source if source.is_dir() else source.parent
            if root not in seen:
                seen.add(root)
                roots.append(root)
        return roots

    def start(self) -> None:
        """Start one Observer watch per configured source root."""
        if self._running:
            return

        roots = self._watch_roots()
        if not roots:
            logger.warning("No source roots to watch; watcher not started")
            return

        handler = _DebouncedHandler(
            self._config, self._on_change, debounce_seconds=self._debounce_seconds
        )
        self._observer = Observer()

        scheduled = 0
        for root in roots:
            if not root.is_dir():
                logger.warning("Skipping non-existent watch root: %s", root)
                continue
            self._observer.schedule(handler, str(root), recursive=True)
            scheduled += 1
            logger.info("Watching source root: %s", root)

        if scheduled == 0:
            logger.warning("No existing source roots; watcher not started")
            self._observer = None
            return

        self._observer.daemon = True
        self._observer.start()
        self._running = True
        logger.info("Watcher started on %d source root(s)", scheduled)

    def stop(self) -> None:
        """Stop the observer cleanly (safe to call when not running)."""
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=_OBSERVER_JOIN_TIMEOUT)
            self._running = False
            logger.info("Watcher stopped")

    @property
    def is_running(self) -> bool:
        """Whether the watcher is currently running."""
        return self._running
