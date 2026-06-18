"""Tests for write_like_me_mcp.watcher — the multi-source, corpus-global watcher.

These tests exercise the public surface of ``WorkspaceWatcher`` and its
debounced handler:
    - a file change under a watched source root fires the (injected, short)
      debounce and invokes the corpus-global rebuild hook exactly once
    - the rebuild hook is a corpus-GLOBAL callback (zero-arg ``on_change``),
      NOT a per-file ``update_file``/``remove_file`` mutation (M1 audit fix)
    - one Observer schedule is created per configured source root
    - irrelevant changes (wrong extension, excluded dir, hidden/temp files)
      do not trigger the rebuild hook

The watcher is decoupled from the analyzer/indexer: it never imports them and
only calls the ``on_change`` callable supplied by the caller. To avoid flaky
real-filesystem-event timing, these tests drive the handler directly by
invoking its ``on_modified``/``on_created``/``on_deleted`` methods with
synthetic events, and use a very short injected debounce.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace


from write_like_me_mcp.config import load_config
from write_like_me_mcp.watcher import (
    DEFAULT_DEBOUNCE_SECONDS,
    WorkspaceWatcher,
    _DebouncedHandler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(path: Path, *, is_directory: bool = False, dest_path: str | None = None):
    """Build a minimal synthetic watchdog-like event object."""
    ns = SimpleNamespace(src_path=str(path), is_directory=is_directory)
    if dest_path is not None:
        ns.dest_path = dest_path
    return ns


def _make_config(tmp_path: Path, monkeypatch, sources: list[str], **profile_extra):
    """Build a real resolved ``StyleConfig`` via ``load_config`` from a temp file."""
    cfg = {
        "active_profile": "default",
        "profiles": {"default": {"sources": sources, **profile_extra}},
        "data_dir": str(tmp_path / "data"),
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    return load_config(config_path=str(cfg_file))


# ---------------------------------------------------------------------------
# Debounced handler -> corpus-global rebuild hook
# ---------------------------------------------------------------------------


def test_file_change_fires_debounced_rebuild_hook(tmp_path, monkeypatch):
    """A relevant file change fires the injected short debounce and calls on_change."""
    src = tmp_path / "writing"
    src.mkdir()
    config = _make_config(tmp_path, monkeypatch, [str(src)])

    fired = threading.Event()
    calls: list[int] = []

    def on_change() -> None:
        calls.append(1)
        fired.set()

    handler = _DebouncedHandler(config, on_change, debounce_seconds=0.05)
    handler.on_modified(_event(src / "essay.md"))

    assert fired.wait(timeout=2.0), "rebuild hook was never invoked"
    assert calls == [1], "rebuild hook should fire exactly once for a single burst"


def test_rebuild_hook_is_corpus_global_not_per_file(tmp_path, monkeypatch):
    """Multiple file events in one burst collapse into ONE corpus-global rebuild.

    Asserts the M1 audit fix: the watcher triggers a single global rebuild via the
    zero-arg ``on_change`` hook rather than per-file mutations. The spy callback
    takes no arguments, proving no per-file path is threaded through.
    """
    src = tmp_path / "writing"
    src.mkdir()
    config = _make_config(tmp_path, monkeypatch, [str(src)])

    fired = threading.Event()
    calls: list[tuple] = []

    def on_change(*args, **kwargs) -> None:
        # Capture any args to prove the contract is zero-arg (corpus-global).
        calls.append((args, kwargs))
        fired.set()

    handler = _DebouncedHandler(config, on_change, debounce_seconds=0.08)

    # A burst of heterogeneous events (create + modify + delete) in quick succession.
    handler.on_created(_event(src / "a.md"))
    handler.on_modified(_event(src / "b.txt"))
    handler.on_deleted(_event(src / "c.md"))

    assert fired.wait(timeout=2.0), "rebuild hook was never invoked"
    assert len(calls) == 1, "burst of events must collapse into a single rebuild"
    assert calls[0] == ((), {}), "on_change must be invoked with no arguments (corpus-global)"


def test_irrelevant_changes_do_not_trigger_rebuild(tmp_path, monkeypatch):
    """Directory events, unsupported extensions, and temp files are ignored."""
    src = tmp_path / "writing"
    src.mkdir()
    config = _make_config(tmp_path, monkeypatch, [str(src)], exclude=["node_modules"])

    calls: list[int] = []

    def on_change() -> None:
        calls.append(1)

    handler = _DebouncedHandler(config, on_change, debounce_seconds=0.05)

    handler.on_modified(_event(src / "subdir", is_directory=True))  # directory
    handler.on_modified(_event(src / "image.png"))                  # unsupported ext
    handler.on_modified(_event(src / "~$essay.docx"))               # office temp
    handler.on_modified(_event(src / ".hidden.md"))                 # hidden
    handler.on_modified(_event(src / "node_modules" / "x.md"))      # excluded dir

    # Give any (erroneously) scheduled timer a chance to fire.
    import time

    time.sleep(0.15)
    assert calls == [], "no rebuild should be triggered by irrelevant changes"


# ---------------------------------------------------------------------------
# Multi-source observer scheduling
# ---------------------------------------------------------------------------


def test_one_observer_schedule_per_source_root(tmp_path, monkeypatch):
    """Each configured source root gets exactly one Observer.schedule call.

    A file source is watched via its parent directory; a directory source is
    watched directly. We patch the Observer to record scheduled watch paths
    rather than touching the real filesystem-event machinery.
    """
    dir_src = tmp_path / "writing"
    dir_src.mkdir()
    file_src = tmp_path / "letters" / "intro.md"
    file_src.parent.mkdir()
    file_src.write_text("hello", encoding="utf-8")

    config = _make_config(tmp_path, monkeypatch, [str(dir_src), str(file_src)])

    scheduled: list[str] = []

    class FakeObserver:
        def __init__(self) -> None:
            self.daemon = False

        def schedule(self, handler, path, recursive=False):  # noqa: D401
            scheduled.append(path)

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            pass

    monkeypatch.setattr("write_like_me_mcp.watcher.Observer", FakeObserver)

    watcher = WorkspaceWatcher(config, lambda: None, debounce_seconds=0.05)
    watcher.start()

    assert watcher.is_running is True
    assert len(scheduled) == 2, "expected one schedule per source root"
    # dir source watched directly; file source watched via its parent dir.
    assert str(dir_src.resolve()) in scheduled
    assert str(file_src.parent.resolve()) in scheduled

    watcher.stop()
    assert watcher.is_running is False


def test_default_debounce_constant_exposed():
    """The named debounce constant exists and matches the spec default (2s)."""
    assert DEFAULT_DEBOUNCE_SECONDS == 2.0
