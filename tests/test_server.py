"""Tests for write_like_me_mcp.server — the FastMCP server and its 5 tools.

These tests exercise the public surface of the server module:

* the readiness contract — tools return a structured ``"building"`` status
  before the ``_profile_ready`` event is set, and real results after;
* the lifespan starts and stops cleanly (background build task cancelled,
  watcher stopped, SQLite index closed);
* each of the 5 tools returns spec-shaped JSON:
  - ``get_style_profile`` -> ``not_built`` before a build, full profile after;
  - ``analyze_writing_style`` -> writes profile.json + examples.db,
    returns ``status: "built"`` summary;
  - ``apply_style`` -> a rewrite brief with concrete observations, rejects
    empty ``draft_text``;
  - ``find_writing_examples`` / ``style_status`` -> basenames only;
* Pydantic inputs reject unknown fields (``extra="forbid"``);
* tool annotations match the spec (analyze is non-idempotent; the read-only
  tools are read-only + idempotent; all are ``openWorldHint=false``).

The tests call the tool implementation functions directly (the ``@mcp.tool``
decorator returns the original async function unchanged), passing constructed
Pydantic input models, and inject a hermetic config via a tmp config file +
tmp data_dir + a tiny fixture corpus. The readiness timeout is patched short so
the "building" path is exercised without a real 30s wait.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from write_like_me_mcp import server
from write_like_me_mcp.config import load_config
from write_like_me_mcp.model import StyleProfile


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

#: A tiny multi-document corpus written into a temp source dir. Repeated short
#: sentences give the analyzer enough signal to populate metrics and to let the
#: dispersion-floored signature-phrase extractor find at least one phrase.
_CORPUS_DOCS: dict[str, str] = {
    "doc1.txt": (
        "We ship code today. Tests keep us honest. Plans stay simple.\n\n"
        "We ship code today. The plan is simple and clear.\n"
    ),
    "doc2.txt": (
        "We ship code today. The plan stays simple. We ship code daily.\n\n"
        "Tests keep us honest. We ship code today and tomorrow.\n"
    ),
    "doc3.txt": (
        "We ship code often. Plans change a lot. Honesty always wins.\n\n"
        "We ship code today. Tests keep us honest and grounded.\n"
    ),
}


def _write_corpus(source_dir: Path) -> None:
    """Write the fixture corpus documents into ``source_dir``."""
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, text in _CORPUS_DOCS.items():
        (source_dir / name).write_text(text, encoding="utf-8")


def _make_config(tmp_path: Path) -> Path:
    """Create a config file pointing at a fresh corpus + tmp data dir.

    Returns the path to the written config file. The active profile is named
    ``default`` and its single source is a directory of fixture documents.
    """
    source_dir = tmp_path / "writing"
    _write_corpus(source_dir)

    cfg = {
        "active_profile": "default",
        "profiles": {"default": {"sources": [str(source_dir)]}},
        "data_dir": str(tmp_path / "data"),
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_file


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Reset the server module's global state around each test for isolation."""
    server._reset_state_for_tests()
    yield
    server._reset_state_for_tests()


@pytest.fixture
def configured(tmp_path: Path, monkeypatch):
    """Load a hermetic StyleConfig and install it as the server's active config.

    Does NOT build the profile or start the watcher — tests opt into a build via
    ``_run_build`` so they can also exercise the pre-build ("building" /
    "not_built") paths.
    """
    cfg_file = _make_config(tmp_path)
    monkeypatch.setenv("WRITE_LIKE_ME_CONFIG", str(cfg_file))
    config = load_config(config_path=str(cfg_file))
    server._install_config_for_tests(config)
    return config


async def _run_build(force_reindex: bool = False) -> None:
    """Run the server's corpus build to completion and mark readiness."""
    await server._build_corpus_async(force_reindex=force_reindex)


# ---------------------------------------------------------------------------
# Readiness contract: "building" before the event, real results after
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tools_return_building_before_ready(configured, monkeypatch):
    """While a background build is in flight, tools return a building status."""
    # Patch the readiness timeout very short so we don't wait the real 30s.
    monkeypatch.setattr(server, "READINESS_TIMEOUT_SECONDS", 0.05)
    assert not server._profile_ready.is_set()

    # Simulate an in-flight background build (never completing within the test)
    # so the readiness gate is exercised, mirroring the real lifespan window.
    async def _never_finish() -> None:
        await asyncio.sleep(3600)

    server._build_task = asyncio.ensure_future(_never_finish())
    try:
        assert server._build_in_flight() is True

        out = json.loads(await server.style_status(server.StatusInput()))
        assert out["readiness"] == "building"

        out = json.loads(
            await server.find_writing_examples(
                server.FindExamplesInput(query="ship code")
            )
        )
        assert out["status"] == "building"

        out = json.loads(await server.get_style_profile(server.GetProfileInput()))
        assert out["status"] == "building"
    finally:
        server._build_task.cancel()
        try:
            await server._build_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_tools_return_real_results_after_ready(configured):
    """After a build sets readiness, read tools serve real results, not 'building'."""
    await _run_build()
    assert server._profile_ready.is_set()

    out = json.loads(
        await server.find_writing_examples(server.FindExamplesInput(query="ship code"))
    )
    assert out["status"] == "ready"
    assert isinstance(out["results"], list)
    assert len(out["results"]) >= 1
    # Basenames only — never absolute paths.
    for r in out["results"]:
        assert "/" not in r["source_filename"]
        assert r["source_filename"] in _CORPUS_DOCS


# ---------------------------------------------------------------------------
# Lifespan: clean startup and shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_cleanly(configured, tmp_path, monkeypatch):
    """The lifespan yields a context, then cancels the task, stops watcher, closes DB."""
    cfg_file = tmp_path / "config.json"
    monkeypatch.setenv("WRITE_LIKE_ME_CONFIG", str(cfg_file))

    async with server.server_lifespan(server.mcp) as ctx:
        assert "config" in ctx
        assert "index" in ctx
        # Let the background build finish so there is a completed task to clean up.
        if server._build_task is not None:
            await asyncio.wait_for(asyncio.shield(server._build_task), timeout=10.0)

    # After the context exits, the watcher is stopped and the DB connection closed.
    assert server._watcher is None or server._watcher.is_running is False
    # The index connection is closed (close() is idempotent and nulls the conn).
    assert server._index is None or server._index._conn is None


# ---------------------------------------------------------------------------
# get_style_profile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_style_profile_not_built_then_full(configured, monkeypatch):
    """``get_style_profile`` returns not_built before a build, full profile after."""
    monkeypatch.setattr(server, "READINESS_TIMEOUT_SECONDS", 0.05)
    out = json.loads(await server.get_style_profile(server.GetProfileInput()))
    assert out["status"] == "not_built"

    await _run_build()
    out = json.loads(await server.get_style_profile(server.GetProfileInput()))
    # A full StyleProfile round-trips through the model loader.
    profile = StyleProfile.from_dict(out)
    assert profile.style_guide_md.strip() != ""
    assert profile.metadata["doc_count"] >= 1
    assert out.get("status") != "not_built"


# ---------------------------------------------------------------------------
# analyze_writing_style
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_writing_style_writes_artifacts_and_summary(configured):
    """analyze_writing_style writes profile.json + examples.db and returns built summary."""
    out = json.loads(await server.analyze_writing_style(server.AnalyzeInput()))
    assert out["status"] == "built"
    assert out["doc_count"] >= 1
    assert out["total_words"] > 0
    assert out["source_count"] >= 1
    assert "generated_at" in out
    # profile_path is a basename/relative form, never an absolute path.
    assert not Path(out["profile_path"]).is_absolute()

    # Artifacts exist on disk.
    assert configured.profile_path.exists()
    assert configured.examples_db_path.exists()

    # A subsequent read tool now serves the freshly built profile.
    prof_out = json.loads(await server.get_style_profile(server.GetProfileInput()))
    assert prof_out.get("status") != "not_built"


# ---------------------------------------------------------------------------
# apply_style
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_style_returns_brief_with_observations(configured):
    """apply_style returns the rewrite-brief shape with concrete observations."""
    await _run_build()

    long_draft = (
        "Notwithstanding the aforementioned considerations, the comprehensive "
        "evaluation of the multifaceted situation was undertaken by the committee "
        "in order that a determination could be reached regarding the eventual "
        "disposition of the relevant materials and associated documentation."
    )
    out = json.loads(
        await server.apply_style(server.ApplyStyleInput(draft_text=long_draft))
    )
    assert out["mode"] == "brief"
    assert isinstance(out["style_guide_md"], str) and out["style_guide_md"].strip()
    assert isinstance(out["observations"], list) and len(out["observations"]) >= 1
    assert isinstance(out["target_metrics"], dict)
    assert isinstance(out["draft_metrics"], dict)
    # The draft's metrics differ from the profile's, so at least one concrete
    # comparison observation must be present.
    joined = " ".join(out["observations"]).lower()
    assert any(k in joined for k in ("sentence", "contraction", "word"))


def test_apply_style_rejects_empty_draft():
    """An empty ``draft_text`` is rejected by Pydantic min_length validation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        server.ApplyStyleInput(draft_text="")
    with pytest.raises(ValidationError):
        # str_strip_whitespace makes a whitespace-only draft empty -> rejected.
        server.ApplyStyleInput(draft_text="   ")


# ---------------------------------------------------------------------------
# find_writing_examples + style_status shapes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_writing_examples_basenames_only(configured):
    """find_writing_examples returns {excerpt, source_filename, score}, basenames only."""
    await _run_build()
    out = json.loads(
        await server.find_writing_examples(
            server.FindExamplesInput(query="ship code", max_results=3)
        )
    )
    assert out["status"] == "ready"
    assert 1 <= len(out["results"]) <= 3
    for r in out["results"]:
        assert set(r.keys()) == {"excerpt", "source_filename", "score"}
        assert "/" not in r["source_filename"]


@pytest.mark.asyncio
async def test_style_status_spec_shape(configured):
    """style_status reports built/doc_count/last_built/sources/readiness/llm_enabled."""
    await _run_build()
    out = json.loads(await server.style_status(server.StatusInput()))
    assert out["built"] is True
    assert out["doc_count"] >= 1
    assert out["readiness"] == "ready"
    assert out["llm_enabled"] is False
    assert out["profile"] == "default"
    assert isinstance(out["sources"], list)
    for s in out["sources"]:
        assert "/" not in s  # basenames only


# ---------------------------------------------------------------------------
# Pydantic input models reject unknown fields (extra="forbid")
# ---------------------------------------------------------------------------


def test_input_models_forbid_unknown_fields():
    """All tool input models reject unknown fields (ConfigDict extra='forbid')."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        server.AnalyzeInput(nope=1)
    with pytest.raises(ValidationError):
        server.GetProfileInput(nope=1)
    with pytest.raises(ValidationError):
        server.ApplyStyleInput(draft_text="hi there", nope=1)
    with pytest.raises(ValidationError):
        server.FindExamplesInput(query="x", nope=1)
    with pytest.raises(ValidationError):
        server.StatusInput(nope=1)


# ---------------------------------------------------------------------------
# Tool annotations match the spec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_annotations_match_spec():
    """Annotations: analyze is non-idempotent/write; read tools are RO+idempotent."""
    tools = await server.mcp.list_tools()
    by_name = {t.name: t for t in tools}

    # All five tools registered.
    expected = {
        "analyze_writing_style",
        "get_style_profile",
        "apply_style",
        "find_writing_examples",
        "style_status",
    }
    assert expected.issubset(by_name.keys())

    # Every tool is closed-world (no network).
    for name in expected:
        assert by_name[name].annotations.openWorldHint is False

    analyze = by_name["analyze_writing_style"].annotations
    assert analyze.readOnlyHint is False
    assert analyze.idempotentHint is False
    assert analyze.destructiveHint is False

    for name in (
        "get_style_profile",
        "apply_style",
        "find_writing_examples",
        "style_status",
    ):
        ann = by_name[name].annotations
        assert ann.readOnlyHint is True
        assert ann.idempotentHint is True


# ---------------------------------------------------------------------------
# Regression: CLI-argument config path (B1) and profile-mismatch guard (W1)
# ---------------------------------------------------------------------------


def test_config_path_from_argv_parses_positional():
    """main() must extract the config path from a positional CLI argument."""
    assert server._config_path_from_argv(["write-like-me-mcp"]) is None
    assert (
        server._config_path_from_argv(["write-like-me-mcp", "/x/write-like-me.json"])
        == "/x/write-like-me.json"
    )
    # Flags are skipped; the first positional is the config path.
    assert (
        server._config_path_from_argv(["prog", "--verbose", "/x/cfg.json"])
        == "/x/cfg.json"
    )


@pytest.mark.asyncio
async def test_lifespan_uses_cli_config_path(tmp_path, monkeypatch):
    """The lifespan must honor the CLI-argument config path (discovery Priority 1).

    Regression for B1: previously main() ran mcp.run() and the lifespan called
    load_config() with no argument, so the documented `write-like-me-mcp <cfg>`
    launch crashed. Here we set ONLY the CLI path (env var cleared) and assert the
    server loads that config.
    """
    cfg_file = _make_config(tmp_path)
    monkeypatch.delenv("WRITE_LIKE_ME_CONFIG", raising=False)
    monkeypatch.setattr(server, "_cli_config_path", str(cfg_file))

    async with server.server_lifespan(server.mcp):
        assert server._config is not None
        assert server._config.profile_name == "default"
        # Resolved data dir lives under the tmp config's data_dir.
        assert str(tmp_path) in str(server._config.data_root)


@pytest.mark.asyncio
async def test_non_active_profile_returns_error(configured):
    """Tools must reject a non-active profile rather than silently using active (W1)."""
    await _run_build()
    raw = await server.get_style_profile(
        server.GetProfileInput(profile="some_other_profile")
    )
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert "active profile" in payload["error"].lower()

    # The active profile name still works.
    raw_ok = await server.get_style_profile(server.GetProfileInput(profile="default"))
    assert json.loads(raw_ok).get("status") != "error"
