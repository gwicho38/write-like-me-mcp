"""Thin end-to-end integration test for write-like-me-mcp.

Exercises the real, wired-up flow a user gets in production — but hermetically:

  1. A tmp config (discovered via ``WRITE_LIKE_ME_CONFIG``) points the active
     profile at a small fixture corpus and a tmp ``data_dir``.
  2. ``analyze_writing_style`` builds the profile and excerpt index.
  3. The persisted ``profile.json`` and ``examples.db`` exist on disk.
  4. ``get_style_profile`` / ``find_writing_examples`` / ``apply_style`` /
     ``style_status`` all return coherent, spec-shaped results over that build.

This complements the per-tool unit tests in ``test_server.py`` by verifying the
tools cooperate over one shared on-disk build (rather than in isolation), and
that the build artifacts land where config says they should. ``HOME`` is
monkeypatched so nothing touches the real home directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from write_like_me_mcp import server  # noqa: E402
from write_like_me_mcp.config import load_config  # noqa: E402
from write_like_me_mcp.model import StyleProfile  # noqa: E402

# A small, repeated-sentence corpus: enough signal for non-trivial metrics and
# at least one dispersion-floored signature phrase across multiple documents.
_CORPUS: dict[str, str] = {
    "essay1.txt": (
        "We ship code today. Tests keep us honest. Plans stay simple.\n\n"
        "We ship code today, and we ship it well. The plan is simple.\n"
    ),
    "essay2.txt": (
        "We ship code daily. Tests keep us honest. We ship code often.\n\n"
        "Honesty wins. The plan stays simple, so we ship code without fuss.\n"
    ),
    "essay3.txt": (
        "We ship code today. Plans change a lot, but tests keep us honest.\n\n"
        "We ship code today and tomorrow. Simple plans beat clever plans.\n"
    ),
}


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Isolate the server module's global state around the test."""
    server._reset_state_for_tests()
    yield
    server._reset_state_for_tests()


def _write_config(tmp_path: Path) -> Path:
    """Write the fixture corpus + a config file; return the config path."""
    source_dir = tmp_path / "writing"
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, text in _CORPUS.items():
        (source_dir / name).write_text(text, encoding="utf-8")

    cfg = {
        "active_profile": "default",
        "profiles": {"default": {"sources": [str(source_dir)]}},
        "data_dir": str(tmp_path / "data"),
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_file


@pytest.mark.asyncio
async def test_end_to_end_analyze_then_all_read_tools(tmp_path: Path, monkeypatch):
    """Build via analyze, then every read tool returns coherent results."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    cfg_file = _write_config(tmp_path)
    monkeypatch.setenv("WRITE_LIKE_ME_CONFIG", str(cfg_file))

    config = load_config(config_path=str(cfg_file))
    server._install_config_for_tests(config)

    # --- analyze_writing_style: build the profile + excerpt index ----------
    built = json.loads(await server.analyze_writing_style(server.AnalyzeInput()))
    assert built["status"] == "built"
    assert built["doc_count"] == len(_CORPUS)
    assert built["total_words"] > 0
    assert built["excerpts_indexed"] >= 1

    # --- artifacts exist on disk where the config said they would ----------
    assert config.profile_path.exists(), "profile.json was not persisted"
    assert config.examples_db_path.exists(), "examples.db was not persisted"
    persisted = StyleProfile.load(config.profile_path)
    assert persisted is not None
    assert persisted.metadata["doc_count"] == len(_CORPUS)

    # --- get_style_profile: full, coherent profile -------------------------
    prof = json.loads(await server.get_style_profile(server.GetProfileInput()))
    assert prof.get("status") != "not_built"
    profile = StyleProfile.from_dict(prof)
    assert profile.style_guide_md.strip() != ""
    assert profile.avg_sentence_length > 0
    assert profile.lexical_diversity > 0

    # --- find_writing_examples: real excerpts, basenames only --------------
    examples = json.loads(
        await server.find_writing_examples(
            server.FindExamplesInput(query="ship code", max_results=5)
        )
    )
    assert examples["status"] == "ready"
    assert examples["result_count"] >= 1
    for r in examples["results"]:
        assert set(r.keys()) == {"excerpt", "source_filename", "score"}
        assert "/" not in r["source_filename"]
        assert r["source_filename"] in _CORPUS
        assert r["excerpt"].strip() != ""

    # --- apply_style: a coherent rewrite brief -----------------------------
    brief = json.loads(
        await server.apply_style(
            server.ApplyStyleInput(
                draft_text=(
                    "Notwithstanding the aforementioned matters, a comprehensive "
                    "determination regarding the disposition was duly undertaken."
                )
            )
        )
    )
    assert brief["mode"] == "brief"
    assert brief["style_guide_md"].strip() != ""
    assert isinstance(brief["observations"], list) and brief["observations"]
    assert "avg_sentence_length" in brief["target_metrics"]
    assert "avg_sentence_length" in brief["draft_metrics"]

    # --- style_status: healthy, ready, llm disabled ------------------------
    status = json.loads(await server.style_status(server.StatusInput()))
    assert status["built"] is True
    assert status["readiness"] == "ready"
    assert status["doc_count"] == len(_CORPUS)
    assert status["llm_enabled"] is False
    assert status["profile"] == "default"
    for s in status["sources"]:
        assert "/" not in s  # basenames only
