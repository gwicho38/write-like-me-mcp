"""OSS-safety acceptance suite for write-like-me-mcp (AC1-AC5).

This module is the project's *reason for existing*: write-like-me-mcp learns a
person's private writing voice from their own documents, so the overriding
guarantee is that **nothing private ever escapes the machine or the repo**. Each
test here encodes one OSS-safety acceptance criterion from the spec as a
machine-checked regression guard:

  * **AC1 — no corpus content in build artifacts.** ``uv build`` produces an
    sdist + wheel that contain only the ``write_like_me_mcp`` package and
    packaging metadata: no corpus files, no ``profile.json``, no ``*.db``.
  * **AC2 — default data dir lives outside the repo.** The resolved default
    ``data_root`` (``~/.write-like-me``) is not under the repo / git toplevel,
    and no committed file hard-codes an absolute user path.
  * **AC2/AC4 — in-repo overrides are gitignored.** A user who keeps their
    config / profile / index *inside* a clone (``.write-like-me/profile.json``,
    ``examples.db``, ``write-like-me.json``) has those paths ignored by git.
  * **AC3 — no network when the LLM path is disabled.** The full
    analyze -> profile -> all-five-tools flow completes with every socket
    constructor patched to raise; no connection is ever attempted.
  * **AC5 — secret scanning is wired up.** ``secret-scan.yml`` runs both
    TruffleHog and Gitleaks.

AC6 ("no verbatim corpus text in the persisted profile") is covered separately
in ``tests/test_no_verbatim_corpus.py`` and is intentionally NOT duplicated here.

All tests are hermetic: builds go to a tmp dir, config/data dirs go to
``tmp_path``, and ``HOME`` is monkeypatched so nothing touches the real home.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from write_like_me_mcp import server  # noqa: E402
from write_like_me_mcp.config import (  # noqa: E402
    DEFAULT_DATA_ROOT_NAME,
    load_config,
)

# ---------------------------------------------------------------------------
# Named constants (no magic literals)
# ---------------------------------------------------------------------------

#: The package directory that is the ONLY source tree allowed inside artifacts.
PACKAGE_DIR_NAME = "write_like_me_mcp"

#: Substrings that must never appear inside a built artifact's file list. These
#: are the private-content footprints: a persisted profile, the excerpt index,
#: the user config file, and the test corpus / fixtures.
FORBIDDEN_ARTIFACT_SUBSTRINGS = (
    "profile.json",
    ".db",
    "write-like-me.json",
    "corpus",
    "fixtures",
    "/tests/",
    "examples.db",
)

#: Absolute-path prefixes that must never be hard-coded in a committed file.
ABSOLUTE_PATH_MARKERS = ("/Users/", "/home/", "C:\\")

#: Repo-relative paths a user might create *inside* a clone; all must be ignored.
IN_REPO_IGNORED_PATHS = (
    ".write-like-me/profile.json",
    "examples.db",
    "write-like-me.json",
)

#: The secret-scan workflow and the two scanners it must configure.
SECRET_SCAN_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "secret-scan.yml"
REQUIRED_SCANNERS = ("trufflehog", "gitleaks")


# ---------------------------------------------------------------------------
# AC1 — no corpus content in build artifacts
# ---------------------------------------------------------------------------


def _artifact_member_names(artifact: Path) -> list[str]:
    """Return the archive member names inside an sdist (.tar.gz) or wheel (.zip)."""
    if artifact.suffixes[-2:] == [".tar", ".gz"] or artifact.suffix == ".gz":
        with tarfile.open(artifact, "r:gz") as tf:
            return tf.getnames()
    with zipfile.ZipFile(artifact) as zf:
        return zf.namelist()


def test_build_artifacts_contain_no_corpus_or_private_content(tmp_path: Path) -> None:
    """AC1: ``uv build`` artifacts hold only the package + metadata, no private data.

    Builds both the sdist and the wheel into a tmp dir (never the repo's own
    ``dist/``), then asserts every member belongs to the ``write_like_me_mcp``
    package or is packaging metadata, and that no forbidden private footprint
    (corpus, fixtures, profile.json, *.db, the user config) appears.
    """
    out_dir = tmp_path / "dist"
    proc = subprocess.run(
        ["uv", "build", "--out-dir", str(out_dir)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"uv build failed:\n{proc.stdout}\n{proc.stderr}"

    artifacts = list(out_dir.glob("*.whl")) + list(out_dir.glob("*.tar.gz"))
    assert len(artifacts) >= 2, f"expected sdist + wheel, got: {artifacts}"

    for artifact in artifacts:
        names = _artifact_member_names(artifact)
        assert names, f"artifact {artifact.name} is empty"

        for name in names:
            lowered = name.lower()
            for forbidden in FORBIDDEN_ARTIFACT_SUBSTRINGS:
                assert forbidden not in lowered, (
                    f"{artifact.name} contains forbidden private content "
                    f"'{forbidden}' via member '{name}'"
                )

        # Every payload member must live under the package dir; the rest must be
        # recognized packaging metadata (sdist top-dir, dist-info, PKG-INFO, etc.).
        payload = [
            n
            for n in names
            if not n.endswith("/")
            and "dist-info" not in n
            and ".egg-info" not in n
            and not n.endswith("PKG-INFO")
            and not n.endswith("METADATA")
            and not n.endswith("RECORD")
            and not n.endswith("WHEEL")
            and not n.endswith("entry_points.txt")
            and not n.endswith("pyproject.toml")
        ]
        for member in payload:
            assert PACKAGE_DIR_NAME in member, (
                f"{artifact.name} ships unexpected non-package member: {member}"
            )


# ---------------------------------------------------------------------------
# AC2 — default data dir resolves outside the repo; no hard-coded user paths
# ---------------------------------------------------------------------------


def test_default_data_root_is_outside_repo(tmp_path: Path, monkeypatch) -> None:
    """AC2: the default data root (~/.write-like-me) is not under the repo root.

    Points ``HOME`` at a tmp dir, loads a config with no ``data_dir`` override,
    and asserts the resolved data root is the home-anchored default and is not a
    descendant of the repo / git toplevel (so a fresh clone never writes the
    user's private profile into the working tree).
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    source_dir = tmp_path / "writing"
    source_dir.mkdir()
    (source_dir / "a.md").write_text("Hello world.", encoding="utf-8")

    cfg = {
        "active_profile": "default",
        "profiles": {"default": {"sources": [str(source_dir)]}},
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

    config = load_config(config_path=str(cfg_file))

    expected_root = (fake_home / DEFAULT_DATA_ROOT_NAME).resolve()
    assert config.data_root == expected_root

    # The resolved data root must NOT be inside the repository tree.
    repo_resolved = REPO_ROOT.resolve()
    assert repo_resolved not in config.data_root.parents
    assert config.data_root != repo_resolved


#: Prefixes whose contents are scanned for hard-coded absolute paths: the
#: server's shippable source, packaging, workflows, and the config fixtures
#: (real config samples that ship-like behavior depends on). Deliberately
#: excluded: docs (README.md / CLAUDE.md describe the no-absolute-path *rule* and
#: necessarily name the markers) and the rest of ``tests/`` (privacy unit tests
#: assert on markers like ``/Users/`` as test data). Those are not leaks.
SCAN_INCLUDE_PREFIXES = (
    "src/",
    "pyproject.toml",
    ".github/",
    "tests/fixtures/",
    ".gitignore",
)


def test_no_committed_file_hardcodes_absolute_user_path() -> None:
    """AC2: no shippable source/packaging/config file hard-codes a user path.

    Static-scans the server's committable runtime surface — ``src/``, packaging,
    workflows, and the config fixtures — for absolute path markers (``/Users/``,
    ``/home/``, ``C:\\``). Docs and non-fixture test files are out of scope (they
    reference the markers as documentation / test data, not as leaked values).
    Config fixtures use tilde (``~``) paths by design, so they pass.
    """
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    candidate_suffixes = {
        ".py",
        ".toml",
        ".md",
        ".yml",
        ".yaml",
        ".json",
        ".txt",
        ".cfg",
        ".ini",
    }

    offenders: list[str] = []
    for rel in listing.stdout.splitlines():
        rel = rel.strip()
        if not rel or not rel.startswith(SCAN_INCLUDE_PREFIXES):
            continue
        path = REPO_ROOT / rel
        if path.suffix.lower() not in candidate_suffixes or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for marker in ABSOLUTE_PATH_MARKERS:
            if marker in text:
                offenders.append(f"{rel}: contains '{marker}'")

    assert not offenders, "absolute user path(s) hard-coded:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# AC2/AC4 — in-repo override artifacts are gitignored
# ---------------------------------------------------------------------------


def test_in_repo_override_artifacts_are_gitignored() -> None:
    """AC2/AC4: config/profile/index kept inside a clone are ignored by git.

    Uses ``git check-ignore`` on repo-relative sample paths. A non-zero exit (no
    match) for any path is a privacy regression: a user's private artifact could
    be accidentally committed.
    """
    for rel in IN_REPO_IGNORED_PATHS:
        proc = subprocess.run(
            ["git", "check-ignore", rel],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0 and proc.stdout.strip() == rel, (
            f"'{rel}' is NOT gitignored (a private artifact could be committed)"
        )


# ---------------------------------------------------------------------------
# AC3 — no network when the LLM path is disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_flow_makes_no_network_connection(tmp_path: Path, monkeypatch):
    """AC3: analyze -> profile -> all-5-tools attempts no network connection.

    Patches both ``socket.socket`` and ``socket.create_connection`` to raise, so
    *any* outbound connection attempt blows up the test. With ``llm.enabled`` off
    (the v0.1 default, set explicitly here), the local-only server must drive the
    entire tool surface without ever opening a socket. This is a regression guard
    against a future LLM/telemetry code path silently phoning home.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    source_dir = tmp_path / "writing"
    source_dir.mkdir()
    (source_dir / "doc1.txt").write_text(
        "We ship code today. Tests keep us honest. Plans stay simple.\n\n"
        "We ship code today and tomorrow. The plan is simple and clear.\n",
        encoding="utf-8",
    )
    (source_dir / "doc2.txt").write_text(
        "We ship code daily. Tests keep us honest. We ship code often.\n\n"
        "Honesty always wins. The plan stays simple and we ship code.\n",
        encoding="utf-8",
    )

    cfg = {
        "active_profile": "default",
        "profiles": {"default": {"sources": [str(source_dir)]}},
        "data_dir": str(tmp_path / "data"),
        "llm": {"enabled": False},
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("WRITE_LIKE_ME_CONFIG", str(cfg_file))

    server._reset_state_for_tests()
    config = load_config(config_path=str(cfg_file))
    assert config.llm_enabled is False
    server._install_config_for_tests(config)

    def _blocked_socket(*args, **kwargs):
        raise AssertionError("network access attempted (socket.socket)")

    def _blocked_create_connection(*args, **kwargs):
        raise AssertionError("network access attempted (socket.create_connection)")

    monkeypatch.setattr(socket, "socket", _blocked_socket)
    monkeypatch.setattr(socket, "create_connection", _blocked_create_connection)

    try:
        # 1) analyze (write path) -> 2) all five tools, end to end.
        built = json.loads(
            await server.analyze_writing_style(server.AnalyzeInput())
        )
        assert built["status"] == "built"

        profile = json.loads(
            await server.get_style_profile(server.GetProfileInput())
        )
        assert profile.get("status") != "not_built"

        status = json.loads(await server.style_status(server.StatusInput()))
        assert status["built"] is True
        assert status["llm_enabled"] is False

        examples = json.loads(
            await server.find_writing_examples(
                server.FindExamplesInput(query="ship code")
            )
        )
        assert examples["status"] == "ready"

        brief = json.loads(
            await server.apply_style(
                server.ApplyStyleInput(draft_text="A short draft to compare.")
            )
        )
        assert brief["mode"] == "brief"
    finally:
        server._reset_state_for_tests()


# ---------------------------------------------------------------------------
# AC5 — secret scanning is present and configured
# ---------------------------------------------------------------------------


def test_secret_scan_workflow_configures_both_scanners() -> None:
    """AC5: secret-scan.yml exists and runs both TruffleHog and Gitleaks."""
    assert SECRET_SCAN_WORKFLOW.is_file(), "secret-scan.yml is missing"
    contents = SECRET_SCAN_WORKFLOW.read_text(encoding="utf-8").lower()
    for scanner in REQUIRED_SCANNERS:
        assert scanner in contents, f"secret-scan.yml does not configure {scanner}"
