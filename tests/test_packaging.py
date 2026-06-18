"""Packaging and OSS-safety baseline tests for write-like-me-mcp.

These tests assert the repo scaffold is correct: version exposure, pyproject
metadata, gitignore safety patterns, MIT license, and CI/secret-scan workflows.

NOTE: The console-script entry point ``write_like_me_mcp.server:main`` is asserted
as a *string in pyproject metadata only*. ``server.py`` is implemented in a later
task group, so this test deliberately does NOT import it at runtime.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"
LICENSE_PATH = REPO_ROOT / "LICENSE"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

EXPECTED_VERSION = "0.1.0"
EXPECTED_ENTRY_POINT = "write_like_me_mcp.server:main"
EXPECTED_WHEEL_PACKAGE = "src/write_like_me_mcp"
REQUIRED_GITIGNORE_PATTERNS = [
    "write-like-me.json",
    ".write-like-me/",
    "profile.json",
    "*.db",
]


def _load_pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as fh:
        return tomllib.load(fh)


def test_version_importable_and_correct() -> None:
    """__version__ is importable from the package and equals 0.1.0."""
    src_path = str(REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import write_like_me_mcp

    assert write_like_me_mcp.__version__ == EXPECTED_VERSION


def test_pyproject_metadata() -> None:
    """pyproject parses; requires-python >=3.10; console script + wheel package set."""
    data = _load_pyproject()

    project = data["project"]
    assert project["name"] == "write-like-me-mcp"
    assert ">=3.10" in project["requires-python"]

    scripts = project["scripts"]
    assert scripts["write-like-me-mcp"] == EXPECTED_ENTRY_POINT

    wheel_packages = data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert EXPECTED_WHEEL_PACKAGE in wheel_packages


def test_gitignore_contains_safety_patterns() -> None:
    """.gitignore contains each required OSS-safety pattern string."""
    contents = GITIGNORE_PATH.read_text(encoding="utf-8")
    for pattern in REQUIRED_GITIGNORE_PATTERNS:
        assert pattern in contents, f"missing gitignore pattern: {pattern}"


def test_license_is_mit_with_copyright_filled() -> None:
    """LICENSE is MIT and the copyright holder line is filled in (not blank)."""
    contents = LICENSE_PATH.read_text(encoding="utf-8")
    assert "MIT License" in contents

    copyright_lines = [
        line for line in contents.splitlines() if line.startswith("Copyright (c)")
    ]
    assert copyright_lines, "no copyright line found"
    copyright_line = copyright_lines[0].strip()
    # Must contain more than just "Copyright (c) 2026" with nothing after the year.
    remainder = copyright_line.replace("Copyright (c)", "").strip()
    assert remainder, "copyright line is blank"
    # A bare year is not enough; a holder must be named.
    assert not remainder.isdigit(), "copyright holder not filled in (year only)"


def test_workflow_files_exist() -> None:
    """Both CI and secret-scan workflow files exist."""
    assert (WORKFLOWS_DIR / "secret-scan.yml").is_file()
    assert (WORKFLOWS_DIR / "ci.yml").is_file()
