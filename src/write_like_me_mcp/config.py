"""Configuration for the Write-Like-Me MCP Server.

Loads and validates the dedicated ``write-like-me.json`` config file, selects the
active writing-style profile, and resolves all runtime paths.

The config schema is *multi-profile-ready*: a ``profiles`` map plus an
``active_profile`` selector ship in v0.1, but only the active profile is built
and served. A future version can select among profiles without a schema change.

Discovery priority (first match wins):
    1. Explicit ``config_path`` argument (e.g. a CLI flag).
    2. ``WRITE_LIKE_ME_CONFIG`` environment variable (points at a config *file*).
    3. ``~/.write-like-me.json`` in the user's home directory.

Privacy note: path resolution (expanduser/resolve) happens at runtime only.
Absolute author paths are never persisted by this module — resolved paths live
solely on the in-memory ``StyleConfig`` instance.

All diagnostic logging goes to **stderr**; stdout is reserved for the MCP
protocol channel.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Named constants (no magic literals)
# ---------------------------------------------------------------------------

#: Name of the dedicated config file looked up in the user's home directory.
CONFIG_FILENAME: str = ".write-like-me.json"

#: Environment variable naming a config *file* path (discovery priority 2).
CONFIG_ENV_VAR: str = "WRITE_LIKE_ME_CONFIG"

#: Default per-profile data root directory name, created under the user's home
#: when no ``data_dir`` override is provided.
DEFAULT_DATA_ROOT_NAME: str = ".write-like-me"

#: Filename of the persisted, human-readable style profile within a profile dir.
PROFILE_FILENAME: str = "profile.json"

#: Filename of the SQLite FTS5 excerpt index within a profile dir.
EXAMPLES_DB_FILENAME: str = "examples.db"

#: Default analyzed file extensions. ``.rtf`` is intentionally excluded in v0.1
#: (the plain-text reader would leak RTF control codes into the corpus and
#: corrupt punctuation / TTR / n-gram metrics; proper RTF handling is deferred).
DEFAULT_FILE_EXTENSIONS: list[str] = [
    ".md",
    ".markdown",
    ".txt",
    ".docx",
    ".pdf",
    ".html",
    ".htm",
]

#: Default (empty) exclusion list when a profile omits ``exclude``.
DEFAULT_EXCLUDE: list[str] = []


# ---------------------------------------------------------------------------
# Exclusion matching
# ---------------------------------------------------------------------------


def is_excluded(file_path: Path, patterns: Iterable[str]) -> bool:
    """Return whether ``file_path`` matches any exclusion pattern.

    A pattern matches when it equals a single path component, or when it is a
    glob (``fnmatch`` syntax) matching a component or the whole path. Bare
    substrings do not match, so ``"notes"`` cannot silently drop
    ``my-notes-archive/``.

    Args:
        file_path: Candidate corpus path.
        patterns: Exclusion patterns from the active profile.

    Returns:
        ``True`` when the path should be excluded from the scan.
    """
    parts = file_path.parts
    whole = file_path.as_posix()
    for raw in patterns:
        # A trailing separator is how users write "this directory" (and how
        # the README documents it); it is not part of the component name.
        pattern = raw.strip().rstrip("/\\")
        if not pattern:
            continue
        if pattern in parts:
            return True
        if any(fnmatch(part, pattern) for part in parts):
            return True
        if fnmatch(whole, pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when the configuration is missing, malformed, or invalid.

    Carries a clear, user-facing message describing how to fix the problem.
    """


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class StyleConfig:
    """Resolved settings for the single active writing-style profile.

    Instances are produced by :func:`load_config`. All paths are already
    expanded and resolved; consumers should treat this object as read-only
    runtime state and must not persist its absolute paths.

    Attributes:
        resolved_sources: Source dirs/files, expanduser + resolve applied.
        exclude: Path substrings / globs excluded from the scan.
        file_extensions: Analyzed file extensions (defaults exclude ``.rtf``).
        profile_name: Name of the active profile.
        data_root: Resolved profile data root (``data_dir`` or ``~/.write-like-me``).
        profile_dir: ``<data_root>/<profile_name>/``.
        profile_path: ``<profile_dir>/profile.json``.
        examples_db_path: ``<profile_dir>/examples.db``.
        llm_enabled: Whether the reserved server-side LLM path is enabled
            (always ``False`` in v0.1 unless explicitly set in config).
    """

    resolved_sources: list[Path]
    exclude: list[str]
    file_extensions: list[str]
    profile_name: str
    data_root: Path
    profile_dir: Path
    profile_path: Path
    examples_db_path: Path
    llm_enabled: bool


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _discover_config_path(config_path: str | None) -> Path:
    """Resolve the config file path via the discovery chain.

    Args:
        config_path: Explicit path (highest priority), or ``None``.

    Returns:
        The resolved :class:`~pathlib.Path` to the config file. The file is not
        guaranteed to exist; existence is validated by the caller so a precise
        error can be raised.
    """
    # Priority 1: explicit argument (e.g. CLI flag).
    if config_path:
        return Path(config_path).expanduser()

    # Priority 2: environment variable pointing at a config file.
    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()

    # Priority 3: ~/.write-like-me.json
    return Path.home() / CONFIG_FILENAME


def _read_config_file(path: Path) -> dict:
    """Read and JSON-parse the config file, raising :class:`ConfigError`.

    Args:
        path: Path to the config file (discovered by the caller).

    Returns:
        The parsed configuration as a dict.

    Raises:
        ConfigError: If the file is missing, unreadable, not valid JSON, or
            not a JSON object.
    """
    if not path.exists():
        raise ConfigError(
            f"No configuration file found at {path}.\n"
            f"Create a '{CONFIG_FILENAME}' file describing your writing sources. "
            f"You can place it at ~/{CONFIG_FILENAME}, point the "
            f"{CONFIG_ENV_VAR} environment variable at it, or pass its path "
            f"explicitly. Minimal example:\n"
            f'  {{"active_profile": "default", '
            f'"profiles": {{"default": {{"sources": ["~/writing"]}}}}}}'
        )

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Config file {path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {path} must contain a JSON object at the top level."
        )

    return data


def _select_active_profile(data: dict) -> tuple[str, dict]:
    """Select and validate the active profile from the parsed config.

    Args:
        data: The parsed top-level config object.

    Returns:
        A ``(profile_name, profile_dict)`` tuple for the active profile.

    Raises:
        ConfigError: If ``profiles`` is missing/empty, ``active_profile`` is
            missing, or ``active_profile`` does not name an existing profile.
    """
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ConfigError(
            "Config must define a non-empty 'profiles' object mapping profile "
            "names to their settings."
        )

    active_name = data.get("active_profile")
    if not active_name:
        raise ConfigError(
            "Config must specify 'active_profile' naming which entry in "
            f"'profiles' to use. Available profiles: {sorted(profiles)}."
        )

    if active_name not in profiles:
        raise ConfigError(
            f"active_profile '{active_name}' is not defined in 'profiles'. "
            f"Available profiles: {sorted(profiles)}."
        )

    profile = profiles[active_name]
    if not isinstance(profile, dict):
        raise ConfigError(
            f"Profile '{active_name}' must be a JSON object."
        )

    return active_name, profile


def _resolve_data_root(data: dict) -> Path:
    """Resolve the profile data root from an optional ``data_dir`` override.

    Args:
        data: The parsed top-level config object.

    Returns:
        The resolved data root path (``data_dir`` if provided, else
        ``~/<DEFAULT_DATA_ROOT_NAME>``).
    """
    data_dir = data.get("data_dir")
    if data_dir:
        return Path(data_dir).expanduser().resolve()
    return (Path.home() / DEFAULT_DATA_ROOT_NAME).resolve()


def load_config(config_path: str | None = None) -> StyleConfig:
    """Load, validate, and resolve the active writing-style configuration.

    Discovery priority: explicit ``config_path`` > ``WRITE_LIKE_ME_CONFIG``
    env var (a config file path) > ``~/.write-like-me.json``.

    Args:
        config_path: Optional explicit path to a config file (highest priority).

    Returns:
        A fully resolved :class:`StyleConfig` for the active profile.

    Raises:
        ConfigError: If no config is found, the JSON is malformed, the active
            profile is unresolvable, or the active profile has no sources.
    """
    path = _discover_config_path(config_path)
    data = _read_config_file(path)

    profile_name, profile = _select_active_profile(data)

    # Validate sources: the active profile must declare at least one source.
    raw_sources = profile.get("sources") or []
    if not isinstance(raw_sources, list) or len(raw_sources) == 0:
        raise ConfigError(
            f"Active profile '{profile_name}' must declare at least one source "
            f"directory or file under 'sources'."
        )

    resolved_sources = [Path(s).expanduser().resolve() for s in raw_sources]

    exclude = profile.get("exclude")
    if not isinstance(exclude, list):
        exclude = list(DEFAULT_EXCLUDE)

    file_extensions = profile.get("file_extensions")
    if not isinstance(file_extensions, list) or len(file_extensions) == 0:
        file_extensions = list(DEFAULT_FILE_EXTENSIONS)

    data_root = _resolve_data_root(data)
    profile_dir = data_root / profile_name
    profile_path = profile_dir / PROFILE_FILENAME
    examples_db_path = profile_dir / EXAMPLES_DB_FILENAME

    # llm block is reserved in v0.1; default disabled when absent or malformed.
    llm_block = data.get("llm")
    llm_enabled = bool(
        isinstance(llm_block, dict) and llm_block.get("enabled", False)
    )

    return StyleConfig(
        resolved_sources=resolved_sources,
        exclude=exclude,
        file_extensions=file_extensions,
        profile_name=profile_name,
        data_root=data_root,
        profile_dir=profile_dir,
        profile_path=profile_path,
        examples_db_path=examples_db_path,
        llm_enabled=llm_enabled,
    )
