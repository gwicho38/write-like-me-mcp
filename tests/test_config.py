"""Tests for write_like_me_mcp.config — discovery chain, profile selection,
path resolution, defaults, and validation errors.

These tests exercise the public surface of ``config.py``:
    - ``load_config`` discovery precedence (CLI arg > env var > ~/.write-like-me.json)
    - active-profile resolution from the ``profiles`` map
    - derived path resolution (data_root, profile_dir, profile_path, examples_db_path)
    - default ``file_extensions`` and ``llm_enabled`` behavior
    - validation errors raising ``ConfigError``
    - tolerance of unknown top-level keys
"""

import json
from pathlib import Path

import pytest

from write_like_me_mcp.config import (
    ConfigError,
    DEFAULT_DATA_ROOT_NAME,
    DEFAULT_FILE_EXTENSIONS,
    StyleConfig,
    load_config,
)


def _write_config(path: Path, data: dict) -> Path:
    """Write a config dict as JSON to ``path`` and return it."""
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _minimal_config(sources: list[str], **extra: object) -> dict:
    """Build a minimal-but-valid config payload."""
    cfg: dict = {
        "active_profile": "default",
        "profiles": {"default": {"sources": sources}},
    }
    cfg.update(extra)
    return cfg


# ---------------------------------------------------------------------------
# Discovery precedence
# ---------------------------------------------------------------------------


def test_cli_arg_takes_priority_over_env_and_home(tmp_path, monkeypatch):
    """An explicit config_path argument wins over env var and home file."""
    src = tmp_path / "writing"
    src.mkdir()

    cli_file = _write_config(tmp_path / "cli.json", _minimal_config([str(src)]))
    env_file = _write_config(
        tmp_path / "env.json", _minimal_config([str(src)], active_profile="env")
    )
    monkeypatch.setenv("WRITE_LIKE_ME_CONFIG", str(env_file))
    # Force a fake home so a stray real ~/.write-like-me.json cannot interfere.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    config = load_config(str(cli_file))

    assert config.profile_name == "default"  # came from cli_file, not env_file


def test_env_var_takes_priority_over_home(tmp_path, monkeypatch):
    """WRITE_LIKE_ME_CONFIG env var wins over ~/.write-like-me.json."""
    src = tmp_path / "writing"
    src.mkdir()

    home = tmp_path / "home"
    home.mkdir()
    _write_config(
        home / ".write-like-me.json",
        _minimal_config([str(src)], active_profile="home"),
    )
    env_file = _write_config(
        tmp_path / "env.json",
        {
            "active_profile": "envprofile",
            "profiles": {"envprofile": {"sources": [str(src)]}},
        },
    )
    monkeypatch.setenv("WRITE_LIKE_ME_CONFIG", str(env_file))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    config = load_config()

    assert config.profile_name == "envprofile"


def test_home_file_used_when_no_arg_or_env(tmp_path, monkeypatch):
    """Falls back to ~/.write-like-me.json when no arg and no env var."""
    src = tmp_path / "writing"
    src.mkdir()

    home = tmp_path / "home"
    home.mkdir()
    _write_config(home / ".write-like-me.json", _minimal_config([str(src)]))
    monkeypatch.delenv("WRITE_LIKE_ME_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    config = load_config()

    assert config.profile_name == "default"


# ---------------------------------------------------------------------------
# Active-profile resolution
# ---------------------------------------------------------------------------


def test_active_profile_selects_the_named_profile(tmp_path, monkeypatch):
    """``active_profile`` selects the matching entry from ``profiles``."""
    src_a = tmp_path / "a"
    src_a.mkdir()
    src_b = tmp_path / "b"
    src_b.mkdir()

    cfg_file = _write_config(
        tmp_path / "c.json",
        {
            "active_profile": "blog",
            "profiles": {
                "default": {"sources": [str(src_a)]},
                "blog": {"sources": [str(src_b)], "exclude": ["drafts/"]},
            },
        },
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    config = load_config(str(cfg_file))

    assert config.profile_name == "blog"
    assert config.exclude == ["drafts/"]
    assert config.resolved_sources == [src_b.resolve()]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_path_resolution_defaults_and_derived_paths(tmp_path, monkeypatch):
    """Sources are expanduser+resolved; derived paths use the default data_root."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "writing").mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    # ``Path.expanduser`` consults $HOME (not Path.home), so set it too for the
    # ``~/writing`` source in the config to resolve under the fake home.
    monkeypatch.setenv("HOME", str(home))

    cfg_file = _write_config(
        tmp_path / "c.json", _minimal_config(["~/writing"])
    )

    config = load_config(str(cfg_file))

    expected_data_root = (home / DEFAULT_DATA_ROOT_NAME).resolve()
    assert config.resolved_sources == [(home / "writing").resolve()]
    assert config.data_root == expected_data_root
    assert config.profile_dir == expected_data_root / "default"
    assert config.profile_path == expected_data_root / "default" / "profile.json"
    assert config.examples_db_path == expected_data_root / "default" / "examples.db"


def test_data_dir_override_changes_data_root(tmp_path, monkeypatch):
    """An explicit ``data_dir`` overrides the default data_root and propagates."""
    src = tmp_path / "writing"
    src.mkdir()
    custom_root = tmp_path / "custom-data"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    cfg_file = _write_config(
        tmp_path / "c.json",
        _minimal_config([str(src)], data_dir=str(custom_root)),
    )

    config = load_config(str(cfg_file))

    assert config.data_root == custom_root.resolve()
    assert config.profile_dir == custom_root.resolve() / "default"
    assert config.profile_path == custom_root.resolve() / "default" / "profile.json"
    assert (
        config.examples_db_path == custom_root.resolve() / "default" / "examples.db"
    )


# ---------------------------------------------------------------------------
# Defaults: file_extensions and llm_enabled
# ---------------------------------------------------------------------------


def test_default_file_extensions_exclude_rtf(tmp_path, monkeypatch):
    """When a profile omits ``file_extensions`` the default set is used (no .rtf)."""
    src = tmp_path / "writing"
    src.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    cfg_file = _write_config(tmp_path / "c.json", _minimal_config([str(src)]))

    config = load_config(str(cfg_file))

    assert config.file_extensions == DEFAULT_FILE_EXTENSIONS
    assert ".rtf" not in config.file_extensions
    assert ".md" in config.file_extensions
    assert ".pdf" in config.file_extensions


def test_llm_enabled_defaults_false_when_block_absent(tmp_path, monkeypatch):
    """``llm_enabled`` is False when the ``llm`` block is absent."""
    src = tmp_path / "writing"
    src.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    cfg_file = _write_config(tmp_path / "c.json", _minimal_config([str(src)]))

    config = load_config(str(cfg_file))

    assert config.llm_enabled is False


def test_llm_enabled_true_when_block_enables_it(tmp_path, monkeypatch):
    """``llm_enabled`` reflects ``llm.enabled: true`` when present."""
    src = tmp_path / "writing"
    src.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    cfg_file = _write_config(
        tmp_path / "c.json",
        _minimal_config([str(src)], llm={"enabled": True}),
    )

    config = load_config(str(cfg_file))

    assert config.llm_enabled is True


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_config_raises_clear_error(tmp_path, monkeypatch):
    """No config anywhere -> ConfigError telling the user to create the file."""
    monkeypatch.delenv("WRITE_LIKE_ME_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    assert "write-like-me.json" in str(exc_info.value)


def test_active_profile_not_in_profiles_raises(tmp_path, monkeypatch):
    """``active_profile`` pointing at a non-existent profile -> ConfigError."""
    src = tmp_path / "writing"
    src.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    cfg_file = _write_config(
        tmp_path / "c.json",
        {
            "active_profile": "ghost",
            "profiles": {"default": {"sources": [str(src)]}},
        },
    )

    with pytest.raises(ConfigError) as exc_info:
        load_config(str(cfg_file))

    assert "ghost" in str(exc_info.value)


def test_active_profile_with_zero_sources_raises(tmp_path, monkeypatch):
    """An active profile with no sources -> ConfigError."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    cfg_file = _write_config(
        tmp_path / "c.json",
        {
            "active_profile": "default",
            "profiles": {"default": {"sources": []}},
        },
    )

    with pytest.raises(ConfigError) as exc_info:
        load_config(str(cfg_file))

    assert "source" in str(exc_info.value).lower()


def test_invalid_json_raises_config_error(tmp_path, monkeypatch):
    """Malformed JSON -> ConfigError (not a raw JSONDecodeError)."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(str(bad))


# ---------------------------------------------------------------------------
# Unknown keys tolerated
# ---------------------------------------------------------------------------


def test_unknown_top_level_keys_are_tolerated(tmp_path, monkeypatch):
    """Unknown top-level keys do not crash loading."""
    src = tmp_path / "writing"
    src.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    cfg_file = _write_config(
        tmp_path / "c.json",
        _minimal_config(
            [str(src)],
            some_future_key="ignored",
            another={"nested": "value"},
        ),
    )

    config = load_config(str(cfg_file))

    assert isinstance(config, StyleConfig)
    assert config.profile_name == "default"
