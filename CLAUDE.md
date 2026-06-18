# CLAUDE.md — write-like-me-mcp

Project-specific instructions for Claude Code working in this repository.

## What this project is

An MCP server that learns a user's personal writing style from a **local**
document corpus and serves self-updating style context to Claude. The defining
constraint is **privacy**: everything runs on-device, nothing crosses the
network, and no verbatim corpus text is ever persisted or shipped. Treat that as
the prime directive — any change that could leak private content or open a
network edge is a bug, not a feature.

## Architecture (where things live)

| Module | Responsibility |
|--------|----------------|
| `src/write_like_me_mcp/config.py` | Load/validate `write-like-me.json`, select active profile, resolve paths. |
| `src/write_like_me_mcp/model.py` | `StyleProfile` dataclass + `profile.json` (de)serialization. |
| `src/write_like_me_mcp/style_analyzer.py` | Compute style metrics + signature phrases from text. |
| `src/write_like_me_mcp/extractors.py` | Extract text from TXT/MD/PDF/DOCX/HTML. |
| `src/write_like_me_mcp/indexer.py` | SQLite FTS5 + BM25 excerpt index (`examples.db`). |
| `src/write_like_me_mcp/watcher.py` | Debounced watchdog corpus-change rebuilds. |
| `src/write_like_me_mcp/server.py` | FastMCP server, lifespan, the 5 tools. |

## Code conventions

These are enforced across the codebase — match them in any new code:

- **stderr-only logging.** stdout is the MCP protocol channel. All diagnostic
  output goes through the module `logger` (configured to `sys.stderr`). Never
  `print()` to stdout and never reconfigure logging to stdout.
- **Pydantic inputs are strict.** Every tool input model uses
  `ConfigDict(str_strip_whitespace=True, extra="forbid")` so unknown fields are
  rejected. Keep new tool inputs the same.
- **Tools return JSON strings.** Tool functions return `json.dumps(...)` (with
  `indent=2, ensure_ascii=False` for human-readable payloads) — not dicts and
  not bare objects.
- **Named constants, no magic literals.** Thresholds, filenames, env-var names,
  limits, and tolerances are module-level `UPPER_SNAKE_CASE` constants with a
  docstring/comment. Don't inline literals into logic.
- **Full type hints.** Every function/method is fully annotated (params +
  return). `from __future__ import annotations` is at the top of each module.
  `uv run mypy src` must stay clean.
- **No hard-coded secrets or paths.** No API keys, tokens, or absolute user
  paths (`/Users/...`, `/home/...`, `C:\...`) in committed source, config, or
  docs. Paths come from config/env and are resolved at runtime only; resolved
  absolute paths live solely on the in-memory `StyleConfig` and are never
  persisted or emitted (results use basenames / data-root-relative forms).

## Privacy invariants (do not regress)

- `profile.json` stores **derived statistics only** — no verbatim sentences;
  signature phrases are ≤3 tokens and pass a document-dispersion floor.
- Search results and status output expose **basenames only**, never absolute
  paths.
- The default data dir is `~/.write-like-me` (outside any repo clone);
  in-repo overrides (`write-like-me.json`, `profile.json`, `*.db`,
  `.write-like-me/`) are gitignored.
- **No network, no LLM call** in v0.1. The reserved `llm` config block is always
  disabled. `tests/test_oss_safety.py` guards all of the above — keep it green.

## Local CI Gate (REQUIRED)

Run the full gate locally before every push:

```bash
uv run ruff check . && uv run mypy src && uv run pytest
```

Optionally validate the GitHub workflows locally with `act` /
`mcli ci preflight`.

### Hosted-workflow policy

Hosted CI (`.github/workflows/ci.yml`) is `workflow_dispatch`-only — it has **no
auto `push` / `pull_request` / `schedule` triggers** (see the
`# mcli-ci: hosted-triggers-stripped` marker). Validate locally; trigger a hosted
run on demand with `gh workflow run ci.yml`. The exception is
`secret-scan.yml`, which is allowed to run on push/PR/schedule. Do not add hosted
auto-triggers to CI/lint/test jobs without explicit approval.

## Testing notes

- Tests are hermetic: use `tmp_path` for builds and data dirs, and monkeypatch
  `HOME` / `WRITE_LIKE_ME_CONFIG` so the real home is never written.
- Async tests use the lightweight `@pytest.mark.asyncio` shim in
  `tests/conftest.py` (no `pytest-asyncio` dependency).
- New features need tests; bug fixes need a regression test reproducing the bug.
