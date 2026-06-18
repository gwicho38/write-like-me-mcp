# write-like-me-mcp

An MCP (Model Context Protocol) server that learns **your personal writing
style** from a local document corpus and gives Claude **self-updating style
context** — so the prose it produces on your behalf actually sounds like *you*.

**100% local. Nothing leaves your machine.** The server reads your documents
on-device, derives statistical style metrics, and serves them to Claude over
stdio. There is **no network call and no LLM code path** in v0.1 — the actual
rewriting is done by the Claude you're already talking to, guided by the style
context this server provides.

## How It Works

1. **You point it at your writing** — a directory (or set of files) of things
   you've written: essays, emails, posts, docs, notes.
2. **It learns your voice locally** — sentence rhythm, lexical diversity,
   contraction habits, punctuation tics, signature phrases, formality markers —
   and writes a `profile.json` plus a searchable excerpt index (`examples.db`).
3. **It watches for changes** — edit or add to your corpus and the profile and
   index rebuild automatically (debounced).
4. **Claude uses it as style context** — before writing for you, Claude loads
   your profile and can pull real excerpts of your writing for grounding, then
   matches your voice. You never upload anything.

## Privacy & Local-Only Posture

This is the whole point of the project, so it is worth being explicit:

| Guarantee | How it's enforced |
|-----------|-------------------|
| No network access | The server has no HTTP client and no LLM call. A regression test runs the full tool flow with all sockets patched to raise. |
| No verbatim corpus stored | `profile.json` holds derived statistics + short (≤3-token) signature phrases only — never your sentences. Enforced by a dispersion floor + a dedicated test. |
| Excerpts stay local | `find_writing_examples` serves excerpts live from a local SQLite index and returns source **basenames only** — never absolute paths. |
| Private data stays out of the repo | The default data dir is `~/.write-like-me` (outside any clone). If you keep config/profile/index inside a clone, those paths are gitignored. |
| Clean build artifacts | `uv build` ships only the package + packaging metadata — no corpus, no `profile.json`, no `*.db`. Verified by a test. |
| Secret scanning | `secret-scan.yml` runs both TruffleHog and Gitleaks. |

## Install

> **Requirement**: Python ≥ 3.10. You must tell the server where your writing
> lives — via a config file, the `WRITE_LIKE_ME_CONFIG` env var, or
> `~/.write-like-me.json`.

### Option A: Install from GitHub

```bash
pip install git+https://github.com/gwicho38/write-like-me-mcp.git
```

### Option B: Run without installing (uvx)

```bash
uvx write-like-me-mcp /path/to/your/write-like-me.json
```

### Option C: Clone and install locally

```bash
git clone https://github.com/gwicho38/write-like-me-mcp.git
cd write-like-me-mcp
pip install -e .
```

## Configuration

The server is driven by a small JSON config that names your writing sources and
which profile is active.

### `write-like-me.json` example

```json
{
  "active_profile": "default",
  "data_dir": "~/.write-like-me",
  "profiles": {
    "default": {
      "sources": ["~/writing", "~/notes"],
      "exclude": ["drafts/"],
      "file_extensions": [".md", ".markdown", ".txt", ".docx", ".pdf", ".html", ".htm"]
    },
    "work": {
      "sources": ["~/work/memos"]
    }
  },
  "llm": { "enabled": false }
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `active_profile` | yes | — | Which entry in `profiles` to build and serve. |
| `profiles` | yes | — | Map of profile name → settings. The schema is multi-profile-ready; v0.1 builds only the active one. |
| `profiles.<name>.sources` | yes | — | Directories and/or files to learn from (`~` expanded). |
| `profiles.<name>.exclude` | no | `[]` | Path components to skip during the scan. |
| `profiles.<name>.file_extensions` | no | see table below | Which extensions to analyze. |
| `data_dir` | no | `~/.write-like-me` | Where `profile.json` + `examples.db` are written. |
| `llm.enabled` | no | `false` | Reserved for a future server-side rewrite path; **always `false` in v0.1**. |

### Config discovery priority

The config file is resolved at startup with this priority (first match wins):

| Priority | Method | Example |
|----------|--------|---------|
| 1 (highest) | CLI argument | `write-like-me-mcp /path/to/write-like-me.json` |
| 2 | Environment variable | `WRITE_LIKE_ME_CONFIG=/path/to/write-like-me.json` |
| 3 | Home-dir config file | `~/.write-like-me.json` |

If no config is found, the server exits with a clear error explaining how to
create one — it never silently defaults to a directory.

## Setup

### Claude Code (CLI)

```bash
claude mcp add write-like-me -- write-like-me-mcp /path/to/your/write-like-me.json
```

That's it. The server starts automatically with every `claude` session.

**To change config**, remove and re-add:

```bash
claude mcp remove write-like-me
claude mcp add write-like-me -- write-like-me-mcp /new/path/to/write-like-me.json
```

### Claude Desktop

Add this to your Claude Desktop config file:

| OS | Config path |
|----|-------------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "write-like-me": {
      "command": "write-like-me-mcp",
      "args": ["/path/to/your/write-like-me.json"]
    }
  }
}
```

### Alternative: Environment Variable

Instead of passing the config path as an argument:

```bash
export WRITE_LIKE_ME_CONFIG="/path/to/your/write-like-me.json"
```

## Supported File Types

| Format | Extensions | Notes |
|--------|------------|-------|
| Plain text | `.txt`, `.md`, `.markdown` | UTF-8 with encoding fallback |
| PDF | `.pdf` | Extracted with PyMuPDF |
| Word | `.docx` | Headings + tables preserved |
| HTML | `.html`, `.htm` | Boilerplate (`script`/`style`/`nav`) stripped |
| Rich Text | `.rtf` | **Not supported in v0.1** — the plain reader would leak RTF control codes into the corpus and corrupt the metrics; proper handling is deferred. |

## Available Tools

Once configured, Claude gains access to these five tools:

| Tool | Description |
|------|-------------|
| `get_style_profile` | Return the full learned style profile (sentence rhythm, vocabulary, punctuation, signature phrases, and a ready-to-use style guide). Call this **before** writing prose on the user's behalf. |
| `find_writing_examples` | FTS5/BM25 search for representative real excerpts of the user's writing, for few-shot grounding. Returns basenames only. |
| `apply_style` | Compare a draft to the user's profile and return a concrete rewrite brief — draft-vs-author metric gaps plus the style guide. The calling LLM does the rewrite. |
| `analyze_writing_style` | (Re)scan the corpus and rebuild `profile.json` + `examples.db`. Runs automatically at startup and on corpus changes; call explicitly to force a rebuild. |
| `style_status` | Report profile/index health, readiness, document count, and whether the (reserved) LLM path is enabled. |

## Example Usage

Once set up, ask Claude things like:

- *"Draft a reply to this email so it sounds like me."*
- *"Rewrite this paragraph in my voice — check my style profile first."*
- *"Write a short post about X the way I'd write it; pull a couple of my real
  examples for reference."*

Claude will load your style profile, optionally retrieve real excerpts, and
match your voice — all from context this server computes locally.

## Architecture

```
Claude (Desktop / Code / API)
    │
    ▼  stdio transport (stdout = protocol; stderr = logs)
write-like-me MCP Server
    ├── FastMCP            (5 tool definitions)
    ├── Style Analyzer     (sentence/lexical/punctuation/phrase metrics)
    ├── StyleProfile       (derived-stats model → profile.json)
    ├── Excerpt Index      (SQLite FTS5 + BM25, paragraph excerpts → examples.db)
    ├── Watchdog Watcher   (debounced corpus-change rebuilds)
    └── Text Extractors    (TXT/MD, PDF, DOCX, HTML)
            │
            ▼
    Your local corpus              Your local data dir
    ~/writing, ~/notes, ...   →    ~/.write-like-me/<profile>/
                                     ├── profile.json   (derived stats only)
                                     └── examples.db    (local excerpt index)
```

Nothing in this diagram crosses the machine boundary: there is no network edge.

## Version Boundary: v0.1 → v0.2

v0.1 is intentionally **local-only and LLM-free**:

- `apply_style` returns a *rewrite brief* (metric gaps + style guide); the
  Claude you're already talking to performs the actual rewrite. A **server-side
  LLM `apply_style`** that returns finished rewritten prose is **deferred to a
  future version**.
- The config carries a reserved `llm` block (`enabled: false`) so this can be
  added later without a schema change. In v0.1 it is always disabled, and a test
  guarantees no socket is ever opened.

## Development

```bash
git clone https://github.com/gwicho38/write-like-me-mcp.git
cd write-like-me-mcp
uv sync --extra dev

# Run the local quality gate (lint, type-check, tests)
uv run ruff check .
uv run mypy src
uv run pytest

# Run the server directly against a test config
uv run write-like-me-mcp /path/to/test/write-like-me.json

# Inspect interactively with the MCP Inspector
npx @modelcontextprotocol/inspector write-like-me-mcp /path/to/test/write-like-me.json
```

### Testing notes

- Tests are hermetic: builds and data dirs go to `tmp_path`, and `HOME` is
  monkeypatched so the real home is never touched.
- The OSS-safety suite (`tests/test_oss_safety.py`) and the no-verbatim-corpus
  suite (`tests/test_no_verbatim_corpus.py`) encode the privacy guarantees in
  the table above — they are the project's reason for existing, so keep them
  green.

## License

MIT — see [LICENSE](LICENSE).
