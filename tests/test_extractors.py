"""Tests for multi-format text extraction (``write_like_me_mcp.extractors``).

These tests prove:
- Round-trip extraction returns expected text for the in-scope formats
  (``.txt``, ``.md``/``.markdown``, ``.docx``, ``.pdf``, ``.html``/``.htm``).
- Malformed/corrupt input returns ``None`` gracefully (never raises).
- ``.rtf`` is excluded from the analyzed extension set (spec Core Req 1:
  RTF is dropped in v0.1 so control codes cannot leak into style metrics).
- Logging is confined to stderr (stdout is the MCP channel).

Binary fixtures (``.docx``, ``.pdf``) are generated SYNTHETICALLY at test time
via python-docx and PyMuPDF. No real author content is used or committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the src/ layout package is importable without an editable install.
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from write_like_me_mcp.extractors import extract_text  # noqa: E402

# Synthetic marker strings — clearly not real author content.
TXT_MARKER = "synthetic plain text fixture line one"
MD_MARKER = "Synthetic Markdown Heading"
DOCX_MARKER = "synthetic docx paragraph content"
PDF_MARKER = "synthetic pdf page content"
HTML_MARKER = "synthetic html body text"


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    """Return a temp directory to hold synthetic, throwaway fixtures."""
    d = tmp_path / "fixtures"
    d.mkdir()
    return d


def _make_txt(directory: Path) -> Path:
    p = directory / "sample.txt"
    p.write_text(f"{TXT_MARKER}\nsecond line\n", encoding="utf-8")
    return p


def _make_md(directory: Path, suffix: str) -> Path:
    p = directory / f"sample{suffix}"
    p.write_text(f"# {MD_MARKER}\n\nBody paragraph here.\n", encoding="utf-8")
    return p


def _make_docx(directory: Path) -> Path:
    from docx import Document

    p = directory / "sample.docx"
    doc = Document()
    doc.add_paragraph(DOCX_MARKER)
    doc.save(str(p))
    return p


def _make_pdf(directory: Path) -> Path:
    import fitz  # PyMuPDF

    p = directory / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), PDF_MARKER)
    doc.save(str(p))
    doc.close()
    return p


def _make_html(directory: Path, suffix: str) -> Path:
    p = directory / f"sample{suffix}"
    p.write_text(
        "<html><head><title>t</title><style>x{}</style></head>"
        f"<body><p>{HTML_MARKER}</p></body></html>",
        encoding="utf-8",
    )
    return p


def _make_malformed_docx(directory: Path) -> Path:
    """A file with a .docx extension that is NOT a valid docx (zip) container."""
    p = directory / "broken.docx"
    p.write_bytes(b"this is not a real docx zip container \x00\x01\x02")
    return p


# --- Round-trip extraction for in-scope formats -----------------------------


def test_extract_txt_round_trip(fixtures_dir: Path) -> None:
    """A .txt file extracts its plain text content."""
    text = extract_text(_make_txt(fixtures_dir))
    assert text is not None
    assert TXT_MARKER in text


@pytest.mark.parametrize("suffix", [".md", ".markdown"])
def test_extract_markdown_round_trip(fixtures_dir: Path, suffix: str) -> None:
    """Both .md and .markdown extract their text content."""
    text = extract_text(_make_md(fixtures_dir, suffix))
    assert text is not None
    assert MD_MARKER in text


def test_extract_docx_round_trip(fixtures_dir: Path) -> None:
    """A synthetic .docx extracts its paragraph text."""
    text = extract_text(_make_docx(fixtures_dir))
    assert text is not None
    assert DOCX_MARKER in text


def test_extract_pdf_round_trip(fixtures_dir: Path) -> None:
    """A synthetic .pdf extracts its page text."""
    text = extract_text(_make_pdf(fixtures_dir))
    assert text is not None
    assert PDF_MARKER in text


@pytest.mark.parametrize("suffix", [".html", ".htm"])
def test_extract_html_round_trip(fixtures_dir: Path, suffix: str) -> None:
    """Both .html and .htm extract visible body text (scripts/styles stripped)."""
    text = extract_text(_make_html(fixtures_dir, suffix))
    assert text is not None
    assert HTML_MARKER in text
    # Style content must be stripped, not leaked into the extracted text.
    assert "x{}" not in text


# --- Graceful failure on malformed input ------------------------------------


def test_malformed_input_returns_none(fixtures_dir: Path) -> None:
    """A corrupt .docx returns None and does not raise."""
    result = extract_text(_make_malformed_docx(fixtures_dir))
    assert result is None


# --- .rtf exclusion (spec Core Req 1) ---------------------------------------


def test_rtf_is_excluded_from_analyzed_set(fixtures_dir: Path) -> None:
    """An .rtf path is NOT routed to any reader: it is excluded (returns None).

    RTF is intentionally dropped in v0.1 so that RTF control codes never reach
    the plain-text reader and corrupt downstream style metrics. We write valid
    plain text into the .rtf file; if RTF were (incorrectly) routed to the
    plain-text reader it would extract successfully. Exclusion means None.
    """
    rtf_path = fixtures_dir / "sample.rtf"
    rtf_path.write_text(r"{\rtf1\ansi plain words here}", encoding="utf-8")
    assert extract_text(rtf_path) is None


# --- Logging is stderr-only -------------------------------------------------


def test_logging_does_not_write_to_stdout(
    fixtures_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Failure-path logging must not emit anything on stdout (the MCP channel)."""
    # Trigger the warning/error log paths: unsupported ext + malformed docx.
    extract_text(fixtures_dir / "nope.rtf")
    extract_text(_make_malformed_docx(fixtures_dir))
    captured = capsys.readouterr()
    assert captured.out == "", f"unexpected stdout output: {captured.out!r}"


# --- Markdown structural noise is stripped (regression) ---------------------


def test_yaml_frontmatter_is_stripped_from_markdown(fixtures_dir: Path) -> None:
    """Frontmatter keys must not reach the analyzer as author prose.

    A vault-style note carries a YAML block whose keys ("type: note",
    "tags: [...]") are metadata, not sentences. Counting them inflates the
    passive-voice rate and lets metadata bigrams win the signature-phrase
    ranking over real writing.
    """
    md = fixtures_dir / "note.md"
    md.write_text(
        "---\n"
        "id: Note\n"
        "type: note\n"
        "tags: [aws, daily-notes, personal]\n"
        "---\n"
        f"{MD_MARKER} body sentence.\n",
        encoding="utf-8",
    )
    text = extract_text(md)
    assert text is not None
    assert MD_MARKER in text
    for key in ("type: note", "tags:", "id: Note"):
        assert key not in text


def test_frontmatter_only_stripped_at_start_of_file(fixtures_dir: Path) -> None:
    """A horizontal rule mid-document is content, not a frontmatter fence."""
    md = fixtures_dir / "rule.md"
    md.write_text(f"{MD_MARKER} opening.\n\n---\n\nclosing sentence.\n", encoding="utf-8")
    text = extract_text(md)
    assert text is not None
    assert "closing sentence." in text


def test_fenced_code_blocks_are_stripped_from_markdown(fixtures_dir: Path) -> None:
    """Code is not prose — fenced blocks must not feed the style metrics."""
    md = fixtures_dir / "code.md"
    md.write_text(
        f"{MD_MARKER} explains the call.\n\n"
        "```python\n"
        "def sneaky_identifier(x):\n"
        "    return x\n"
        "```\n\n"
        "trailing sentence.\n",
        encoding="utf-8",
    )
    text = extract_text(md)
    assert text is not None
    assert MD_MARKER in text
    assert "trailing sentence." in text
    assert "sneaky_identifier" not in text


def test_plain_text_files_keep_their_content_verbatim(fixtures_dir: Path) -> None:
    """.txt is not markdown: no stripping is applied to it."""
    txt = fixtures_dir / "note.txt"
    txt.write_text("---\nnot: frontmatter\n---\nbody\n", encoding="utf-8")
    text = extract_text(txt)
    assert text is not None
    assert "not: frontmatter" in text
