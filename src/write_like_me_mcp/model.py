"""StyleProfile model and JSON persistence for write-like-me-mcp.

This module defines the :class:`StyleProfile` dataclass — the schema
describing a learned writing style — together with its human-readable JSON
serialization.

Schema contract
---------------
The fields of :class:`StyleProfile` are the persistence contract. They are
written verbatim to ``profile.json`` under ``<data_root>/<profile>/`` and read
back with typed reconstruction. Fields added after the initial release carry a
default so that a profile written by an older version still loads;
``SCHEMA_VERSION`` records which set was written.

Privacy invariants
------------------
* The profile stores **no verbatim corpus text** — only derived statistics, a
  small set of signature phrases, and a generated style guide.
* The profile stores **no absolute filesystem paths**. ``metadata`` records
  counts and identifiers only; callers MUST pass basenames or paths relative to
  the data root (never absolute paths). This module does not introduce any
  absolute path of its own during serialization.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

#: Version of the on-disk ``profile.json`` schema. Bump on breaking changes to
#: the :class:`StyleProfile` field set so loaders can detect incompatibility.
SCHEMA_VERSION: int = 2


@dataclass
class StyleProfile:
    """A learned writing-style profile.

    Each field is a derived statistic or artifact computed from a user's
    document corpus. The full set of fields is the persistence contract; see
    the module docstring for the privacy invariants that govern ``metadata``.

    Attributes:
        avg_sentence_length: Mean words per sentence across the corpus.
        median_sentence_length: Median words per sentence across the corpus.
        sentence_length_distribution: Mapping of sentence-length buckets to
            their relative frequencies (e.g. ``{"short": 0.2, ...}``).
        lexical_diversity: Type-token ratio (unique words / total words).
        avg_paragraph_length: Mean sentences per paragraph.
        punctuation_per_1k: Mapping of punctuation-mark name to occurrences
            per 1,000 words.
        contraction_rate: Fraction of contractible constructions written as
            contractions (0.0–1.0).
        passive_voice_rate: Fraction of sentences in the passive voice
            (0.0–1.0).
        signature_phrases: Distinctive recurring phrases (no verbatim corpus
            sentences — short, characteristic n-grams only).
        formality_markers: Mapping of formality dimension to score
            (e.g. ``{"formal": 0.4, "informal": 0.6}``).
        style_guide_md: A generated, human-readable style guide in Markdown.
        metadata: Provenance and bookkeeping. Required keys: ``profile_name``,
            ``source_count``, ``doc_count``, ``total_words``, ``generated_at``
            (ISO8601 string), ``schema_version`` (``SCHEMA_VERSION``). Values
            must be basenames/relative paths or scalars — never absolute paths.
        languages: Per-language metric breakdown keyed by ISO 639-1 code. The
            top-level metric fields describe the dominant language; this maps
            every detected language to its own metrics so a multilingual
            corpus is not reported as a blend. Empty for profiles written
            before schema version 2.
    """

    avg_sentence_length: float
    median_sentence_length: float
    sentence_length_distribution: dict
    lexical_diversity: float
    avg_paragraph_length: float
    punctuation_per_1k: dict
    contraction_rate: float
    passive_voice_rate: float
    signature_phrases: list
    formality_markers: dict
    style_guide_md: str
    metadata: dict
    languages: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` representation of this profile.

        The dict is JSON-serializable and contains every field exactly as
        stored (nested dicts and lists are deep-copied by :func:`asdict`).
        """
        return asdict(self)

    def to_json(self) -> str:
        """Serialize this profile to an indented, human-readable JSON string.

        Uses ``indent=2`` for readability and ``ensure_ascii=False`` so
        non-ASCII characters (smart quotes, accented letters, em-dashes) are
        written as themselves rather than escaped.
        """
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        """Write this profile to ``path`` as human-readable ``profile.json``.

        Creates parent directories as needed. The output is indented and
        UTF-8 encoded. This method does not inject any absolute path into the
        serialized content (privacy invariant); it only persists the fields it
        was given.

        Args:
            path: Destination file path (typically
                ``<data_root>/<profile>/profile.json``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleProfile:
        """Reconstruct a :class:`StyleProfile` from a plain ``dict``.

        Only the known schema fields are consumed; any extra keys in ``data``
        are ignored so that forward-compatible additions do not break loading
        of older readers. Fields carrying a default are optional, so a profile
        written by an earlier schema version still loads.

        Args:
            data: Mapping containing the profile's fields.

        Returns:
            A typed :class:`StyleProfile` instance.

        Raises:
            KeyError: If a required schema field is missing from ``data``.
        """
        kwargs = {}
        for f in fields(cls):
            if f.name in data:
                kwargs[f.name] = data[f.name]
            elif f.default is MISSING and f.default_factory is MISSING:
                raise KeyError(f.name)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, payload: str) -> StyleProfile:
        """Reconstruct a :class:`StyleProfile` from a JSON string.

        Args:
            payload: JSON text previously produced by :meth:`to_json`.

        Returns:
            A typed :class:`StyleProfile` instance.
        """
        return cls.from_dict(json.loads(payload))

    @classmethod
    def load(cls, path: str | Path) -> StyleProfile | None:
        """Load a profile from ``path``, or return ``None`` if it is missing.

        A missing file is the normal "not yet built" state and is signalled by
        returning ``None`` rather than raising, so callers can map it to a
        ``not_built`` status. A file that exists but is malformed is a real
        error and is allowed to propagate.

        Args:
            path: Path to a ``profile.json`` file.

        Returns:
            The loaded :class:`StyleProfile`, or ``None`` if ``path`` does not
            exist.
        """
        path = Path(path)
        if not path.exists():
            return None
        return cls.from_json(path.read_text(encoding="utf-8"))
