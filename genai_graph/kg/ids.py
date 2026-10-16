"""Typed, human-readable identifiers for Document Graph nodes.

Every identifier is ``<kind>_<digest>`` where ``digest`` is the 16-hex-digit XXH3-64
hash of the node's content. The prefix tells what the id refers to:

- ``srcfold_``: a folder of original source files (or a single source archive)
- ``mdfold_``: a folder of Markdown renderings, e.g. the output of a conversion step
- ``file_``: a source file, identified by its content; it is also the key of its Markdown sections
"""

from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import Path
from typing import Literal, NamedTuple, cast

from genai_tk.utils.hashing import buffer_digest, file_digest

IdKind = Literal["srcfold", "mdfold", "file"]

FOLDER_KINDS: tuple[IdKind, ...] = ("srcfold", "mdfold")
ID_KINDS: tuple[IdKind, ...] = ("srcfold", "mdfold", "file")

_TYPED_ID_RE = re.compile(r"^(srcfold|mdfold|file)_([0-9a-f]{16})$")
_BARE_DIGEST_RE = re.compile(r"^[0-9a-f]{16}$")


class ParsedId(NamedTuple):
    """A typed id split into its kind and digest."""

    kind: IdKind
    digest: str


def make_id(kind: IdKind, digest: str) -> str:
    """Return the typed identifier for a hex digest.

    Args:
        kind: Identifier kind (folder, file, ...).
        digest: 16-hex-digit XXH3-64 digest.

    Returns:
        The typed id, e.g. ``file_aba538ff22367499``.
    """
    return f"{kind}_{digest}"


def parse_id(node_id: str) -> ParsedId | None:
    """Split a typed id into kind and digest, or return None for any other string.

    Args:
        node_id: Candidate identifier.

    Returns:
        A `ParsedId`, or None when *node_id* is not a typed id.
    """
    match = _TYPED_ID_RE.match(node_id)
    if match is None:
        return None
    return ParsedId(kind=cast("IdKind", match.group(1)), digest=match.group(2))


def is_bare_digest(value: str) -> bool:
    """Return True when *value* is a bare 16-hex-digit digest without a kind prefix."""
    return _BARE_DIGEST_RE.match(value) is not None


def file_id(path: Path) -> str:
    """Return the ``file_`` id of a source file, derived from its bytes."""
    return make_id("file", file_digest(path))


def source_folder_id(path: Path, exclude: Collection[str] = ()) -> str:
    """Return the ``srcfold_`` id of a source directory or single source file.

    A directory's id is computed over the sorted relative paths and content digests
    of every file it contains, so any added, removed, renamed or changed source file
    yields a new id. A single file or archive is identified by its name and bytes.

    Args:
        path: Source directory, zip archive or single file.
        exclude: Top-level entry names of a directory to leave out, e.g. the Markdown
            output folder written inside the source folder by a previous run.

    Returns:
        The typed `srcfold_` id.
    """
    if path.is_dir():
        entries = sorted(
            f"{p.relative_to(path).as_posix()}:{file_digest(p)}"
            for p in path.rglob("*")
            if p.is_file() and p.relative_to(path).parts[0] not in exclude
        )
    else:
        entries = [f"{path.name}:{file_digest(path)}"]
    return make_id("srcfold", buffer_digest("\n".join(entries).encode("utf-8")))
