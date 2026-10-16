"""Tests for typed Document Graph identifiers."""

from pathlib import Path

import pytest

from genai_graph.kg.ids import (
    ParsedId,
    is_bare_digest,
    make_id,
    parse_id,
    source_folder_id,
)

DIGEST = "aba538ff22367499"


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["srcfold", "mdfold", "file"])
def test_make_id_round_trips_through_parse_id(kind: str) -> None:
    assert parse_id(make_id(kind, DIGEST)) == ParsedId(kind=kind, digest=DIGEST)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("value", ["alko", "folder_273e65da416b2e72", "file_short", f"other_{DIGEST}", DIGEST])
def test_parse_id_rejects_non_typed_values(value: str) -> None:
    assert parse_id(value) is None


@pytest.mark.unit
def test_is_bare_digest_accepts_sixteen_hex_digits() -> None:
    assert is_bare_digest(DIGEST)


@pytest.mark.unit
@pytest.mark.parametrize("value", ["srcfold_" + DIGEST, "abc", "ABA538FF22367499"])
def test_is_bare_digest_rejects_other_values(value: str) -> None:
    assert not is_bare_digest(value)


@pytest.mark.unit
def test_source_folder_id_is_srcfold_prefixed(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"content")
    assert source_folder_id(tmp_path).startswith("srcfold_")


@pytest.mark.unit
def test_source_folder_id_changes_when_a_file_content_changes(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"v1")
    before = source_folder_id(tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"v2")
    assert source_folder_id(tmp_path) != before


@pytest.mark.unit
def test_source_folder_id_changes_when_a_file_is_renamed(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"same")
    before = source_folder_id(tmp_path)
    (tmp_path / "a.pdf").rename(tmp_path / "b.pdf")
    assert source_folder_id(tmp_path) != before


@pytest.mark.unit
def test_source_folder_id_is_stable_for_unchanged_content(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"same")
    assert source_folder_id(tmp_path) == source_folder_id(tmp_path)


@pytest.mark.unit
def test_source_folder_id_ignores_excluded_top_level_entries(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"source")
    before = source_folder_id(tmp_path, exclude=("markdown",))
    (tmp_path / "markdown").mkdir()
    (tmp_path / "markdown" / "a.md").write_text("converted")
    assert source_folder_id(tmp_path, exclude=("markdown",)) == before
    assert source_folder_id(tmp_path) != before
