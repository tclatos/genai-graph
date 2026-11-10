"""Tests for GraphCommands CLI (`cli graph display`, `cli graph schema`, `cli graph query`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from genai_graph.core.commands_graph import GraphCommands, resolve_graph_db_path
from genai_graph.kg.backend import KuzuBackend


def _cli_app() -> typer.Typer:
    app = typer.Typer()
    GraphCommands().register(app)
    return app


@pytest.fixture
def sample_graph_db(temp_db_path: str) -> str:
    backend = KuzuBackend()
    backend.connect(temp_db_path)
    backend.execute("CREATE NODE TABLE Item(id INT64, title STRING, PRIMARY KEY(id))")
    backend.execute("CREATE (i:Item {id: 1, title: 'Test Item'})")
    backend.close()
    return temp_db_path


def test_graph_commands_description() -> None:
    cmd = GraphCommands()
    name, desc = cmd.get_description()
    assert name == "graph"
    assert "Graph database" in desc


def test_resolve_graph_db_path_explicit() -> None:
    path = resolve_graph_db_path("/custom/path/db.db")
    assert path == "/custom/path/db.db"


def test_cli_graph_display(sample_graph_db: str, tmp_path: Path) -> None:
    runner = CliRunner()
    app = _cli_app()
    out_html = tmp_path / "viz.html"

    result = runner.invoke(
        app,
        [
            "graph",
            "display",
            "--db",
            sample_graph_db,
            "--output",
            str(out_html),
            "--no-browser",
        ],
    )
    assert result.exit_code == 0
    assert out_html.exists()
    assert len(out_html.read_text(encoding="utf-8")) > 1000
    assert "HTML graph visualization created" in result.output


def test_cli_graph_query_batch(sample_graph_db: str) -> None:
    runner = CliRunner()
    app = _cli_app()

    result = runner.invoke(
        app,
        [
            "graph",
            "query",
            "--db",
            sample_graph_db,
            "-q",
            "MATCH (i:Item) RETURN i.title",
            "--batch",
        ],
    )
    assert result.exit_code == 0
    assert "Test Item" in result.output
    assert "1 rows" in result.output


def test_cli_graph_query_batch_missing_query(sample_graph_db: str) -> None:
    runner = CliRunner()
    app = _cli_app()

    result = runner.invoke(
        app,
        [
            "graph",
            "query",
            "--db",
            sample_graph_db,
            "--batch",
        ],
    )
    assert result.exit_code == 1
    assert "--query is required" in result.output


def test_cli_graph_missing_db() -> None:
    runner = CliRunner()
    app = _cli_app()

    result = runner.invoke(
        app,
        [
            "graph",
            "display",
            "--db",
            "/nonexistent/db_path_12345.db",
            "--no-browser",
        ],
    )
    assert result.exit_code == 1
    assert "Graph database not found" in result.output


def test_cli_graph_schema(sample_graph_db: str, tmp_path: Path) -> None:
    runner = CliRunner()
    app = _cli_app()
    out_html = tmp_path / "schema.html"

    result = runner.invoke(
        app,
        [
            "graph",
            "schema",
            "--db",
            sample_graph_db,
            "--output",
            str(out_html),
            "--no-browser",
        ],
    )
    assert result.exit_code == 0
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")
    assert len(content) > 1000
    assert "Item" in content
    assert "HTML schema visualization created" in result.output


def test_cli_graph_schema_with_relationships(temp_db_path: str, tmp_path: Path) -> None:
    backend = KuzuBackend()
    backend.connect(temp_db_path)
    backend.execute("CREATE NODE TABLE Document(id INT64, title STRING, PRIMARY KEY(id))")
    backend.execute("CREATE NODE TABLE Section(id INT64, heading STRING, PRIMARY KEY(id))")
    backend.execute("CREATE REL TABLE HAS_SECTION(FROM Document TO Section, position INT64)")
    backend.close()

    runner = CliRunner()
    app = _cli_app()
    out_html = tmp_path / "doc_schema.html"

    result = runner.invoke(
        app,
        [
            "graph",
            "schema",
            "--db",
            temp_db_path,
            "--output",
            str(out_html),
            "--no-browser",
        ],
    )
    assert result.exit_code == 0
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")
    assert "Document" in content
    assert "Section" in content
    assert "HAS_SECTION" in content
    assert "HTML schema visualization created" in result.output


def test_cli_graph_schema_from_json(tmp_path: Path) -> None:
    schema_json = tmp_path / "custom_schema.json"
    schema_data = {
        "meta": {"format": "genai_graph.resolved_schema", "format_version": 1, "graphs": ["custom"]},
        "nodes": [
            {
                "id": "Entity",
                "label": "Entity",
                "description": "An entity",
                "primary_key": "id",
                "name_from": "id",
                "index_fields": ["id"],
                "fields": [
                    {
                        "name": "id",
                        "type_human": "STRING",
                        "type_kuzu": "STRING",
                        "description": "",
                        "indexed": True,
                        "embedded": False,
                    }
                ],
            }
        ],
        "links": [],
        "vector_indexes": [],
    }
    schema_json.write_text(json.dumps(schema_data), encoding="utf-8")

    runner = CliRunner()
    app = _cli_app()
    out_html = tmp_path / "json_schema.html"

    result = runner.invoke(
        app,
        [
            "graph",
            "schema",
            "--db",
            str(schema_json),
            "--output",
            str(out_html),
            "--no-browser",
        ],
    )
    assert result.exit_code == 0
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")
    assert "Entity" in content
    assert "HTML schema visualization created" in result.output


def test_cli_graph_schema_missing_db() -> None:
    runner = CliRunner()
    app = _cli_app()

    result = runner.invoke(
        app,
        [
            "graph",
            "schema",
            "--db",
            "/nonexistent/db_path_schema_12345.db",
            "--no-browser",
        ],
    )
    assert result.exit_code == 1
    assert "Graph database not found" in result.output
