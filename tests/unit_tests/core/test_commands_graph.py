"""Tests for GraphCommands CLI (`cli graph display`, `cli graph query`)."""

from __future__ import annotations

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
