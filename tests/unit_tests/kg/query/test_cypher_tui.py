"""Tests for the Cypher Query Textual TUI (`cli graph query`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, DataTable, OptionList, Static, TextArea

from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.query.cypher_tui import (
    CypherQueryApp,
    introspect_schema,
    load_cypher_examples,
)


@pytest.fixture
def sample_graph_db(temp_db_path: str) -> str:
    """Create a small test database with nodes and relationships."""
    backend = KuzuBackend()
    backend.connect(temp_db_path)
    backend.execute("CREATE NODE TABLE Company(id INT64, name STRING, PRIMARY KEY(id))")
    backend.execute("CREATE NODE TABLE Employee(id INT64, name STRING, role STRING, PRIMARY KEY(id))")
    backend.execute("CREATE REL TABLE WORKS_AT(FROM Employee TO Company)")
    backend.execute("CREATE (c:Company {id: 1, name: 'Atos'})")
    backend.execute("CREATE (e:Employee {id: 10, name: 'Alice', role: 'Engineer'})")
    backend.execute("CREATE (e:Employee {id: 20, name: 'Bob', role: 'Designer'})")
    backend.execute("MATCH (e:Employee), (c:Company) WHERE e.id = 10 AND c.id = 1 CREATE (e)-[:WORKS_AT]->(c)")
    backend.close()
    return temp_db_path


def test_load_cypher_examples() -> None:
    examples = load_cypher_examples()
    assert len(examples) > 5
    for ex in examples:
        assert ex.name
        assert ex.query
        assert ex.category


def test_introspect_schema(sample_graph_db: str) -> None:
    backend = KuzuBackend()
    backend.connect(sample_graph_db, read_only=True)
    nodes, rels = introspect_schema(backend)
    backend.close()

    assert "Company" in nodes
    assert "Employee" in nodes
    assert "WORKS_AT" in rels


@pytest.mark.anyio
async def test_cypher_tui_app_lifecycle(sample_graph_db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Set CWD to tmp_path for export
    monkeypatch.chdir(tmp_path)

    app = CypherQueryApp(db_path=sample_graph_db, read_only=True)
    async with app.run_test() as pilot:
        table_list = app.query_one("#table-list", OptionList)
        example_list = app.query_one("#example-list", OptionList)
        editor = app.query_one("#query-editor", TextArea)
        status = app.query_one("#status-bar", Static)
        table = app.query_one("#results-table", DataTable)

        # 1. Verify table options populated
        assert table_list.option_count > 0

        # 2. Verify example options populated
        assert example_list.option_count > 0

        # 3. Run default query (CALL show_tables() RETURN name, type)
        run_btn = app.query_one("#run-btn", Button)
        await pilot.click(run_btn)
        await pilot.pause()

        assert table.row_count == 3  # Company, Employee, WORKS_AT
        assert "row(s) returned" in str(status.render())

        # 4. Select a table to populate editor with a node query
        table_list.highlighted = 1  # First selectable option (e.g. 🏷️ Company)
        table_list.action_select()
        await pilot.pause()

        assert "MATCH (n:Company) RETURN n LIMIT 25" in editor.text

        # Run the node query
        await pilot.click(run_btn)
        await pilot.pause()
        assert table.row_count == 1

        # 5. Export CSV
        export_btn = app.query_one("#export-btn", Button)
        await pilot.click(export_btn)
        await pilot.pause()

        assert "Exported 1 row(s)" in str(status.render())
        assert (tmp_path / "query_results.csv").exists()

        # 6. Test query error handling
        editor.load_text("MATCH (z:NonExistentTable) RETURN z")
        await pilot.click(run_btn)
        await pilot.pause()

        assert "Query Error" in str(status.render())
        assert table.row_count == 0

        # 7. Test clear button
        clear_btn = app.query_one("#clear-btn", Button)
        await pilot.click(clear_btn)
        await pilot.pause()

        assert editor.text == ""
        assert "cleared" in str(status.render())
