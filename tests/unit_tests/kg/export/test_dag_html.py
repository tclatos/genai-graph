"""Tests for the DAG HTML generator and the refactored force-directed generator.

Builds a tiny in-memory Kuzu graph (Folder -> Document -> Section -> Subsection)
and asserts that:
* ``generate_dag_html`` produces a d3-dag/sugiyama page with the expected nodes
  and links injected as valid JSON.
* ``generate_html`` (force-directed) still works after the ``_graph_model``
  refactor — i.e. it produces the force template (not the DAG one) with the same
  injected data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genai_graph.kg.backend import create_in_memory_backend
from genai_graph.kg.export import generate_dag_html, generate_html


def _build_tiny_tree(backend: object) -> None:
    """Create Folder -> Document -> Section -> Subsection tables and rows."""
    # Node tables.
    backend.execute("CREATE NODE TABLE Folder(id STRING PRIMARY KEY, name STRING)")
    backend.execute("CREATE NODE TABLE Document(id STRING PRIMARY KEY, name STRING, filename STRING)")
    backend.execute("CREATE NODE TABLE Section(id STRING PRIMARY KEY, name STRING, title STRING, level INT64)")
    # Relationship tables (acyclic tree).
    backend.execute("CREATE REL TABLE CONTAINS_DOC(FROM Folder TO Document)")
    backend.execute("CREATE REL TABLE HAS_SECTION(FROM Document TO Section)")
    backend.execute("CREATE REL TABLE HAS_SUBSECTION(FROM Section TO Section)")
    # Nodes.
    backend.execute("CREATE (f:Folder {id: 'f1', name: 'RFQ Folder'})")
    backend.execute("CREATE (d:Document {id: 'd1', name: 'rfq.docx', filename: 'rfq.docx'})")
    backend.execute("CREATE (s1:Section {id: 's1', name: 'Introduction', title: 'Introduction', level: 1})")
    backend.execute("CREATE (s2:Section {id: 's2', name: 'Background', title: 'Background', level: 2})")
    # Relationships.
    backend.execute("MATCH (f:Folder {id: 'f1'}), (d:Document {id: 'd1'}) CREATE (f)-[:CONTAINS_DOC]->(d)")
    backend.execute("MATCH (d:Document {id: 'd1'}), (s:Section {id: 's1'}) CREATE (d)-[:HAS_SECTION]->(s)")
    backend.execute("MATCH (p:Section {id: 's1'}), (c:Section {id: 's2'}) CREATE (p)-[:HAS_SUBSECTION]->(c)")


def _extract_json_var(html: str, var: str, until: str) -> list[dict]:
    """Extract a ``var <name> = <json>;`` assignment from *html* and parse it."""
    start = html.index(f"var {var} = ") + len(f"var {var} = ")
    end = html.index(until, start)
    raw = html[start:end].strip().rstrip(";").strip()
    return json.loads(raw)


@pytest.fixture
def dag_backend() -> object:
    """An in-memory Kuzu backend populated with a tiny document tree."""
    backend = create_in_memory_backend()
    _build_tiny_tree(backend)
    return backend


def test_generate_dag_html_contains_sugiyama_and_data(dag_backend: object, tmp_path: Path) -> None:
    """The DAG page embeds d3-dag/sugiyama and the graph nodes/links as JSON."""
    out = tmp_path / "dag.html"
    html = generate_dag_html(dag_backend, destination_file_path=str(out))

    assert out.exists() and out.read_text(encoding="utf-8") == html
    # DAG-specific markers (from the page JS + the d3-dag bundle).
    assert "d3.sugiyama" in html
    assert "d3.graphStratify" in html
    assert "d3-dag" in html
    # The force template's own simulation call must not be present. The bundled
    # d3 v5 library defines forceSimulation, so check for the template's usage
    # ("d3.forceSimulation(nodes)") rather than the bare library symbol.
    assert "d3.forceSimulation(nodes)" not in html
    # Injected node/edge data is valid JSON with the expected contents.
    nodes = _extract_json_var(html, "nodes", "\n        var links")
    links = _extract_json_var(html, "links", "\n\n        var ORIENTATION")
    names = {n["name"] for n in nodes}
    assert {"RFQ Folder", "rfq.docx", "Introduction", "Background"} <= names
    assert all(n["color"].startswith("#") for n in nodes)
    relations = {lk["relation"] for lk in links}
    assert {"CONTAINS_DOC", "HAS_SECTION", "HAS_SUBSECTION"} <= relations
    # Every link references existing node ids (roots have no incoming edge).
    node_ids = {n["id"] for n in nodes}
    assert all(lk["source"] in node_ids and lk["target"] in node_ids for lk in links)
    # Default orientation is TB (top-down)
    assert 'var ORIENTATION = "TB"' in html

    # Orientation override to LR
    html_lr = generate_dag_html(dag_backend, orientation="LR")
    assert 'var ORIENTATION = "LR"' in html_lr


def test_generate_html_force_regression(dag_backend: object, tmp_path: Path) -> None:
    """generate_html still produces the force-directed page after the refactor."""
    out = tmp_path / "force.html"
    html = generate_html(dag_backend, destination_file_path=str(out))

    assert out.exists()
    assert "d3.forceSimulation(nodes)" in html
    # The DAG library must NOT be loaded by the force view.
    assert "d3.sugiyama" not in html
    assert "d3.graphStratify" not in html
    # Same shared model: the four nodes and three edges are present.
    nodes = _extract_json_var(html, "nodes", "\n        var links")
    links = _extract_json_var(html, "links", "\n\n        // State variables")
    assert len(nodes) == 4
    assert len(links) == 3
