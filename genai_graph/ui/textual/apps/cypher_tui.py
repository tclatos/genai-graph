"""Textual TUI for testing and inspecting Cypher queries on graph databases (``cli graph query``).

Left panel lists schema tables (nodes and relationships) and curated generic
example Cypher queries. Selecting a table or example populates the query editor.
The right panel provides an interactive SQL/Cypher editor and a scrollable DataTable
displaying query results, execution times, and row counts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Label, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from genai_graph.kg.backend import KgBackend, KuzuBackend


class CypherExample(BaseModel):
    """Predefined example Cypher query for the TUI."""

    name: str
    description: str
    query: str
    category: str = "General"


DEFAULT_CYPHER_EXAMPLES: list[CypherExample] = [
    # Universal Discovery
    CypherExample(
        name="Show All Tables",
        description="List all node and relationship tables in the database",
        query="CALL show_tables() RETURN name, type",
        category="Discovery",
    ),
    CypherExample(
        name="Count Nodes by Label",
        description="Count total number of nodes grouped by label/table",
        query="MATCH (n) RETURN label(n) AS node_type, count(n) AS count ORDER BY count DESC",
        category="Discovery",
    ),
    CypherExample(
        name="Count Relationships by Type",
        description="Count total number of relationships grouped by type",
        query="MATCH ()-[r]->() RETURN label(r) AS rel_type, count(r) AS count ORDER BY count DESC",
        category="Discovery",
    ),
    # General Graph Exploration
    CypherExample(
        name="Sample Any Nodes",
        description="Return 25 sample nodes with their properties",
        query="MATCH (n) RETURN n LIMIT 25",
        category="General",
    ),
    CypherExample(
        name="Sample Nodes & Relationships",
        description="Return 25 connected node pairs and their connecting relationship",
        query="MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25",
        category="General",
    ),
    CypherExample(
        name="Top Connected Nodes",
        description="Find nodes with the most outgoing relationships",
        query=(
            "MATCH (n)-[r]->() RETURN label(n) AS node_type, count(r) AS outgoing_degree "
            "ORDER BY outgoing_degree DESC LIMIT 10"
        ),
        category="General",
    ),
    CypherExample(
        name="Sample Node Properties",
        description="Inspect full property dictionaries of sample nodes",
        query="MATCH (n) RETURN label(n) AS node_type, properties(n) AS props LIMIT 10",
        category="General",
    ),
    CypherExample(
        name="Paths Between Nodes",
        description="Traverse 1-to-2 hop relationship paths across the graph",
        query="MATCH p = (a)-[*1..2]->(b) RETURN p LIMIT 10",
        category="General",
    ),
    # Document Graph Queries
    CypherExample(
        name="List Ingested Documents",
        description="List documents with filename, file size, and section count",
        query="MATCH (d:Document) RETURN d.filename, d.file_size, d.section_count ORDER BY d.filename LIMIT 20",
        category="Document Graph",
    ),
    CypherExample(
        name="Folder Hierarchy",
        description="Show folders and their contained documents",
        query="MATCH (f:Folder)-[:CONTAINS]->(d:Document) RETURN f.name AS folder, d.filename AS document LIMIT 20",
        category="Document Graph",
    ),
    CypherExample(
        name="Document Sections",
        description="List sections belonging to documents with titles and levels",
        query=(
            "MATCH (d:Document)-[r:HAS_SECTION]->(s:MarkdownSection) "
            "RETURN d.filename AS doc, s.title AS title, s.level AS level, s.line_start AS line "
            "LIMIT 25"
        ),
        category="Document Graph",
    ),
    CypherExample(
        name="Section Tree & Subsections",
        description="Show parent sections and their child subsections",
        query=(
            "MATCH (s:MarkdownSection)-[:HAS_SUBSECTION]->(sub:MarkdownSection) "
            "RETURN s.title AS parent_section, sub.title AS subsection "
            "LIMIT 25"
        ),
        category="Document Graph",
    ),
    CypherExample(
        name="Search Sections by Keyword",
        description="Find sections whose title or text contains a target keyword",
        query=(
            "MATCH (s:MarkdownSection) "
            "WHERE s.title CONTAINS 'SLA' OR s.text CONTAINS 'SLA' "
            "RETURN s.title AS title, s.line_start AS line, s.description AS description "
            "LIMIT 10"
        ),
        category="Document Graph",
    ),
    CypherExample(
        name="Document Images",
        description="List extracted images with names, captions, and source documents",
        query="MATCH (d:Document)-[:HAS_IMAGE]->(i:Image) RETURN d.filename AS doc, i.name AS image, i.caption AS caption LIMIT 20",
        category="Document Graph",
    ),
]


def load_cypher_examples() -> list[CypherExample]:
    """Load default generic examples plus any project-specific queries from config."""
    examples = list(DEFAULT_CYPHER_EXAMPLES)
    for candidate in ("config/cypher_examples.yaml", "cypher_examples.yaml"):
        p = Path(candidate)
        if p.exists():
            try:
                import yaml

                data = yaml.safe_load(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "queries" in data:
                    for item in data["queries"]:
                        examples.append(
                            CypherExample(
                                name=item.get("name", "Custom Query"),
                                description=item.get("description", ""),
                                query=item.get("cypher", ""),
                                category="Custom",
                            )
                        )
            except Exception:
                pass
            break
    return examples


def introspect_schema(backend: KgBackend) -> tuple[list[str], list[str]]:
    """Introspect all node tables and relationship tables in the database.

    Args:
        backend: Connected KgBackend instance.

    Returns:
        Tuple of (node_table_names, relationship_table_names).
    """
    try:
        df = backend.execute_get_as_df("CALL show_tables() RETURN name, type", union=False)
        nodes = sorted(df[df["type"] == "NODE"]["name"].tolist())
        rels = sorted(df[df["type"] == "REL"]["name"].tolist())
        return nodes, rels
    except Exception:
        return [], []


def _detect_editor_language(lang: str = "sql") -> str | None:
    """Return language for TextArea if supported by tree-sitter, otherwise None for plain text."""
    try:
        from textual._tree_sitter import TREE_SITTER, get_language

        if TREE_SITTER and get_language(lang) is not None:
            return lang
    except Exception:
        pass
    return None


def clean_internal_value(val: Any) -> Any:
    """Recursively strip internal Kuzu metadata (_ID, _SRC, _DST) from dicts/lists."""
    if isinstance(val, dict):
        return {k: clean_internal_value(v) for k, v in val.items() if str(k).upper() not in ("_ID", "_SRC", "_DST")}
    if isinstance(val, list):
        return [clean_internal_value(v) for v in val]
    return val


def clean_node_or_rel_dict(d: dict[str, Any], include_label: bool = False) -> dict[str, Any]:
    """Clean a node or relationship dictionary by removing internal keys."""
    cleaned: dict[str, Any] = {}
    for k, v in d.items():
        k_upper = str(k).upper()
        if k_upper in ("_ID", "_SRC", "_DST"):
            continue
        if k_upper == "_LABEL":
            if include_label:
                cleaned["label"] = v
            continue
        cleaned[k] = clean_internal_value(v)
    return cleaned


def is_node_dict(v: Any) -> bool:
    """Return True if value appears to be a Kuzu/Ladybug node dictionary."""
    if not isinstance(v, dict):
        return False
    keys_upper = {str(k).upper() for k in v}
    return ("_ID" in keys_upper or "_LABEL" in keys_upper) and not ("_SRC" in keys_upper or "_DST" in keys_upper)


def is_rel_dict(v: Any) -> bool:
    """Return True if value appears to be a Kuzu/Ladybug relationship dictionary."""
    if not isinstance(v, dict):
        return False
    keys_upper = {str(k).upper() for k in v}
    return "_SRC" in keys_upper or "_DST" in keys_upper


def format_node_display(node: Any) -> tuple[str, str]:
    """Extract (label, summary) from a node dict for concise relationship display.

    Returns:
        (label, identifier): e.g. ("Employee", "Alice (id: 10)") or ("Opportunity", "Project Titan")
    """
    if not isinstance(node, dict):
        return "", str(node) if node is not None else ""

    label = str(node.get("_LABEL") or node.get("_label") or "")
    clean = {k: v for k, v in node.items() if not str(k).startswith("_") and v is not None and v != ""}

    name_val = None
    for k in ("name", "title", "caption", "doc_name", "filename"):
        if k in clean:
            name_val = str(clean[k])
            break

    id_val = None
    for k in ("opportunity_id", "requirement_id", "section_id", "table_id", "folder_id", "id", "code", "key"):
        if k in clean:
            id_val = str(clean[k])
            break

    if name_val and id_val and name_val != id_val:
        summary = f"{name_val} ({id_val})"
    elif name_val:
        summary = name_val
    elif id_val:
        summary = id_val
    elif clean:
        parts = [f"{k}={v}" for k, v in list(clean.items())[:2]]
        summary = ", ".join(parts)
    else:
        summary = label or "(empty)"

    return label, summary


def _first_non_null(series: pd.Series) -> Any:
    """Return the first non-null, non-NA value in a pandas Series."""
    for v in series:
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return v
    return None


def format_graph_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format and unpack a graph DataFrame for display in tables.

    - If the DataFrame contains a relationship triple (node, rel, node), formats
      into clean columns: source_type, source, relationship, [rel_props...], target_type, target.
    - If the DataFrame contains a relationship pair (node, node), formats
      into clean columns: source_type, source, target_type, target.
    - If the DataFrame contains a single column with node/rel dicts (e.g. ``RETURN n``),
      unpacks properties into individual table columns and strips internal ``_ID``.
    - If columns share a common alias prefix (e.g. ``n.id``, ``n.name`` from ``RETURN n.*``),
      strips the prefix for cleaner headers.
    - Removes internal system columns (``_ID``, ``_SRC``, ``_DST``).
    - Recursively strips internal IDs from any dict or list cells.
    """
    if df.empty:
        return df

    # Case 1: Relationship triple (node, rel, node) e.g. RETURN a, r, b
    if len(df.columns) == 3:
        v0 = _first_non_null(df.iloc[:, 0])
        v1 = _first_non_null(df.iloc[:, 1])
        v2 = _first_non_null(df.iloc[:, 2])
        if is_node_dict(v0) and is_rel_dict(v1) and is_node_dict(v2):
            records: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                src_label, src_summary = format_node_display(row.iloc[0])
                rel_dict = row.iloc[1] if isinstance(row.iloc[1], dict) else {}
                rel_type = str(rel_dict.get("_LABEL") or rel_dict.get("_label") or "")
                rel_props = clean_node_or_rel_dict(rel_dict, include_label=False)
                tgt_label, tgt_summary = format_node_display(row.iloc[2])

                entry: dict[str, Any] = {
                    "source_type": src_label,
                    "source": src_summary,
                    "relationship": rel_type,
                    **rel_props,
                    "target_type": tgt_label,
                    "target": tgt_summary,
                }
                records.append(entry)
            return pd.DataFrame(records)

    # Case 2: Relationship pair (node, node) e.g. RETURN a, b
    if len(df.columns) == 2:
        v0 = _first_non_null(df.iloc[:, 0])
        v1 = _first_non_null(df.iloc[:, 1])
        if is_node_dict(v0) and is_node_dict(v1):
            records = []
            for _, row in df.iterrows():
                src_label, src_summary = format_node_display(row.iloc[0])
                tgt_label, tgt_summary = format_node_display(row.iloc[1])
                records.append(
                    {
                        "source_type": src_label,
                        "source": src_summary,
                        "target_type": tgt_label,
                        "target": tgt_summary,
                    }
                )
            return pd.DataFrame(records)

    # Case 3: Single column containing node or rel dicts (e.g. RETURN n, RETURN r)
    if len(df.columns) == 1:
        col = df.columns[0]
        has_dicts = any(
            isinstance(v, dict) for v in df[col] if v is not None and not (isinstance(v, float) and pd.isna(v))
        )
        if has_dicts:
            labels: set[str] = set()
            for val in df[col]:
                if isinstance(val, dict):
                    lbl = val.get("_LABEL") or val.get("_label")
                    if lbl:
                        labels.add(str(lbl))
            include_label = len(labels) > 1

            records = []
            for val in df[col]:
                if isinstance(val, dict):
                    records.append(clean_node_or_rel_dict(val, include_label=include_label))
                else:
                    records.append({str(col): clean_internal_value(val)})
            return pd.DataFrame(records)

    # Case 4: Column names with single alias prefix (e.g. RETURN n.* -> n.id, n.name)
    if all("." in str(c) for c in df.columns):
        prefixes = {str(c).split(".", 1)[0] for c in df.columns}
        if len(prefixes) == 1:
            df = df.rename(columns=lambda c: str(c).split(".", 1)[1])

    # Case 5: Drop top-level internal system columns (_ID, _SRC, _DST)
    clean_cols = [c for c in df.columns if str(c).upper() not in ("_ID", "_SRC", "_DST")]
    if clean_cols and len(clean_cols) < len(df.columns):
        df = df[clean_cols]

    # Clean any dict/list cells in remaining columns
    out_df = df.copy()
    for col in out_df.columns:
        if out_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            out_df[col] = out_df[col].apply(clean_internal_value)

    return out_df


class CypherQueryApp(App[None]):
    """Interactive Textual application for querying graph databases with Cypher."""

    CSS = """
    Screen {
        background: $surface;
    }

    #sidebar {
        width: 38;
        height: 100%;
        border-right: solid $accent;
        padding: 0 1;
        background: $panel;
    }

    .sidebar-title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
        margin-bottom: 0;
        padding-left: 1;
    }

    #table-list {
        height: 11;
        border: round $primary;
        margin-bottom: 0;
    }

    #example-list {
        height: 1fr;
        border: round $primary;
    }

    #main-panel {
        width: 1fr;
        height: 100%;
        padding: 0 1;
    }

    .panel-title {
        text-style: bold;
        color: $text;
        margin-top: 1;
        margin-bottom: 0;
    }

    #editor-container {
        height: 11;
        border: round $accent;
        margin-bottom: 0;
        padding: 0;
    }

    #query-editor {
        height: 1fr;
        border: none;
    }

    #button-bar {
        height: 3;
        align-vertical: middle;
        margin: 1 0 0 0;
    }

    #button-bar Button {
        margin-right: 1;
    }

    #status-bar {
        height: 2;
        padding: 0 1;
        background: $panel;
        color: $text;
        margin: 0 0 1 0;
    }

    #results-container {
        height: 1fr;
        border: round $primary;
        padding: 0;
    }

    #results-table {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("f5", "run_query", "Run Query"),
        ("ctrl+r", "run_query", "Run Query"),
        ("ctrl+j", "run_query", "Run Query"),
        ("ctrl+l", "clear_editor", "Clear"),
        ("ctrl+s", "export_csv", "Export CSV"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        db_path: str,
        read_only: bool = True,
        initial_query: str | None = None,
    ) -> None:
        """Initialize the Cypher Query TUI application.

        Args:
            db_path: Path to the Ladybug / Kùzu database directory.
            read_only: Whether to connect in read-only mode.
            initial_query: Optional query string to pre-load into the editor.
        """
        super().__init__()
        self.db_path = str(Path(db_path).resolve())
        self.read_only = read_only
        self.initial_query = initial_query
        self.backend = KuzuBackend()
        self.backend.connect(self.db_path, read_only=read_only)
        self.examples: list[CypherExample] = load_cypher_examples()
        self._last_df: pd.DataFrame | None = None

    def compose(self) -> ComposeResult:
        """Compose application layout."""
        mode_str = "READ-ONLY" if self.read_only else "READ-WRITE"
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("📊 Schema Tables", classes="sidebar-title")
                yield OptionList(id="table-list")
                yield Label("💡 Example Queries", classes="sidebar-title")
                yield OptionList(id="example-list")
            with Vertical(id="main-panel"):
                yield Label("Cypher Query Editor (F5 to run, Ctrl+L to clear)", classes="panel-title")
                with Vertical(id="editor-container"):
                    default_q = self.initial_query or "CALL show_tables() RETURN name, type"
                    yield TextArea(
                        default_q,
                        id="query-editor",
                        language=_detect_editor_language("sql"),
                        theme="vscode_dark",
                    )
                with Horizontal(id="button-bar"):
                    yield Button("▶ Run Query (F5)", variant="primary", id="run-btn")
                    yield Button("Clear (Ctrl+L)", variant="default", id="clear-btn")
                    yield Button("💾 Export CSV", variant="default", id="export-btn")
                yield Static(
                    f"Connected: {Path(self.db_path).name} [{mode_str}] — Press F5 or click Run Query",
                    id="status-bar",
                )
                with Vertical(id="results-container"):
                    yield DataTable(id="results-table", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        """Populate schema tables and examples on mount."""
        # Introspect schema tables
        table_list = self.query_one("#table-list", OptionList)
        nodes, rels = introspect_schema(self.backend)

        if nodes or rels:
            if nodes:
                table_list.add_option(Option("── Node Tables ──", disabled=True))
                for n in nodes:
                    table_list.add_option(Option(f"🏷️ {n}", id=f"tbl:{n}"))
            if rels:
                table_list.add_option(Option("── Relationships ──", disabled=True))
                for r in rels:
                    table_list.add_option(Option(f"🔗 {r}", id=f"rel:{r}"))
        else:
            table_list.add_option(Option("(No tables detected yet)", disabled=True))

        # Populate example queries categorized
        example_list = self.query_one("#example-list", OptionList)
        current_cat: str | None = None
        for idx, ex in enumerate(self.examples):
            if ex.category != current_cat:
                current_cat = ex.category
                example_list.add_option(Option(f"── {current_cat} ──", disabled=True))
            example_list.add_option(Option(f"• {ex.name}", id=f"ex:{idx}"))

        # If initial query passed, execute it automatically
        if self.initial_query:
            self.action_run_query()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle selection of a schema table or example query."""
        opt_id = event.option_id
        if not opt_id:
            return

        editor = self.query_one("#query-editor", TextArea)

        if opt_id.startswith("tbl:"):
            table_name = opt_id.split(":", 1)[1]
            query = f"MATCH (n:{table_name}) RETURN n LIMIT 25"
            editor.load_text(query)
            self.action_run_query()
        elif opt_id.startswith("rel:"):
            rel_name = opt_id.split(":", 1)[1]
            query = f"MATCH (a)-[r:{rel_name}]->(b) RETURN a, r, b LIMIT 25"
            editor.load_text(query)
            self.action_run_query()
        elif opt_id.startswith("ex:"):
            idx_str = opt_id.split(":", 1)[1]
            try:
                idx = int(idx_str)
                ex = self.examples[idx]
                editor.load_text(ex.query)
                self.action_run_query()
            except (IndexError, ValueError):
                pass

    def action_run_query(self) -> None:
        """Execute the query currently in the query editor."""
        editor = self.query_one("#query-editor", TextArea)
        query = editor.text.strip()
        if not query:
            return

        status = self.query_one("#status-bar", Static)
        table = self.query_one("#results-table", DataTable)
        start_time = time.perf_counter()

        try:
            df = self.backend.execute_get_as_df(query, union=True)
            duration_ms = (time.perf_counter() - start_time) * 1000
            display_df = format_graph_df(df)
            self._last_df = display_df
            table.clear(columns=True)

            if display_df.empty:
                status.update(f"[yellow]Query executed in {duration_ms:.1f}ms — No rows returned[/yellow]")
                return

            for idx, col in enumerate(display_df.columns):
                table.add_column(str(col), key=f"col_{idx}_{col}")

            max_rows = 500
            for i, (_, row) in enumerate(display_df.iterrows()):
                if i >= max_rows:
                    break
                formatted_row: list[str] = []
                for val in row:
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        formatted_row.append("")
                    elif isinstance(val, (dict, list)):
                        s = json.dumps(val, default=str)
                        if len(s) > 80:
                            s = s[:77] + "..."
                        formatted_row.append(s)
                    else:
                        s = str(val).replace("\r\n", " ").replace("\n", " ")
                        if len(s) > 100:
                            s = s[:97] + "..."
                        formatted_row.append(s)
                table.add_row(*formatted_row, label=str(i + 1))

            suffix = f" (showing first {max_rows})" if len(display_df) > max_rows else ""
            status.update(
                f"[bold green]✓ {len(display_df)} row(s) returned ({len(display_df.columns)} columns) in {duration_ms:.1f}ms{suffix}[/bold green]"
            )
        except Exception as exc:
            table.clear(columns=True)
            status.update(f"[bold red]Query Error:[/bold red] {exc}")

    def action_clear_editor(self) -> None:
        """Clear the query editor."""
        self.query_one("#query-editor", TextArea).load_text("")
        self.query_one("#status-bar", Static).update("Editor cleared — type a Cypher query and press F5")

    def action_export_csv(self) -> None:
        """Export the latest query results to a CSV file."""
        status = self.query_one("#status-bar", Static)
        if self._last_df is None or self._last_df.empty:
            status.update("[yellow]No query results available to export[/yellow]")
            return

        out_path = Path("query_results.csv").resolve()
        try:
            self._last_df.to_csv(out_path, index=False)
            status.update(f"[bold green]✓ Exported {len(self._last_df)} row(s) to {out_path}[/bold green]")
        except Exception as exc:
            status.update(f"[bold red]Export Error:[/bold red] {exc}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button click events."""
        btn_id = event.button.id
        if btn_id == "run-btn":
            self.action_run_query()
        elif btn_id == "clear-btn":
            self.action_clear_editor()
        elif btn_id == "export-btn":
            self.action_export_csv()


def run_cypher_tui(
    db_path: str,
    initial_query: str | None = None,
    read_only: bool = True,
) -> None:
    """Launch the interactive Cypher Query Textual TUI.

    Args:
        db_path: Path to the Ladybug / Kùzu database directory.
        initial_query: Optional Cypher query to pre-fill in the editor.
        read_only: Whether to open database in read-only mode (default: True).
    """
    app = CypherQueryApp(db_path=db_path, read_only=read_only, initial_query=initial_query)
    app.run()
