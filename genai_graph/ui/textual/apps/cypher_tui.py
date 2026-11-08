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
                    yield TextArea(default_q, id="query-editor", language="sql", theme="vscode_dark")
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
        status = self.query_one("#status-bar", Static)

        if opt_id.startswith("tbl:"):
            table_name = opt_id.split(":", 1)[1]
            query = f"MATCH (n:{table_name}) RETURN n LIMIT 25"
            editor.load_text(query)
            status.update(
                f"Selected table [bold cyan]{table_name}[/bold cyan] — Press [bold green]F5[/bold green] to execute"
            )
        elif opt_id.startswith("rel:"):
            rel_name = opt_id.split(":", 1)[1]
            query = f"MATCH (a)-[r:{rel_name}]->(b) RETURN a, r, b LIMIT 25"
            editor.load_text(query)
            status.update(
                f"Selected relationship [bold cyan]{rel_name}[/bold cyan] — Press [bold green]F5[/bold green] to execute"
            )
        elif opt_id.startswith("ex:"):
            idx_str = opt_id.split(":", 1)[1]
            try:
                idx = int(idx_str)
                ex = self.examples[idx]
                editor.load_text(ex.query)
                status.update(
                    f"💡 [bold]{ex.name}:[/bold] {ex.description} — Press [bold green]F5[/bold green] to execute"
                )
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
            self._last_df = df
            table.clear(columns=True)

            if df.empty:
                status.update(f"[yellow]Query executed in {duration_ms:.1f}ms — No rows returned[/yellow]")
                return

            for col in df.columns:
                table.add_column(str(col), key=str(col))

            max_rows = 500
            for i, (_, row) in enumerate(df.iterrows()):
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
                        s = str(val)
                        if len(s) > 100:
                            s = s[:97] + "..."
                        formatted_row.append(s)
                table.add_row(*formatted_row)

            suffix = f" (showing first {max_rows})" if len(df) > max_rows else ""
            status.update(f"[bold green]✓ {len(df)} row(s) returned in {duration_ms:.1f}ms{suffix}[/bold green]")
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
