"""Textual TUI for browsing a Document Graph (``cli docgraph tui``).

Left panel is a `Tree` of folders → documents → sections, built lazily (on
expand) from the Ladybug graph via `genai_graph.kg.query.document_graph_tools`.
The right panel shows metadata for the selected node and renders Markdown
content for documents and sections.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, LoadingIndicator, Markdown, Static, Tree
from textual.widgets.tree import TreeNode

from genai_graph.kg.backend import KgBackend, KuzuBackend
from genai_graph.kg.query.document_graph_tools import (
    get_document_toc,
    list_documents,
    reconstruct_document,
    reconstruct_section,
)

# Written by genai_tk.workflow.markdownize.markdownize_flow as the first line of
# every converted Markdown file — records the original document it came from.
_ORIGIN_COMMENT_RE = re.compile(r"<!--\s*source:\s*(.+?)\s*-->")


def _read_origin_path(md_path: str | None) -> str | None:
    """Return the original source document path recorded in a converted Markdown file, if any."""
    if not md_path:
        return None
    try:
        with open(md_path, encoding="utf-8") as f:
            first_line = f.readline()
    except OSError:
        return None
    match = _ORIGIN_COMMENT_RE.search(first_line)
    return match.group(1) if match else None


class NodeData(BaseModel):
    """Payload attached to each `Tree` node identifying what it represents."""

    kind: Literal["root", "folder", "document", "section"]
    path: str | None = None
    markdown_hash: str | None = None
    section_id: str | None = None
    filename: str | None = None
    title: str | None = None
    level: int | None = None
    line_start: int | None = None
    count: int | None = None
    loaded: bool = False


def _folder_of(row: dict[str, Any]) -> str:
    """Parent directory of a document row's path, ``"."`` for the folder root."""
    path = row.get("path") or row["filename"]
    return str(PurePosixPath(path).parent)


def _dedupe_documents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one Document per source path — the richest (most sections).

    Re-markdownizing a source can leave several Document versions (different
    content hashes) for the same path in the graph; showing them all clutters the
    tree. Keep the version with the most sections (ties broken by first seen).
    """
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("path") or row["filename"]
        current = best.get(key)
        if current is None or row["section_count"] > current["section_count"]:
            best[key] = row
    return sorted(best.values(), key=lambda r: r.get("path") or r["filename"])


class DocumentGraphApp(App[None]):
    """Browse a Document Graph: folders → documents → sections."""

    CSS = """
    Horizontal {
        height: 1fr;
    }
    #tree-panel {
        width: 38%;
        border-right: solid $accent;
    }
    #info-panel {
        width: 62%;
        padding: 1 2;
    }
    #meta {
        height: auto;
        border-bottom: solid $accent;
        padding-bottom: 1;
        margin-bottom: 1;
    }
    #loading {
        height: 3;
        display: none;
        margin: 1 0;
    }
    #loading.active {
        display: block;
    }
    #content {
        height: auto;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_tree", "Refresh"),
        ("m", "open_markdown", "Open .md"),
        ("o", "open_source", "Open source"),
    ]

    def __init__(self, db_path: str) -> None:
        super().__init__()
        self.db_path = db_path
        self.backend: KgBackend = KuzuBackend()
        self.backend.connect(db_path)
        self._doc_rows: list[dict[str, Any]] = []
        self._active_node_key: tuple[str, str | None] | None = None
        self._current_md_path: str | None = None  # Converted Markdown file, for "m"
        self._current_origin_path: str | None = None  # Original source document, for "o"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield Tree("Folder", id="tree-panel")
            with VerticalScroll(id="info-panel"):
                yield Static("Select a node to see details.", id="meta")
                yield LoadingIndicator(id="loading")
                yield Markdown("", id="content")
        yield Footer()

    def on_mount(self) -> None:
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        self._active_node_key = None
        tree = self.query_one(Tree)
        tree.clear()
        tree.root.data = NodeData(kind="root", loaded=True)
        self._doc_rows = _dedupe_documents(list_documents(self.backend))
        folders: dict[str, int] = {}
        for row in self._doc_rows:
            folders[_folder_of(row)] = folders.get(_folder_of(row), 0) + 1
        for folder, count in sorted(folders.items()):
            label = folder if folder != "." else "(root)"
            node = tree.root.add(label, data=NodeData(kind="folder", path=folder, count=count))
            node.allow_expand = True
        tree.root.expand()

    def _load_documents(self, node: TreeNode, folder: str) -> None:
        for row in self._doc_rows:
            if _folder_of(row) != folder:
                continue
            leaf = node.add(
                str(row["filename"]),
                data=NodeData(
                    kind="document",
                    path=row.get("path"),
                    markdown_hash=row["markdown_hash"],
                    filename=row["filename"],
                    count=row["section_count"],
                ),
            )
            leaf.allow_expand = True
        node.data.loaded = True

    def _load_sections(self, node: TreeNode, markdown_hash: str) -> None:
        toc = get_document_toc(self.backend, markdown_hash)
        by_parent: dict[str | None, list[dict[str, Any]]] = {}
        for row in toc:  # type: ignore[union-attr]
            by_parent.setdefault(row["parent_section_id"], []).append(row)

        def add_children(parent_node: TreeNode, parent_id: str | None) -> None:
            for row in sorted(by_parent.get(parent_id, []), key=lambda r: r["sequence"]):
                child = parent_node.add(
                    str(row["title"]),
                    data=NodeData(
                        kind="section",
                        markdown_hash=markdown_hash,
                        section_id=row["section_id"],
                        title=row["title"],
                        level=row["level"],
                        line_start=row["line_start"],
                    ),
                )
                child.allow_expand = bool(by_parent.get(row["section_id"]))
                add_children(child, row["section_id"])

        # Every document has a synthetic level-0 "(document root)" wrapper section
        # (see tree_parser.ROOT_SECTION_TITLE). Skip it and attach its real headings
        # directly under the document node so they show up as leaves right away —
        # unless the document has no real headings, in which case fall back to it.
        roots = by_parent.get(None, [])
        if len(roots) == 1 and roots[0]["level"] == 0 and by_parent.get(roots[0]["section_id"]):
            add_children(node, roots[0]["section_id"])
        else:
            add_children(node, None)
        node.data.loaded = True

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        node = event.node
        data: NodeData | None = node.data
        if data is None or data.loaded:
            return
        if data.kind == "folder":
            self._load_documents(node, data.path or ".")
        elif data.kind == "document":
            self._load_sections(node, data.markdown_hash or "")

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        self._show_node(event.node)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        self._show_node(event.node)

    def _set_loading(self, loading: bool) -> None:
        indicator = self.query_one("#loading", LoadingIndicator)
        content = self.query_one("#content", Markdown)
        if loading:
            indicator.add_class("active")
            content.display = False
        else:
            indicator.remove_class("active")
            content.display = True

    def _show_node(self, node: TreeNode) -> None:
        data: NodeData | None = node.data
        if data is None:
            return

        node_key = (data.kind, data.section_id or data.markdown_hash or data.path)
        if self._active_node_key == node_key:
            return
        self._active_node_key = node_key

        meta = self.query_one("#meta", Static)
        self._current_md_path = None
        self._current_origin_path = None

        if data.kind == "root":
            self._set_loading(False)
            meta.update(f"[b]Folder[/b]\n{len(self._doc_rows)} document(s)")
            self.query_one("#content", Markdown).update("")
        elif data.kind == "folder":
            self._set_loading(False)
            meta.update(f"[b]Folder[/b] {data.path}\n{data.count} document(s)")
            self.query_one("#content", Markdown).update("")
        elif data.kind == "document":
            self._current_md_path = data.path
            self._current_origin_path = _read_origin_path(data.path)
            meta_text = (
                f"[b]Document[/b] {data.filename}\n"
                f"markdown: {data.path}\n"
                f"source: {self._current_origin_path or '(none recorded — not converted from a raw document)'}\n"
                f"hash: {data.markdown_hash}\nsections: {data.count}\n"
                f"\n[dim](press [bold]m[/bold] to open the .md, [bold]o[/bold] to open the source)[/dim]\n"
                f"[yellow]⏳ Loading content...[/yellow]"
            )
            meta.update(meta_text)
            self._set_loading(True)
            self._load_content_worker("document", data.markdown_hash or "", meta_text)
        elif data.kind == "section":
            self._current_md_path = self._path_for_markdown_hash(data.markdown_hash)
            self._current_origin_path = _read_origin_path(self._current_md_path)
            meta_text = (
                f"[b]Section[/b] {data.title}\n"
                f"id: {data.section_id}\n"
                f"doc hash: {data.markdown_hash}\n"
                f"level: {data.level}   line: {data.line_start}\n"
                f"[yellow]⏳ Loading content...[/yellow]"
            )
            meta.update(meta_text)
            self._set_loading(True)
            self._load_content_worker("section", data.section_id or "", meta_text)

    @work(exclusive=True, thread=True)
    def _load_content_worker(self, kind: str, target_id: str, meta_text: str) -> None:
        try:
            if kind == "document":
                text = reconstruct_document(self.backend, target_id)
                if not text:
                    text = "*(Document content is empty or could not be reconstructed.)*"
            elif kind == "section":
                text = reconstruct_section(self.backend, target_id)
                if not text:
                    text = "*(Section content is empty.)*"
            else:
                text = ""
        except Exception as exc:
            text = f"*(Error loading content: {exc})*"

        clean_meta = meta_text.replace("\n[yellow]⏳ Loading content...[/yellow]", "")
        self.app.call_from_thread(self._finish_content, text, clean_meta)

    async def _finish_content(self, text: str, clean_meta: str) -> None:
        self.query_one("#meta", Static).update(clean_meta)
        content = self.query_one("#content", Markdown)
        await content.update(text)
        self._set_loading(False)
        self.query_one("#info-panel", VerticalScroll).scroll_to(y=0, animate=False)

    def _path_for_markdown_hash(self, markdown_hash: str | None) -> str | None:
        """Look up a document's converted-Markdown path by its content hash."""
        if not markdown_hash:
            return None
        return next((row.get("path") for row in self._doc_rows if row["markdown_hash"] == markdown_hash), None)

    def action_refresh_tree(self) -> None:
        self._rebuild_tree()

    def action_open_markdown(self) -> None:
        """Open the converted Markdown (.md) file in the system's default application."""
        self._open_path(self._current_md_path, what="Markdown file")

    def action_open_source(self) -> None:
        """Open the original source document (PDF/DOCX/...) the Markdown was converted from."""
        if not self._current_origin_path:
            self.notify("No source document recorded for this file — opening the Markdown instead.", severity="warning")
            self._open_path(self._current_md_path, what="Markdown file")
            return
        self._open_path(self._current_origin_path, what="source document")

    def _open_path(self, path: str | None, *, what: str) -> None:
        if not path:
            self.notify("No file selected.", severity="warning")
            return

        try:
            import subprocess
            import sys

            file_path = Path(path)
            if not file_path.exists():
                self.notify(f"{what} not found: {file_path}", severity="error")
                return

            # Open file with system default application
            if sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", str(file_path)])
            elif sys.platform == "win32":  # Windows
                subprocess.Popen(["explorer", str(file_path)])
            else:  # Linux and others
                subprocess.Popen(["xdg-open", str(file_path)])
        except Exception as e:
            self.notify(f"Failed to open {what}: {e}", severity="error")


def run_document_graph_tui(db_path: str) -> None:
    """Launch the Document Graph TUI over a Ladybug database."""
    DocumentGraphApp(db_path).run()
