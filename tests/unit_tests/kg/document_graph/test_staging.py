"""Unit tests for Document Graph Parquet staging and ingestion."""

from __future__ import annotations

import base64
from pathlib import Path

import pyarrow.parquet as pq

from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.document_graph.staging import (
    ingest_document_graph_from_staging,
    stage_document_graph_to_parquet,
)
from genai_graph.kg.factories.document_graph_factory import DocumentGraphBundle
from genai_graph.kg.nodes.document import Document, Folder
from genai_graph.kg.nodes.document_section import Image, MarkdownSection


def test_stage_and_ingest_document_graph_parquet(tmp_path: Path) -> None:
    # 1. Create a synthetic DocumentGraphBundle with Folders, Document, Section, Image
    folder = Folder(
        folder_id="root_folder",
        name="root",
        path=str(tmp_path),
        relative_path="",
        uri=str(tmp_path),
    )
    doc = Document(
        content_hash="doc_hash_123",
        markdown_hash="md_hash_123",
        filename="report.md",
        folder_id="root_folder",
        relative_path="report.md",
        path=str(tmp_path / "report.md"),
        section_count=1,
        language="en",
    )
    sec = MarkdownSection(
        section_id="md_hash_123::0",
        markdown_hash="md_hash_123",
        title="Executive Summary",
        level=1,
        line_start=1,
        line_end=10,
        text="## Executive Summary\n\nAll figures in millions.",
        token_count=10,
        sequence=0,
        description="Summary of performance",
    )
    img = Image(
        image_id="md_hash_123::0::img1",
        section_id="md_hash_123::0",
        markdown_hash="md_hash_123",
        image_hash="img1",
        filename="chart1.png",
        format="png",
        caption="Figure 1: Net Income",
        description="Quarterly net income chart",
        base64_data=base64.b64encode(b"fake image bytes").decode("ascii"),
        file_size_bytes=16,
    )
    bundle = DocumentGraphBundle(
        folders=[folder],
        document=doc,
        sections=[sec],
        images=[img],
    )

    # 2. Stage to Parquet
    staging_dir = tmp_path / "staging"
    stats = stage_document_graph_to_parquet([bundle], staging_dir)
    assert stats.total_files_staged > 0
    assert (staging_dir / "manifest.json").exists()
    assert (staging_dir / "nodes" / "Folder.parquet").exists()
    assert (staging_dir / "nodes" / "Document.parquet").exists()
    assert (staging_dir / "nodes" / "MarkdownSection.parquet").exists()
    assert (staging_dir / "nodes" / "Image.parquet").exists()
    assert (staging_dir / "rels" / "HAS_IMAGE.parquet").exists()

    # Verify Image Parquet contains base64 data
    img_table = pq.read_table(str(staging_dir / "nodes" / "Image.parquet"))
    img_df = img_table.to_pandas()
    assert len(img_df) == 1
    assert img_df.iloc[0]["filename"] == "chart1.png"
    assert img_df.iloc[0]["base64_data"] == base64.b64encode(b"fake image bytes").decode("ascii")

    # 3. Ingest into Ladybug DB
    db_path = tmp_path / "test.db"
    backend = KuzuBackend()
    backend.connect(str(db_path))
    try:
        res = ingest_document_graph_from_staging(backend, staging_dir)
        assert res["status"] == "success"
        assert res["nodes_ingested"] >= 4
        assert res["rels_ingested"] >= 3

        # Query Image node directly
        df = backend.execute_get_as_df(
            "MATCH (s:MarkdownSection)-[:HAS_IMAGE]->(i:Image) RETURN s.title, i.filename, i.caption, i.base64_data"
        )
        assert len(df) == 1
        assert df.iloc[0]["i.filename"] == "chart1.png"
        assert df.iloc[0]["i.caption"] == "Figure 1: Net Income"
        assert df.iloc[0]["i.base64_data"] == base64.b64encode(b"fake image bytes").decode("ascii")

        # 4. Verify TOC includes image captions / descriptions
        from genai_graph.kg.query.document_graph_tools import (
            build_toc_tree,
            get_document_toc,
            render_toc_outline,
        )

        toc_rows = get_document_toc(backend, "doc_hash_123")
        assert len(toc_rows) == 1
        assert "images" in toc_rows[0]
        assert len(toc_rows[0]["images"]) == 1
        assert toc_rows[0]["images"][0]["caption"] == "Figure 1: Net Income"

        outline = render_toc_outline(toc_rows)
        assert "Figure 1: Net Income" in outline

        toc_tree = build_toc_tree(toc_rows)
        assert "images" in toc_tree[0]
        assert toc_tree[0]["images"][0]["filename"] == "chart1.png"
    finally:
        backend.close()
