"""Unit tests for Document Graph image extraction and caption parsing."""

from genai_graph.kg.document_graph.images import extract_section_images
from genai_graph.kg.nodes.document_section import MarkdownSection


def test_extract_section_images_mistral_comment():
    text = """## Section 1

Some introductory text.

<!-- Image: 7a8b9c0d.png (hash: 7a8b9c0d) -->
![Survey Results](images/7a8b9c0d.png)
*Fig. 1: Trends in economic mobility among Latinos (2008-2015)*

Further explanation follows.
"""
    sec = MarkdownSection(
        section_id="doc1::0",
        markdown_hash="doc1",
        title="Section 1",
        level=1,
        line_start=1,
        line_end=12,
        text=text,
        token_count=50,
        sequence=0,
    )

    images = extract_section_images(sec)
    assert len(images) == 1
    img = images[0]
    assert img.image_hash == "7a8b9c0d"
    assert img.filename == "7a8b9c0d.png"
    assert img.description == "Fig. 1: Trends in economic mobility among Latinos (2008-2015)"
    assert img.section_id == "doc1::0"
    assert img.markdown_hash == "doc1"
    assert img.image_id == "doc1::0::7a8b9c0d"


def test_extract_section_images_markdown_syntax_fallback():
    text = """## Section 2

![Detailed flowchart of the architecture](assets/diagram.png)

_Figure 2: Architecture diagram of the pipeline._
"""
    sec = MarkdownSection(
        section_id="doc1::1",
        markdown_hash="doc1",
        title="Section 2",
        level=1,
        line_start=13,
        line_end=20,
        text=text,
        token_count=30,
        sequence=1,
    )

    images = extract_section_images(sec)
    assert len(images) == 1
    img = images[0]
    assert img.image_hash == "diagram"
    assert img.filename == "diagram.png"
    assert "Figure 2: Architecture diagram" in (img.description or "")


def test_extract_section_images_alt_text_fallback():
    text = """## Section 3

![Distribution of surveyed population across age groups](assets/chart_123.jpg)
"""
    sec = MarkdownSection(
        section_id="doc1::2",
        markdown_hash="doc1",
        title="Section 3",
        level=1,
        line_start=21,
        line_end=26,
        text=text,
        token_count=20,
        sequence=2,
    )

    images = extract_section_images(sec)
    assert len(images) == 1
    img = images[0]
    assert img.description == "Distribution of surveyed population across age groups"
