"""Token-budget chunking for Document Graph sections.

A ``MarkdownSection`` whose ``text`` exceeds the chunk budget is split into
consecutive ~budget-token pieces at paragraph (blank-line) boundaries, so a
chunk never cuts a paragraph in half. When a paragraph contains a large Markdown
pipe table exceeding the budget, it is split row-by-row with repeated column
headers and unit annotations so sub-tables preserve full tabular context.
Non-table paragraphs larger than the budget are split by words. Short sections
yield a single chunk equal to the whole section text. Chunks are the unit of
semantic embedding (see :mod:`genai_graph.kg.document_graph.retrieval`); the
section hierarchy itself is unchanged.
"""

from __future__ import annotations

import re

from genai_tk.utils.tokens import count_tokens
from genai_tk.workflow.rag.chonkie_splitter import is_markdown_table, split_markdown_table

_PARA_SPLIT = re.compile(r"\n\s*\n")


def split_paragraphs(text: str) -> list[str]:
    """Return the non-empty paragraph blocks of *text* (split on blank lines)."""
    return [p for p in _PARA_SPLIT.split(text) if p.strip()]


def _hard_split(text: str, *, size_tokens: int) -> list[str]:
    """Split a non-table paragraph with no usable paragraph breaks that exceeds the budget, by words."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w) if cur else w
        if cur and count_tokens(candidate) > size_tokens:
            chunks.append(cur)
            cur = w
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    return chunks


def chunk_section_text(text: str, *, size_tokens: int) -> list[tuple[str, int]]:
    """Split *text* into chunks of at most ~*size_tokens* tokens.

    Preserves Markdown tables by splitting large tables row-wise with repeated
    headers on continuation chunks.

    Args:
        text: A section's own Markdown text (heading line + body).
        size_tokens: Target chunk size; sections longer than this are split into
            consecutive ~*size_tokens*-token pieces at paragraph boundaries.

    Returns:
        A list of ``(chunk_text, token_count)`` pairs in document order. Short
        or empty text yields a single chunk (or none for blank text).
    """
    if not text or not text.strip():
        return []
    if count_tokens(text) <= size_tokens:
        return [(text, count_tokens(text))]

    chunks: list[str] = []
    buf = ""
    buf_tokens = 0
    for para in split_paragraphs(text):
        para_tokens = count_tokens(para)
        if para_tokens > size_tokens:
            if buf:
                chunks.append(buf)
                buf, buf_tokens = "", 0
            if is_markdown_table(para):
                table_chunks = split_markdown_table(para, max_tokens=size_tokens)
                chunks.extend(table_chunks)
            else:
                chunks.extend(_hard_split(para, size_tokens=size_tokens))
            continue
        if buf and buf_tokens + para_tokens > size_tokens:
            chunks.append(buf)
            buf, buf_tokens = "", 0
        buf = (buf + "\n\n" + para) if buf else para
        buf_tokens += para_tokens
    if buf:
        chunks.append(buf)
    return [(c, count_tokens(c)) for c in chunks]
