"""Domain ingestion flows for genai-graph, selected via ingest route tables.

These flows satisfy the router's ingestion contract — ``sources: list[str]``
plus ``md_output_dir: str`` — and are referenced from
``config/workflows/data_injection.yaml``. The ``json_document`` workflow
replaces the bench's hard-coded JSON special case (Migration Phase 4 in
``docs/studies/rule_selected_ingest_workflows.md``): a route rule like
``pathspec: "**/*.json"`` dispatches structured JSON documents here.

Example:

```yaml
ingest_routes:
  my_corpus:
    routes:
      - pathspec: "**/*.json"
        workflow: json_document
      - pathspec: "**/*"
        workflow: markdownize_documents
        with:
          profile: best
```
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from prefect import flow, task


def _json_to_markdown(path: Path) -> str:
    """Return Markdown text for one structured JSON document.

    Uses the consuming project's transformer when it ships one (e.g. OfficeQA's
    ``parse_json_document_to_markdown``); otherwise falls back to a fenced JSON
    code block titled with the document stem.
    """
    try:
        from officeqa.json_transformer import parse_json_document_to_markdown

        return parse_json_document_to_markdown(path)
    except ImportError:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return f"# {path.stem}\n\n```json\n{json.dumps(raw, indent=2)}\n```\n"


@task(log_prints=False)
def _json_document_task(src: str, out_abs: str, rel_out: str, force: bool) -> tuple[str, str]:
    """Convert one JSON document to Markdown, skipping unchanged cached outputs."""
    out_path = Path(out_abs)
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        logger.debug(f"JSON document already converted: {out_path}")
        return src, rel_out

    content = _json_to_markdown(Path(src))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    logger.success(f"Wrote Markdown ({len(content)} bytes) -> {out_path}")
    return src, rel_out


@flow(name="json_document")
def json_document_flow(sources: list[str], md_output_dir: str, *, force_stage: str | None = None) -> dict[str, int]:
    """Convert structured JSON documents to Markdown under ``md_output_dir``.

    Output files follow the markdownize naming convention (``<stem>_<suffix>.md``),
    so routed corpora converge to the same staging contract as any other ingestion
    workflow.

    Args:
        sources: JSON files to convert.
        md_output_dir: Directory the Markdown files are written to.
        force_stage: Cache-invalidation stage; from ``md`` on, conversion is redone.

    Returns:
        Summary dict with the number of converted documents.
    """
    from genai_tk.workflow.force import ForceStage, stage_active

    force = stage_active(force_stage, ForceStage.md)
    output_dir = Path(md_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    for spec in sources:
        src = Path(spec)
        rel_out = f"{src.stem}_{src.suffix.lstrip('.')}.md"
        futures.append(_json_document_task.submit(str(src), str(output_dir / rel_out), rel_out, force))
    for future in futures:
        future.result()

    return {"processed": len(futures)}
