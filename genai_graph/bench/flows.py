"""Prefect-orchestrated flows and tasks for benchmark pipelines.

Provides parallelized tasks with in-process concurrency control, automatic retries,
and pipeline stages: fetch -> markdownize -> build -> run -> grade.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

from loguru import logger
from prefect import flow, task

from genai_graph.bench.adapters.base import get_benchmark_adapter
from genai_graph.bench.config import BenchConfig, configure_bench_monitoring
from genai_graph.bench.judge import evaluate_single_run, load_existing_scores
from genai_graph.bench.models import BenchQuestion, BenchRunRecord, BenchScoreRecord
from genai_graph.bench.runner import append_run_record, load_existing_runs, run_one_question

_QUESTION_SEMAPHORES: dict[int, threading.Semaphore] = {}
_JUDGE_SEMAPHORES: dict[int, threading.Semaphore] = {}


def _get_semaphore(cache: dict[int, threading.Semaphore], capacity: int) -> threading.Semaphore:
    if capacity not in cache:
        cache[capacity] = threading.Semaphore(max(1, capacity))
    return cache[capacity]


# ---------------------------------------------------------------------------
# Prefect Tasks
# ---------------------------------------------------------------------------


@task(retries=3, retry_delay_seconds=2, task_run_name="fetch-doc-{doc_name}")
def fetch_doc_task(doc_name: str, *, pdfs_dir: str, adapter_name: str | None = None) -> str:
    """Download a raw benchmark document PDF/file using the adapter."""
    adapter = get_benchmark_adapter(adapter_name)
    target = adapter.fetch_document(doc_name, output_dir=Path(pdfs_dir))
    return str(target)


@task(retries=2, retry_delay_seconds=2, task_run_name="markdownize-{doc_name}")
def markdownize_doc_task(
    doc_name: str,
    *,
    force: bool,
    pdfs_dir: str,
    saved_markdown_dir: str,
    markdownize_profile: str,
    markdown_dir: str,
    skip_ocr: bool,
) -> str:
    """Convert/OCR a document to Markdown and stage it in the project markdown directory."""
    from genai_graph.bench.build_graph import (
        MD_FILENAME_SUFFIX,
        copy_markdown_to_project,
        markdownize_target,
    )

    saved_p = Path(saved_markdown_dir)
    md_p = Path(markdown_dir)
    md_p.mkdir(parents=True, exist_ok=True)

    if skip_ocr:
        source_md = saved_p / f"{doc_name}{MD_FILENAME_SUFFIX}"
        if not source_md.exists():
            raise FileNotFoundError(f"--skip-ocr requested but markdown file missing: {source_md}")
    else:
        source_md = markdownize_target(
            doc_name,
            force=force,
            pdfs_dir=Path(pdfs_dir),
            saved_markdown_dir=saved_p,
            markdownize_profile=markdownize_profile,
        )

    dest = copy_markdown_to_project(source_md, markdown_dir=md_p)
    return str(dest)


@task(task_run_name="batch-markdownize")
def batch_markdownize_task(
    docs: list[str],
    *,
    force: bool,
    pdfs_dir: str,
    saved_markdown_dir: str,
    markdownize_profile: str,
    markdown_dir: str,
    skip_ocr: bool,
) -> list[str]:
    """Convert/OCR benchmark documents in batch (using Batch OCR API when available) and stage in markdown_dir."""
    from genai_graph.bench.build_graph import (
        MD_FILENAME_SUFFIX,
        copy_markdown_to_project,
        markdownize_targets_batch,
    )

    saved_p = Path(saved_markdown_dir)
    md_p = Path(markdown_dir)
    md_p.mkdir(parents=True, exist_ok=True)

    if skip_ocr:
        results = []
        for doc in docs:
            source_md = saved_p / f"{doc}{MD_FILENAME_SUFFIX}"
            if not source_md.exists():
                raise FileNotFoundError(f"--skip-ocr requested but markdown file missing: {source_md}")
            dest = copy_markdown_to_project(source_md, markdown_dir=md_p)
            results.append(str(dest))
        return results

    saved_md_files = markdownize_targets_batch(
        docs,
        force=force,
        pdfs_dir=Path(pdfs_dir),
        saved_markdown_dir=saved_p,
        markdownize_profile=markdownize_profile,
    )
    results = []
    for md_file in saved_md_files:
        dest = copy_markdown_to_project(md_file, markdown_dir=md_p)
        results.append(str(dest))
    return results


@task(retries=2, retry_delay_seconds=5, task_run_name="outline-{doc_name}")
def extract_outline_task(
    doc_name: str,
    *,
    markdown_dir: str,
    kg_db: str,
    build_llm: str | None,
    structure_strategy: str = "auto",
    generate_summaries: bool = True,
    workers: int = 4,
    summary_min_tokens: int = 800,
    context_safety_ratio: float = 0.9,
) -> dict[str, Any]:
    """Extract one document's outline (LLM) without touching the database.

    Per-document Prefect task: failures are isolated to this document, retries
    are cheap (content-addressed cache), and the merge task degrades the doc to
    algorithmic parsing if extraction ultimately fails.
    """
    from genai_graph.bench.build_graph import warm_outline_cache

    return warm_outline_cache(
        [doc_name],
        markdown_dir=Path(markdown_dir),
        kg_db=Path(kg_db),
        llm=build_llm,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        workers=workers,
        summary_min_tokens=summary_min_tokens,
        context_safety_ratio=context_safety_ratio,
    )


@task(task_run_name="merge-document-graph")
def merge_graph_task(
    docs: list[str],
    *,
    markdown_dir: str,
    kg_db: str,
    force: bool,
    build_llm: str | None,
    structure_strategy: str = "auto",
    generate_summaries: bool = True,
    workers: int = 4,
    summary_min_tokens: int = 800,
    context_safety_ratio: float = 0.9,
    embeddings_id: str | None = None,
    fts: bool = True,
    chunk_size_tokens: int = 1500,
) -> dict[str, Any]:
    """Ingest Markdown files into the Ladybug Document Graph (single-writer).

    Reads outlines from the cache warmed by `extract_outline_task` (no LLM
    calls), embeds chunks concurrently in-process, and merges. One task because
    Ladybug allows a single read-write Database per file per process.
    """
    from genai_graph.bench.build_graph import build_document_graph

    return build_document_graph(
        doc_names=docs,
        force=force,
        llm=build_llm,
        structure_strategy=structure_strategy,
        generate_summaries=generate_summaries,
        workers=workers,
        summary_min_tokens=summary_min_tokens,
        context_safety_ratio=context_safety_ratio,
        markdown_dir=Path(markdown_dir),
        kg_db=Path(kg_db),
        embeddings_id=embeddings_id,
        fts=fts,
        chunk_size_tokens=chunk_size_tokens,
        outline_pre_pass=False,
    )


@task(retries=2, retry_delay_seconds=3, task_run_name="run-question-{q.id}")
def run_question_task(
    q: BenchQuestion,
    *,
    llm: str,
    db_path: str,
    folder_id: str | None = None,
    profile_name: str = "default",
    embeddings_id: str | None = None,
    runs_path: str | None = None,
    concurrency: int = 10,
) -> BenchRunRecord:
    """Run a question through the docgraph agent harness with concurrency gating."""
    from genai_tk.agents.harness.profiles import load_langchain_profiles

    from genai_graph.agent import create_docgraph_agent

    sem = _get_semaphore(_QUESTION_SEMAPHORES, concurrency)

    async def _execute() -> BenchRunRecord:
        profiles = load_langchain_profiles()
        if profile_name not in profiles:
            raise KeyError(f"Agent profile '{profile_name}' not found. Available: {sorted(profiles)}")
        profile = profiles[profile_name]
        harness = create_docgraph_agent(
            profile,
            llm=llm,
            db_path=db_path,
            folder_id=folder_id,
            embeddings_id=embeddings_id,
        )
        try:
            return await run_one_question(harness, q, llm)
        finally:
            await harness.aclose()

    with sem:
        record = asyncio.run(_execute())

    if runs_path:
        append_run_record(record, Path(runs_path))

    return record


@task(retries=5, retry_delay_seconds=3, task_run_name="grade-run-{run.id}")
def grade_run_task(
    run: BenchRunRecord,
    *,
    judge_llm: str,
    system_rubric: str,
    scores_path: str | None = None,
    concurrency: int = 5,
) -> BenchScoreRecord:
    """Grade a question run with the LLM-as-judge."""
    sem = _get_semaphore(_JUDGE_SEMAPHORES, concurrency)

    with sem:
        score = asyncio.run(evaluate_single_run(judge_llm, run, system_rubric=system_rubric))

    if scores_path:
        from genai_graph.bench.judge import append_score_record

        append_score_record(score, Path(scores_path))

    return score


# ---------------------------------------------------------------------------
# Prefect Flows
# ---------------------------------------------------------------------------


@flow(name="bench-fetch")
def fetch_flow(cfg: BenchConfig) -> list[str]:
    """Fetch raw documents in parallel."""
    docs = cfg.files.docs
    logger.info("Fetching {} document(s) in parallel...", len(docs))
    futures = [
        fetch_doc_task.submit(
            doc,
            pdfs_dir=cfg.docgraph.paths.sources_dir,
            adapter_name=cfg.dataset_adapter,
        )
        for doc in docs
    ]
    return [f.result() for f in futures]


@flow(name="bench-markdownize")
def markdownize_flow(cfg: BenchConfig) -> list[str]:
    """Convert/OCR documents to Markdown in batch."""
    docs = cfg.files.docs
    logger.info("Markdownizing {} document(s)...", len(docs))
    if not docs:
        return []
    return batch_markdownize_task(
        docs,
        force=cfg.docgraph.build.force,
        pdfs_dir=cfg.docgraph.paths.sources_dir,
        saved_markdown_dir=cfg.docgraph.paths.saved_markdown_dir,
        markdownize_profile=cfg.docgraph.markdownize_profile,
        markdown_dir=cfg.docgraph.paths.markdown_dir,
        skip_ocr=cfg.docgraph.build.skip_ocr,
    )


@flow(name="bench-build-graph")
def build_graph_flow(cfg: BenchConfig) -> dict[str, Any]:
    """Build the Ladybug Document Graph from staged Markdown files.

    Fan-out: one per-document outline-extraction task (LLM, DB-free, retried and
    isolated per document). Fan-in: one single-writer merge task that reads the
    warmed outline cache and embeds/merges. A document whose outline task failed
    after retries still merges, degraded to algorithmic parsing.
    """
    docs = cfg.files.docs
    llm_arg = cfg.docgraph.llms.summary
    logger.info(
        "Building Document Graph (db={}, llm={})...",
        cfg.docgraph.paths.kg_db,
        llm_arg or "algorithmic",
    )

    outline_futures = [
        extract_outline_task.submit(
            doc,
            markdown_dir=cfg.docgraph.paths.markdown_dir,
            kg_db=cfg.docgraph.paths.kg_db,
            build_llm=llm_arg,
            structure_strategy=cfg.docgraph.build.structure_strategy,
            generate_summaries=cfg.docgraph.build.generate_summaries,
            workers=cfg.docgraph.build.workers,
            summary_min_tokens=cfg.docgraph.build.summary_min_tokens,
            context_safety_ratio=cfg.docgraph.build.context_safety_ratio,
        )
        for doc in docs
    ]
    for doc, fut in zip(docs, outline_futures, strict=True):
        try:
            fut.result()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Outline extraction failed for {}: {}; merge will degrade it to algorithmic parsing",
                doc,
                exc,
            )

    future = merge_graph_task.submit(
        docs,
        markdown_dir=cfg.docgraph.paths.markdown_dir,
        kg_db=cfg.docgraph.paths.kg_db,
        force=cfg.docgraph.build.force,
        build_llm=llm_arg,
        structure_strategy=cfg.docgraph.build.structure_strategy,
        generate_summaries=cfg.docgraph.build.generate_summaries,
        workers=cfg.docgraph.build.workers,
        summary_min_tokens=cfg.docgraph.build.summary_min_tokens,
        context_safety_ratio=cfg.docgraph.build.context_safety_ratio,
        embeddings_id=None,
        fts=cfg.docgraph.build.fts,
        chunk_size_tokens=cfg.docgraph.build.chunk_size_tokens,
    )
    return future.result()


@flow(name="bench-run-questions")
def run_questions_flow(
    cfg: BenchConfig,
    questions: list[BenchQuestion] | None = None,
) -> list[str]:
    """Run questions in parallel through the docgraph agent.

    Returns the run IDs covered (cached + newly executed). Full run records stay
    on disk in the runs JSONL — passing them between Prefect flows would blow the
    flow-run parameter size limit once tool results accumulate.
    """
    configure_bench_monitoring(cfg.runner.monitoring, project_name=f"bench-{cfg.profile_name}")

    if questions is None:
        adapter = get_benchmark_adapter(cfg.dataset_adapter)
        all_qs = adapter.load_dataset()
        docs = cfg.files.docs
        doc_set = {d.removesuffix(".pdf") for d in docs} | set(docs)
        questions = [
            q
            for q in all_qs
            if not doc_set or any(d.removesuffix(".pdf") in doc_set or d in doc_set for d in q.doc_names)
        ]
        if cfg.question_ids:
            q_set = set(cfg.question_ids)
            questions = [q for q in questions if q.id in q_set]
        if cfg.files.limit:
            questions = questions[: cfg.files.limit]

    runs_path = Path(cfg.runs)
    existing_runs = {} if cfg.force_run else load_existing_runs(runs_path)

    to_run: list[BenchQuestion] = []
    cached_runs: list[BenchRunRecord] = []
    for q in questions:
        if q.id in existing_runs:
            cached_runs.append(existing_runs[q.id])
        else:
            to_run.append(q)

    if cached_runs:
        logger.info("Reusing {} existing run record(s) from {}", len(cached_runs), runs_path)

    if not to_run:
        return [r.id for r in cached_runs]

    logger.info("Executing {} question(s) (concurrency={})...", len(to_run), cfg.runner.concurrency)
    futures = [
        run_question_task.submit(
            q,
            llm=cfg.agent_llm,
            db_path=cfg.docgraph.paths.kg_db,
            folder_id=cfg.runner.folder_id,
            profile_name=cfg.agent_profile,
            embeddings_id=None,
            runs_path=cfg.runs,
            concurrency=cfg.runner.concurrency,
        )
        for q in to_run
    ]
    new_run_ids = [f.result().id for f in futures]
    return [r.id for r in cached_runs] + new_run_ids


@flow(name="bench-grade")
def grade_flow(
    cfg: BenchConfig,
    run_ids: list[str] | None = None,
) -> list[BenchScoreRecord]:
    """Grade question runs using LLM-as-judge.

    Run records are always loaded from the runs JSONL (never passed through flow
    parameters — they carry large tool results that exceed Prefect's parameter
    size limit). When *run_ids* is given, only those runs are graded; otherwise
    every record in the runs file is graded.
    """
    runs_path = Path(cfg.runs)
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs file not found: {runs_path}. Execute 'run' step first.")
    runs = list(load_existing_runs(runs_path).values())
    if run_ids is not None:
        id_set = set(run_ids)
        runs = [r for r in runs if r.id in id_set]

    adapter = get_benchmark_adapter(cfg.dataset_adapter)
    rubric = adapter.get_judge_rubric()
    scores_path = Path(cfg.scores)
    existing_scores = load_existing_scores(scores_path)

    to_grade: list[BenchRunRecord] = []
    already_graded: list[BenchScoreRecord] = []

    for r in runs:
        if r.id in existing_scores:
            s_dict = existing_scores[r.id]
            from genai_graph.bench.models import JudgeVerdict

            already_graded.append(
                BenchScoreRecord(
                    run=r,
                    verdict=JudgeVerdict(
                        correctness=s_dict.get("correctness", "incorrect"),
                        numeric_match=s_dict.get("numeric_match"),
                        groundedness=s_dict.get("groundedness", "partial"),
                        error_category=s_dict.get("error_category"),
                        rationale=s_dict.get("rationale", ""),
                    ),
                    judge_llm=s_dict.get("judge_llm", cfg.grader.llm),
                    scored_at=s_dict.get("scored_at"),
                )
            )
        else:
            to_grade.append(r)

    if already_graded:
        logger.info("Reusing {} existing score(s) from {}", len(already_graded), scores_path)

    if not to_grade:
        return already_graded

    logger.info("Grading {} run(s) with judge LLM {}...", len(to_grade), cfg.grader.llm)
    futures = [
        grade_run_task.submit(
            run,
            judge_llm=cfg.grader.llm,
            system_rubric=rubric,
            scores_path=cfg.scores,
            concurrency=cfg.grader.concurrency,
        )
        for run in to_grade
    ]
    new_scores = [f.result() for f in futures]
    return already_graded + new_scores


@flow(name="bench-full")
def full_bench_flow(cfg: BenchConfig, step: str | None = None, skip: list[str] | None = None) -> dict[str, Any]:
    """Execute the end-to-end benchmark workflow or specific steps."""
    skip_set = set(skip or [])
    outcome: dict[str, Any] = {}

    def _should_run(stage: str) -> bool:
        if step:
            return step == stage
        return stage not in skip_set

    # 1. Fetch
    if _should_run("fetch"):
        outcome["fetched"] = fetch_flow(cfg)

    # 2. Markdownize
    if _should_run("fetch") or _should_run("build"):
        outcome["markdownized"] = markdownize_flow(cfg)

    # 3. Build Graph
    if _should_run("build"):
        outcome["graph"] = build_graph_flow(cfg)

    # 4. Run Questions
    run_ids: list[str] | None = None
    if _should_run("run"):
        run_ids = run_questions_flow(cfg)
        outcome["runs"] = len(run_ids)

    # 5. Grade
    if _should_run("grade") and cfg.grader.enabled:
        scores = grade_flow(cfg, run_ids=run_ids)
        outcome["scores"] = len(scores)

        # Compute and persist summary
        from genai_graph.bench.summary import (
            compute_bench_summary,
            display_bench_summary,
            save_bench_summary,
        )

        summary = compute_bench_summary(
            scores,
            profile_name=cfg.profile_name,
            agent_llm=cfg.agent_llm,
            judge_llm=cfg.grader.llm,
        )
        save_bench_summary(summary, Path(cfg.scores_summary))
        display_bench_summary(summary)
        outcome["summary"] = summary.model_dump()

    return outcome
