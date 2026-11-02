"""Decision-model navigation over the Document Graph (DecisionRAG study).

Implements the paper's core mechanism — *Retriever-Free Retrieval: Navigating
Documents with Decision Models* (arXiv:2610.35868) — on top of the Document
Graph hierarchy (``Folder ─CONTAINS→ Document ─HAS_SECTION→ MarkdownSection``)
and the per-node routing text (``description`` / ``summary``) generated at
ingestion.

At each level of the hierarchy every candidate branch becomes one ``Noul``
(yes/no) question; all candidates of a level are answered **jointly in a single
forward pass against a shared context** (the query plus optional domain routing
knowledge) via :meth:`BaseDecisionModel.batch_invoke`, generating zero tokens.
Only the top-k branches are expanded; pruned branches are never visited. The
output is an :class:`EvidencePack` that a synthesis LLM reads in one shot —
navigation itself uses no LLM.

The navigator is deterministic and replayable: every scored probability, call
and token is recorded in a :class:`NavigationTrace` that is serialized into the
tool result (``[[navigation-trace]]`` footer line) so the bench runner can log
a per-question ``navigation`` block in ``runs.jsonl``.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from genai_tk.core.decision_models import BaseDecisionModel, Noul, NoulAnswer
from genai_tk.core.factories.decision_model_factory import get_decision_model
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from genai_graph.kg.backend import KgBackend, KuzuBackend
from genai_graph.kg.connections import release_connection
from genai_graph.kg.query.document_graph_tools import (
    _connect,
    get_document,
    get_document_toc,
    get_section_content,
    list_documents,
    search_sections,
)

TRACE_MARKER = "[[navigation-trace]]"

_HINT_CHARS = 400

_ANSWERABLE_QUESTION_ID = "__answerable__"


class DecisionNavigatorConfig(BaseModel):
    """Tuning knobs of the decision-model navigator.

    All fields are exposed as tool-factory kwargs in agent profiles (the
    ``factory:`` tool-spec passes every extra key to
    :func:`genai_graph.agent.decision_navigation.create_decision_navigation_tools`).
    """

    model_config = ConfigDict(extra="ignore")

    decision_model: str = "fast"
    beam_width: int = 3
    top_r_sections: int = 6
    min_confidence: float = 0.5
    # Per-level confidence gates. L1 scores are compressed low because ~48 documents
    # are scored jointly against one query (correct picks measured at 0.26), while L2
    # separates a section within one document (correct picks 0.89-0.96). A single
    # absolute 0.5 gate therefore flagged correct L1 selections as low-confidence and
    # pushed the agent into the expensive fallback loop.
    l1_min_confidence: float = 0.15
    l2_min_confidence: float = 0.5
    # Do not descend into a child section scoring below this floor; keep the parent.
    # Descent otherwise picks noise (observed child scores of 0.02-0.07).
    descent_min_confidence: float = 0.2
    max_depth: int = 2
    candidate_cap: int = 48
    prefilter: str = "off"
    domain_context: str | None = None
    answerable_gate: bool = False
    pack_char_budget: int = 20000
    l2_workers: int = 1
    trace_top_k: int = 12


class Candidate(BaseModel):
    """One scorable branch of the hierarchy (a document or a section)."""

    id: str
    label: str
    hint: str = ""
    parent_id: str | None = None


class LevelTrace(BaseModel):
    """Record of one scored level of the beam search."""

    level: int
    n_candidates: int
    scores: dict[str, float] = Field(default_factory=dict)
    kept: list[str] = Field(default_factory=list)


class NavigationTrace(BaseModel):
    """Decision-model telemetry accumulated over one ``navigate`` call."""

    dm_calls: int = 0
    dm_input_tokens: int = 0
    dm_output_tokens: int = 0
    dm_cost: float | None = None
    levels: list[LevelTrace] = Field(default_factory=list)
    low_confidence: bool = False
    unanswerable: bool = False
    elapsed_ms: float = 0.0

    def merge(self, other: NavigationTrace) -> None:
        """Fold another trace (e.g. computed on a worker thread) into this one."""
        self.dm_calls += other.dm_calls
        self.dm_input_tokens += other.dm_input_tokens
        self.dm_output_tokens += other.dm_output_tokens
        if other.dm_cost is not None:
            self.dm_cost = round((self.dm_cost or 0.0) + other.dm_cost, 6)
        self.levels.extend(other.levels)
        self.low_confidence = self.low_confidence or other.low_confidence
        self.unanswerable = self.unanswerable or other.unanswerable

    def merge_usage(self, response: Any) -> None:
        """Accumulate usage from a ``ClassifierResponse``."""
        self.dm_calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.dm_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.dm_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        cost = getattr(usage, "cost", None)
        if cost is not None:
            self.dm_cost = round((self.dm_cost or 0.0) + float(cost), 6)


class SectionEvidence(BaseModel):
    """One fetched section's text plus its navigation score."""

    section_id: str
    title: str | None = None
    document_id: str = ""
    document_name: str | None = None
    score: float = 0.0
    text: str = ""


class EvidencePack(BaseModel):
    """Selected sections + telemetry returned by :meth:`DecisionNavigator.navigate`."""

    query: str
    sections: list[SectionEvidence] = Field(default_factory=list)
    low_confidence: bool = False
    unanswerable: bool = False
    trace: NavigationTrace = Field(default_factory=NavigationTrace)

    def render(self, pack_char_budget: int | None = None) -> str:
        """Render the pack as the ``navigate_folder`` tool result text.

        Sections are included best-score-first under *pack_char_budget*; sections
        that no longer fit are replaced by a pointer so the synthesis agent can
        fetch them selectively with ``get_section_content`` — an oversized pack
        must never trigger observation eviction (the pack-paging failure mode
        observed in the first subset run, docs/studies §6.5).

        Ends with a single ``[[navigation-trace]]`` JSON footer line that the
        bench runner extracts into the run record's ``navigation`` block; the
        synthesis agent also reads it to apply the low-confidence fallback.
        """
        footer = f"\n--- navigation trace ---\n{TRACE_MARKER} {self.trace.model_dump_json()}"
        if self.unanswerable:
            return (
                "The decision model judged this question NOT answerable from the available "
                "documents (answerable probability below threshold). If you have reasons to "
                "doubt this (e.g. the question requires external facts or visual data), fall "
                f"back to the graph tools directly.{footer}"
            )
        if not self.sections:
            return (
                "No relevant sections were found for this question. Fall back to "
                "`search_sections` / `get_document_toc` / `get_section_content`."
                f"{footer}"
            )

        budget = pack_char_budget if pack_char_budget is not None else 20000
        blocks: list[str] = [
            f"## Evidence Pack — {len(self.sections)} section(s) selected by decision-model navigation"
        ]
        remaining = budget
        by_score = sorted(self.sections, key=lambda s: s.score, reverse=True)
        included: set[str] = set()
        for s in by_score:
            # All-or-nothing: a silently truncated table is worse than a pointer.
            if len(s.text) <= remaining:
                included.add(s.section_id)
                remaining -= len(s.text)
        for s in self.sections:
            header = (
                f"### [{s.section_id}] {s.title or ''} "
                f"(document: {s.document_name or s.document_id}, score {s.score:.3f})"
            )
            if s.section_id in included:
                blocks.append(f"{header}\n\n{s.text}")
            else:
                blocks.append(
                    f"{header}\n\n[text omitted: {len(s.text)} chars over pack budget — "
                    f"fetch with get_section_content('{s.section_id}') if needed]"
                )
        return "\n\n".join(blocks) + footer


class DecisionNavigator:
    """Beam-search document/section navigation powered by a decision model.

    Per level, all candidates sharing a parent are scored in requests sharing
    the same ``state`` text (query + hints + ``domain_context``) — the paper's
    shared-context joint scoring. Chunking introduced by ``batch_invoke`` when a
    level exceeds the provider's ``max_questions`` stays within one parent's
    candidate list, so per-candidate probabilities remain comparable.
    """

    def __init__(
        self,
        config: DecisionNavigatorConfig | None = None,
        *,
        model: BaseDecisionModel | None = None,
        db_path: str | None = None,
    ) -> None:
        self.config = config or DecisionNavigatorConfig()
        self.model = model or get_decision_model(self.config.decision_model)
        self.db_path = db_path

    # -- backend -------------------------------------------------------------

    def _backend(self) -> KgBackend:
        """Return the (thread-local, connection-cached) graph backend."""
        return _connect(self.db_path) if self.db_path else KuzuBackend()

    # -- state composition ---------------------------------------------------

    def _build_state(self, query: str, hints: str | None = None) -> str:
        """Compose the shared decision context: query, refinement hints, domain knowledge."""
        parts = [f"Question to answer: {query}"]
        if hints:
            parts.append(f"A previous navigation pass found these gaps; weight them higher: {hints}")
        if self.config.domain_context:
            parts.append(f"Domain knowledge for judging relevance:\n{self.config.domain_context}")
        return "\n\n".join(parts)

    # -- scoring primitive ---------------------------------------------------

    def _score_candidates(
        self,
        state: str,
        candidates: list[Candidate],
        trace: NavigationTrace,
        *,
        extra_questions: dict[str, Noul] | None = None,
    ) -> tuple[list[tuple[Candidate, float]], dict[str, float]]:
        """Score every candidate of one level jointly (chunked automatically).

        Returns the candidates ranked by probability plus the raw score map
        (including any ``extra_questions`` answers, which are not candidates)."""
        if not candidates:
            return [], {}
        questions: dict[str, Noul] = {
            c.id: Noul(instructions=f"Is '{c.label}' likely to contain information needed to answer? {c.hint}".strip())
            for c in candidates
        }
        if extra_questions:
            questions.update(extra_questions)
        response = self.model.batch_invoke({"state": state, "questions": questions})
        trace.merge_usage(response)
        scores: dict[str, float] = {}
        for q_id, answer in response.answers.items():
            if isinstance(answer, NoulAnswer):
                scores[q_id] = answer.noul
        ranked = sorted((c for c in candidates if c.id in scores), key=lambda c: scores[c.id], reverse=True)
        return [(c, scores[c.id]) for c in ranked], scores

    # -- candidate assembly --------------------------------------------------

    @staticmethod
    def _compose_hint(
        *,
        edition: str | None = None,
        description: str | None = None,
        summary: str | None = None,
    ) -> str:
        """Compose the routing text the decision model scores for one candidate.

        Both ingestion signals are included rather than picking one: in the
        OfficeQA corpus a section ``summary`` exists for only ~21% of sections and
        averages 42 characters, while ``description`` covers ~90% at 105
        characters, so preferring the summary sends the model *less* text than is
        available. The description leads, the summary is appended when it adds
        text, and the edition (bulletin filename) is always included because
        period-specific questions hinge on it.
        """
        parts: list[str] = []
        if edition:
            parts.append(f"From {edition}.")
        desc = (description or "").strip()
        summ = (summary or "").strip()
        if desc:
            parts.append(desc)
        if summ and summ not in desc:
            parts.append(summ)
        return " ".join(parts)[:_HINT_CHARS]

    @classmethod
    def _candidate_from_doc(cls, doc: dict[str, Any]) -> Candidate:
        return Candidate(
            id=doc["content_hash"],
            label=doc.get("filename") or doc["content_hash"],
            hint=cls._compose_hint(description=doc.get("description"), summary=doc.get("summary")),
        )

    @classmethod
    def _candidate_from_section(cls, row: dict[str, Any], document_id: str, doc_label: str | None = None) -> Candidate:
        return Candidate(
            id=row["section_id"],
            label=row.get("title") or row["section_id"],
            hint=cls._compose_hint(
                edition=doc_label,
                description=row.get("description"),
                summary=row.get("summary"),
            ),
            parent_id=document_id,
        )

    @staticmethod
    def _top_level_sections(toc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the root-level (non-synthetic) section rows of a document TOC.

        Real TOCs hang the level-1 sections off the synthetic level-0 root section
        (whose id is ``{hash}::0``), so a section counts as top-level when its parent
        is absent OR is that synthetic root.
        """
        root_ids = {row["section_id"] for row in toc_rows if int(row.get("level") or 0) == 0}
        return [
            row
            for row in toc_rows
            if int(row.get("level") or 0) >= 1
            and (not row.get("parent_section_id") or row.get("parent_section_id") in root_ids)
        ]

    def _apply_prefilter(
        self,
        backend: KgBackend,
        query: str,
        candidates: list[Candidate],
        *,
        folder_id: str | None = None,
        document_id: str | None = None,
    ) -> list[Candidate]:
        """Cap the candidate list, optionally using hybrid retrieval to pick the kept subset."""
        if len(candidates) <= self.config.candidate_cap:
            return candidates
        if self.config.prefilter != "hybrid":
            logger.warning(
                "Candidate list ({}) exceeds candidate_cap ({}) with prefilter=off — truncating lexically.",
                len(candidates),
                self.config.candidate_cap,
            )
            return candidates[: self.config.candidate_cap]
        hits = search_sections(
            backend, query, limit=self.config.candidate_cap, folder_id=folder_id, document_id=document_id
        )
        keyed = {c.id: c for c in candidates}
        prefiltered: list[Candidate] = []
        for hit in hits:
            key = hit.get("content_hash") if document_id is None else hit.get("section_id")
            candidate = keyed.get(key or "")
            if candidate and all(p.id != candidate.id for p in prefiltered):
                prefiltered.append(candidate)
        for candidate in candidates:
            if len(prefiltered) >= self.config.candidate_cap:
                break
            if all(p.id != candidate.id for p in prefiltered):
                prefiltered.append(candidate)
        return prefiltered

    # -- level primitives ------------------------------------------------------

    def rank_documents(
        self,
        query: str,
        folder_id: str | None = None,
        *,
        hints: str | None = None,
        trace: NavigationTrace | None = None,
    ) -> list[tuple[Candidate, float]]:
        """L1 — score every document in scope, return the top ``beam_width``.

        With ``answerable_gate`` enabled, a corpus-level answerability Noul is
        answered in the same shared-context request and flags ``trace.unanswerable``.
        """
        trace = trace or NavigationTrace()
        backend = self._backend()
        docs = list_documents(backend, folder_id=folder_id)
        candidates = [self._candidate_from_doc(d) for d in docs]
        candidates = self._apply_prefilter(backend, query, candidates, folder_id=folder_id)

        extra: dict[str, Noul] | None = None
        if self.config.answerable_gate:
            extra = {
                _ANSWERABLE_QUESTION_ID: Noul(
                    instructions="Can this question plausibly be answered from the documents described above?"
                )
            }
        ranked, all_scores = self._score_candidates(
            self._build_state(query, hints), candidates, trace, extra_questions=extra
        )

        if extra:
            answerable = all_scores.get(_ANSWERABLE_QUESTION_ID)
            if answerable is not None and answerable < self.config.min_confidence:
                trace.unanswerable = True
        top = ranked[: self.config.beam_width]
        if ranked:
            trace.levels.append(
                LevelTrace(
                    level=1,
                    n_candidates=len(ranked),
                    kept=[c.id for c, _ in top],
                    scores=self._top_scores(ranked),
                )
            )
        if top:
            trace.low_confidence = top[0][1] < self.config.l1_min_confidence
        return top

    def rank_sections(
        self,
        query: str,
        document_id: str,
        *,
        hints: str | None = None,
        r: int | None = None,
        trace: NavigationTrace | None = None,
        doc_label: str | None = None,
    ) -> tuple[list[tuple[Candidate, float]], list[dict[str, Any]]]:
        """L2 — score a document's top-level sections; return (top-r ranked, its TOC rows)."""
        trace = trace or NavigationTrace()
        backend = self._backend()
        toc_rows = get_document_toc(backend, document_id)
        candidates = [
            self._candidate_from_section(row, document_id, doc_label) for row in self._top_level_sections(toc_rows)
        ]
        candidates = self._apply_prefilter(backend, query, candidates, document_id=document_id)
        ranked, _scores = self._score_candidates(self._build_state(query, hints), candidates, trace)
        kept = ranked[: (r or self.config.top_r_sections)]
        if ranked:
            trace.levels.append(
                LevelTrace(
                    level=2,
                    n_candidates=len(ranked),
                    kept=[c.id for c, _ in kept],
                    scores=self._top_scores(ranked),
                )
            )
            if kept[0][1] < self.config.l2_min_confidence:
                trace.low_confidence = True
        return kept, toc_rows

    def _top_scores(self, ranked: list[tuple[Candidate, float]]) -> dict[str, float]:
        """Return the top ``trace_top_k`` candidate probabilities, for offline tuning."""
        return {c.id: round(p, 4) for c, p in ranked[: self.config.trace_top_k]}

    def _descend(
        self,
        state: str,
        toc_rows: list[dict[str, Any]],
        section: Candidate,
        score: float,
        depth: int,
        trace: NavigationTrace,
        doc_label: str | None = None,
    ) -> tuple[Candidate, float]:
        """Descend into a selected section's subtree while the depth budget remains."""
        by_parent: dict[str | None, list[dict[str, Any]]] = {}
        for row in toc_rows:
            by_parent.setdefault(row.get("parent_section_id"), []).append(row)
        current, current_score = section, score
        for _ in range(depth - 1):
            children = by_parent.get(current.id, [])
            if not children:
                break
            child_candidates = [
                self._candidate_from_section(row, current.id, doc_label)
                for row in sorted(children, key=lambda r: r.get("sequence") or 0)
            ]
            ranked, _scores = self._score_candidates(state, child_candidates, trace)
            if not ranked or ranked[0][1] < self.config.descent_min_confidence:
                # No child stands out: keep the parent rather than descend into noise.
                break
            trace.levels.append(
                LevelTrace(
                    level=3,
                    n_candidates=len(ranked),
                    kept=[ranked[0][0].id],
                    scores=self._top_scores(ranked),
                )
            )
            current, current_score = ranked[0]
        return current, current_score

    @staticmethod
    def _ancestor_ids(toc_rows: list[dict[str, Any]], section_id: str) -> list[str]:
        """Return the ancestor chain of *section_id* (root-first, excluding itself and the synthetic root)."""
        by_id = {row["section_id"]: row for row in toc_rows}
        chain: list[str] = []
        current = by_id.get(section_id)
        while current is not None and current.get("parent_section_id"):
            parent_id = current["parent_section_id"]
            parent = by_id.get(parent_id)
            if parent is not None and int(parent.get("level") or 0) == 0:
                break  # synthetic root: a container, not fetchable content
            chain.append(parent_id)
            current = parent
        chain.reverse()
        return chain

    # -- navigation ------------------------------------------------------------

    def navigate(
        self,
        query: str,
        folder_id: str | None = None,
        *,
        document_id: str | None = None,
        hints: str | None = None,
    ) -> EvidencePack:
        """Beam-search the hierarchy and return the selected sections' content.

        Args:
            query: The user question.
            folder_id: Scope of the search (None = every ingested document).
            document_id: When given, skip L1 and rank this document's sections only
                (single-document corpora, e.g. MMLongBench-Doc).
            hints: Optional refinement hints from a previous navigation pass.

        Returns:
            An :class:`EvidencePack` with the fetched section texts, the
            low-confidence / unanswerable flags and the full navigation trace.
        """
        start = time.perf_counter()
        trace = NavigationTrace()
        state = self._build_state(query, hints)
        backend = self._backend()

        # L1 — document beam (skipped in single-document scope).
        doc_ids: list[tuple[str, str | None]] = []
        if document_id:
            doc = get_document(backend, document_id)
            if doc:
                doc_ids = [(doc["content_hash"], doc.get("filename"))]
        else:
            top = self.rank_documents(query, folder_id, hints=hints, trace=trace)
            doc_ids = [(c.id, c.label) for c, _ in top]
            if trace.unanswerable:
                trace.elapsed_ms = (time.perf_counter() - start) * 1000
                return EvidencePack(query=query, unanswerable=True, trace=trace)

        # L2 — section ranking per kept document (optionally parallel across
        # documents; decision models are stateless per request), then subtree descent.
        def _worker(doc: tuple[str, str | None]) -> tuple[list[SectionEvidence], NavigationTrace]:
            # Worker threads each get their own thread-local connection (~120 MB of
            # native working set). ThreadPoolExecutor recreates its threads per
            # navigate() call, so without this release every navigation leaked the
            # pool's connections and the buffer manager eventually failed.
            try:
                return self._select_sections_for_doc(query, doc[0], doc[1], state, hints)
            finally:
                if self.db_path:
                    release_connection(self.db_path)

        if len(doc_ids) > 1 and self.config.l2_workers > 1:
            with ThreadPoolExecutor(max_workers=min(self.config.l2_workers, len(doc_ids))) as pool:
                results = list(pool.map(_worker, doc_ids))
        else:
            results = [
                self._select_sections_for_doc(query, doc_id, doc_name, state, hints) for doc_id, doc_name in doc_ids
            ]
        selected: list[SectionEvidence] = []
        for sections, doc_trace in results:
            trace.merge(doc_trace)
            selected.extend(sections)

        trace.elapsed_ms = (time.perf_counter() - start) * 1000
        return EvidencePack(query=query, sections=selected, low_confidence=trace.low_confidence, trace=trace)

    def _select_sections_for_doc(
        self, query: str, doc_id: str, doc_name: str | None, state: str, hints: str | None
    ) -> tuple[list[SectionEvidence], NavigationTrace]:
        """L2 for one document: rank sections, descend the best branches, fetch content."""
        doc_trace = NavigationTrace()
        backend = self._backend()
        kept, toc_rows = self.rank_sections(query, doc_id, hints=hints, trace=doc_trace, doc_label=doc_name)
        selected: list[SectionEvidence] = []
        seen_sections: set[str] = set()
        for section, score in kept:
            leaf, leaf_score = self._descend(
                state, toc_rows, section, score, self.config.max_depth, doc_trace, doc_label=doc_name
            )
            fetch_ids = [sid for sid in (*self._ancestor_ids(toc_rows, leaf.id), leaf.id) if sid not in seen_sections]
            seen_sections.update(fetch_ids)
            for row in get_section_content(backend, fetch_ids):
                selected.append(
                    SectionEvidence(
                        section_id=row["section_id"],
                        title=row.get("title"),
                        document_id=row.get("content_hash") or doc_id,
                        document_name=doc_name,
                        score=leaf_score if row["section_id"] == leaf.id else 0.0,
                        text=row.get("text") or "",
                    )
                )
        return selected, doc_trace


def extract_navigation_trace(text: str | None) -> dict[str, Any] | None:
    """Extract the ``[[navigation-trace]]`` JSON footer from a tool result (None when absent)."""
    if not text or TRACE_MARKER not in text:
        return None
    tail = text.rsplit(TRACE_MARKER, 1)[1].strip()
    try:
        data = json.loads(tail.splitlines()[0])
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, IndexError):
        logger.warning("Malformed navigation trace footer could not be parsed")
        return None


def merge_navigation_traces(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Merge per-call navigation trace blocks (multiple ``navigate_folder`` calls) into one run block."""
    if not blocks:
        return None
    merged: dict[str, Any] = {
        "dm_calls": sum(int(b.get("dm_calls") or 0) for b in blocks),
        "dm_input_tokens": sum(int(b.get("dm_input_tokens") or 0) for b in blocks),
        "dm_output_tokens": sum(int(b.get("dm_output_tokens") or 0) for b in blocks),
        "dm_cost": round(sum(float(b.get("dm_cost") or 0.0) for b in blocks), 6),
        "navigations": len(blocks),
        "low_confidence": any(bool(b.get("low_confidence")) for b in blocks),
        "unanswerable": any(bool(b.get("unanswerable")) for b in blocks),
        "elapsed_ms": round(sum(float(b.get("elapsed_ms") or 0.0) for b in blocks), 1),
        "levels": [lvl for b in blocks for lvl in (b.get("levels") or [])],
    }
    if not merged["dm_cost"]:
        merged["dm_cost"] = None
    return merged
