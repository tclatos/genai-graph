"""Unit tests for the decision-model navigator (offline, scripted decision model)."""

from __future__ import annotations

from typing import Any

import pytest
from genai_tk.core.decision_models import (
    BaseDecisionModel,
    ClassifierRequest,
    ClassifierResponse,
    NoulAnswer,
    Usage,
)
from langchain_core.runnables import RunnableConfig
from pydantic import Field

import genai_graph.agent.decision_navigation as decision_navigation
import genai_graph.agent.decision_navigator as dn
from genai_graph.agent.decision_navigator import (
    DecisionNavigator,
    DecisionNavigatorConfig,
    EvidencePack,
    NavigationTrace,
    SectionEvidence,
    extract_navigation_trace,
    merge_navigation_traces,
)


class ScriptedDecisionModel(BaseDecisionModel):
    """Deterministic stub: per-question-id probability map + request capture."""

    model: str = "scripted@fake"
    probabilities: dict[str, float] = Field(default_factory=dict)
    default_probability: float = 0.9
    requests: list[dict[str, Any]] = Field(default_factory=list)

    def invoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        self.validate_request(request)
        self.requests.append(
            {
                "state": str(request.state),
                "questions": {q_id: str(q.instructions) for q_id, q in request.questions.items()},
            }
        )
        answers = {
            q_id: NoulAnswer(noul=self.probabilities.get(q_id, self.default_probability)) for q_id in request.questions
        }
        return ClassifierResponse(
            model=self.model,
            answers=answers,
            usage=Usage(input_tokens=10 * len(request.questions), output_tokens=0, cost=0.01),
            provider="fake",
        )

    async def ainvoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        return self.invoke(input, config=config, **kwargs)


def _stub_graph(
    monkeypatch: pytest.MonkeyPatch,
    *,
    docs: list[dict[str, Any]] | None = None,
    tocs: dict[str, list[dict[str, Any]]] | None = None,
    sections: dict[str, str] | None = None,
) -> None:
    """Monkeypatch the graph row-fetch functions used by the navigator."""
    monkeypatch.setattr(dn, "list_documents", lambda backend, folder_id=None, **kw: docs or [])
    monkeypatch.setattr(dn, "get_document_toc", lambda backend, document_id, **kw: (tocs or {}).get(document_id, []))
    monkeypatch.setattr(
        dn, "get_document", lambda backend, document_id, **kw: {"content_hash": document_id, "filename": document_id}
    )
    monkeypatch.setattr(dn, "search_sections", lambda backend, query, **kw: [])

    def _content(backend: Any, section_ids: list[str], **kw: Any) -> list[dict[str, Any]]:
        store = sections or {}
        return [
            {
                "section_id": sid,
                "title": sid,
                "content_hash": sid.split("::")[0],
                "text": store.get(sid, f"text of {sid}"),
            }
            for sid in section_ids
        ]

    monkeypatch.setattr(dn, "get_section_content", _content)


DOC_ROWS = [
    {
        "content_hash": f"doc_{i}",
        "filename": f"bulletin_{1990 + i}",
        "description": "Treasury bulletin",
        "summary": None,
    }
    for i in range(4)
]


def _toc(doc_id: str, n_sections: int = 2, with_children: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(n_sections):
        sid = f"{doc_id}::{i + 1}"
        rows.append(
            {
                "section_id": sid,
                "parent_section_id": None,
                "title": f"Section {i + 1} of {doc_id}",
                "level": 1,
                "sequence": i + 1,
                "description": f"describes {sid}",
                "summary": None,
            }
        )
        if with_children:
            rows.append(
                {
                    "section_id": f"{sid}::1",
                    "parent_section_id": sid,
                    "title": f"Child of {sid}",
                    "level": 2,
                    "sequence": 1,
                    "description": f"child detail of {sid}",
                    "summary": None,
                }
            )
    return rows


def test_beam_selection_keeps_only_top_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_graph(
        monkeypatch,
        docs=DOC_ROWS,
        tocs={d["content_hash"]: _toc(d["content_hash"]) for d in DOC_ROWS},
        sections={},
    )
    model = ScriptedDecisionModel(probabilities={"doc_2": 0.95, "doc_0": 0.80, "doc_1": 0.30, "doc_3": 0.10})
    navigator = DecisionNavigator(DecisionNavigatorConfig(beam_width=2, top_r_sections=1), model=model)
    pack = navigator.navigate("Which bulletin raised the debt limit?")
    selected_docs = {s.document_id for s in pack.sections}
    assert selected_docs == {"doc_2", "doc_0"}
    # Sections carry their own (section-level) joint score; the first section is from the best document.
    assert pack.sections[0].document_id == "doc_2"
    assert pack.sections[0].score == pytest.approx(0.9)


def test_low_confidence_flag_when_all_scores_below_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_graph(
        monkeypatch,
        docs=DOC_ROWS,
        tocs={d["content_hash"]: _toc(d["content_hash"]) for d in DOC_ROWS},
        sections={},
    )
    model = ScriptedDecisionModel(default_probability=0.2)
    navigator = DecisionNavigator(DecisionNavigatorConfig(min_confidence=0.5), model=model)
    pack = navigator.navigate("Anything about gold reserves?")
    assert pack.low_confidence is True
    assert pack.sections  # evidence still returned; the flag drives the fallback policy


def test_chunked_level_scoring_scores_all_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_graph(
        monkeypatch,
        docs=DOC_ROWS,
        tocs={d["content_hash"]: _toc(d["content_hash"], n_sections=3) for d in DOC_ROWS},
        sections={},
    )
    model = ScriptedDecisionModel(max_questions=2)  # batch_invoke chunks 4 docs into 2 sub-requests
    navigator = DecisionNavigator(DecisionNavigatorConfig(beam_width=2, top_r_sections=1), model=model)
    trace = NavigationTrace()
    ranked = navigator.rank_documents("q", trace=trace)
    assert len(ranked) == 2
    assert trace.dm_calls == 1  # one shared-context batch per level (chunks are internal)
    assert trace.dm_input_tokens == 10 * 4  # every candidate scored exactly once across chunks


def test_domain_context_and_hints_compose_shared_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_graph(
        monkeypatch,
        docs=DOC_ROWS,
        tocs={d["content_hash"]: _toc(d["content_hash"]) for d in DOC_ROWS},
        sections={},
    )
    model = ScriptedDecisionModel()
    config = DecisionNavigatorConfig(domain_context="FFO = Federal Fiscal Operations")
    navigator = DecisionNavigator(config, model=model)
    navigator.navigate("What was the deficit?", hints="need December 1953 table")
    states = [r["state"] for r in model.requests]
    assert any("FFO = Federal Fiscal Operations" in state for state in states)
    assert any("need December 1953 table" in state for state in states)
    assert all(state.startswith("Question to answer: What was the deficit?") for state in states)


def test_answerable_gate_short_circuits_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_graph(monkeypatch, docs=DOC_ROWS, tocs={}, sections={})
    model = ScriptedDecisionModel(probabilities={dn._ANSWERABLE_QUESTION_ID: 0.1}, default_probability=0.95)
    navigator = DecisionNavigator(DecisionNavigatorConfig(answerable_gate=True), model=model)
    pack = navigator.navigate("What is the meaning of life?")
    assert pack.unanswerable is True
    assert pack.sections == []
    assert model.requests  # L1 was scored with the gate question included


def test_answerable_gate_passes_when_probability_high(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_graph(
        monkeypatch,
        docs=DOC_ROWS,
        tocs={d["content_hash"]: _toc(d["content_hash"]) for d in DOC_ROWS},
        sections={},
    )
    model = ScriptedDecisionModel(probabilities={dn._ANSWERABLE_QUESTION_ID: 0.99}, default_probability=0.95)
    navigator = DecisionNavigator(DecisionNavigatorConfig(answerable_gate=True), model=model)
    pack = navigator.navigate("Deficit for fiscal year 2011?")
    assert pack.unanswerable is False
    assert pack.sections


def test_descent_fetches_leaf_and_ancestors(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = DOC_ROWS[0]["content_hash"]
    _stub_graph(
        monkeypatch,
        docs=DOC_ROWS[:1],
        tocs={doc_id: _toc(doc_id, n_sections=1, with_children=True)},
        sections={},
    )
    top_section = f"{doc_id}::1"
    child = f"{top_section}::1"
    model = ScriptedDecisionModel(probabilities={child: 0.97, top_section: 0.60})
    navigator = DecisionNavigator(DecisionNavigatorConfig(beam_width=1, top_r_sections=1, max_depth=2), model=model)
    pack = navigator.navigate("Narrow question")
    fetched = [s.section_id for s in pack.sections]
    assert child in fetched  # best child selected by descent
    assert top_section in fetched  # ancestor chain fetched alongside the leaf
    leaf = next(s for s in pack.sections if s.section_id == child)
    assert leaf.score == pytest.approx(0.97)


def test_question_instructions_mention_candidate_label(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_graph(monkeypatch, docs=DOC_ROWS, tocs={}, sections={})
    model = ScriptedDecisionModel()
    navigator = DecisionNavigator(DecisionNavigatorConfig(), model=model)
    navigator.rank_documents("q")
    questions = model.requests[0]["questions"]
    assert "doc_2" in questions
    assert "bulletin_1992" in questions["doc_2"]
    assert "Treasury bulletin" in questions["doc_2"]


def test_joint_scoring_shares_one_state_per_level(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_graph(monkeypatch, docs=DOC_ROWS, tocs={}, sections={})
    model = ScriptedDecisionModel()
    navigator = DecisionNavigator(DecisionNavigatorConfig(), model=model)
    navigator.rank_documents("q")
    assert len(model.requests) == 1  # all candidates answered against ONE shared context
    assert len(model.requests[0]["questions"]) == 4


def test_prefilter_hybrid_prefers_retrieval_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = [{"content_hash": f"doc_{i}", "filename": f"b_{i}", "description": "d", "summary": None} for i in range(10)]
    _stub_graph(monkeypatch, docs=docs, tocs={}, sections={})
    monkeypatch.setattr(
        dn,
        "search_sections",
        lambda backend, query, **kw: [{"content_hash": "doc_3"}, {"content_hash": "doc_7"}],
    )
    model = ScriptedDecisionModel()
    navigator = DecisionNavigator(
        DecisionNavigatorConfig(beam_width=4, candidate_cap=4, prefilter="hybrid"), model=model
    )
    ranked = navigator.rank_documents("q")
    kept_ids = [c.id for c, _ in ranked]
    assert kept_ids[0] in {"doc_3", "doc_7"}  # retrieval hits are scored (first in candidate order)
    assert len(kept_ids) == 4  # capped at candidate_cap


def test_pack_budget_truncates_lowest_scores_first() -> None:
    sections = [
        dn.SectionEvidence(section_id=f"d::{i}", title=f"s{i}", document_id="d", score=1.0 - i / 10, text="x" * 500)
        for i in range(6)
    ]
    pack = EvidencePack(query="q", sections=sections, trace=NavigationTrace(dm_calls=1))
    text = pack.render(pack_char_budget=1200)  # fits 2 of the 6 sections fully
    assert "x" * 500 in text  # best sections fully included
    assert text.count("text omitted") == 4  # the rest become pointers
    worst = sections[-1].section_id  # lowest score
    pointer = "get_section_content('" + worst + "')"
    assert pointer in text  # pointers name the fetch call for the worst section
    assert text.index("x" * 500) < text.index("text omitted")  # best sections first


def test_parallel_l2_workers_merge_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_graph(
        monkeypatch,
        docs=DOC_ROWS,
        tocs={d["content_hash"]: _toc(d["content_hash"]) for d in DOC_ROWS},
        sections={},
    )
    model = ScriptedDecisionModel()
    navigator = DecisionNavigator(DecisionNavigatorConfig(beam_width=3, top_r_sections=1, l2_workers=3), model=model)
    pack = navigator.navigate("q")
    assert pack.trace.dm_calls == 1 + 3  # L1 + one L2 batch per kept document
    assert pack.trace.dm_input_tokens == 10 * (4 + 3 * 2)  # 4 docs at L1 + 2 sections per doc at L2
    assert len([lvl for lvl in pack.trace.levels if lvl.level == 2]) == 3


def test_trace_footer_extraction_and_merge() -> None:
    trace = NavigationTrace(dm_calls=2, dm_input_tokens=120, dm_output_tokens=0, dm_cost=0.02, low_confidence=True)
    pack = EvidencePack(query="q", trace=trace)
    extracted = extract_navigation_trace(pack.render())
    assert extracted is not None
    assert extracted["dm_calls"] == 2
    assert extracted["low_confidence"] is True

    merged = merge_navigation_traces([extracted, extracted])
    assert merged is not None
    assert merged["dm_calls"] == 4
    assert merged["dm_input_tokens"] == 240
    assert merged["navigations"] == 2
    assert merged["low_confidence"] is True
    assert extract_navigation_trace("no footer here") is None
    assert merge_navigation_traces([]) is None


def _fake_navigator(monkeypatch: pytest.MonkeyPatch, pack: EvidencePack) -> None:
    class _FakeNavigator:
        config = DecisionNavigatorConfig()

        def __init__(self, config: DecisionNavigatorConfig, *, db_path: str | None = None) -> None: ...

        def navigate(self, query: str, folder_id: str | None = None, *, hints: str | None = None) -> EvidencePack:
            return pack

    monkeypatch.setattr(decision_navigation, "DecisionNavigator", _FakeNavigator)
    monkeypatch.setattr(decision_navigation, "resolve_db_path", lambda db_path, profile: "/tmp/fake.db")


def test_tool_factory_kwargs_flow_into_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    context_file = tmp_path / "decision-routing.md"
    context_file.write_text("USCC = Currency and Coin", encoding="utf-8")
    captured: dict[str, Any] = {}

    class _FakeNavigator:
        def __init__(self, config: DecisionNavigatorConfig, *, db_path: str | None = None) -> None:
            self.config = config
            captured["config"] = config
            captured["db_path"] = db_path

        def navigate(self, query: str, folder_id: str | None = None, *, hints: str | None = None) -> EvidencePack:
            return EvidencePack(query=query, trace=NavigationTrace(dm_calls=1))

    monkeypatch.setattr(decision_navigation, "DecisionNavigator", _FakeNavigator)
    monkeypatch.setattr(decision_navigation, "resolve_db_path", lambda db_path, profile: "/tmp/fake.db")

    tools = decision_navigation.create_decision_navigation_tools(
        profile="officeqa",
        decision_model="fast",
        beam_width=7,
        top_r_sections=2,
        domain_context_file=str(context_file),
        unknown_future_kwarg="ignored",
    )
    config = captured["config"]
    assert config.decision_model == "fast"
    assert config.beam_width == 7
    assert config.top_r_sections == 2
    assert config.domain_context == "USCC = Currency and Coin"
    assert captured["db_path"] == "/tmp/fake.db"
    names = [t.name for t in tools]
    assert names[0] == "navigate_folder"
    assert "get_document_toc" in names and "search_sections" in names  # fallback tools retained


def test_tool_factory_navigate_folder_renders_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    pack = EvidencePack(
        query="What was the deficit?",
        sections=[
            SectionEvidence(
                section_id="h::1", title="T", document_id="h", document_name="d.pdf", score=0.9, text="body"
            )
        ],
        trace=NavigationTrace(dm_calls=3, dm_input_tokens=60, dm_output_tokens=0, dm_cost=0.003),
    )
    _fake_navigator(monkeypatch, pack)
    tools = decision_navigation.create_decision_navigation_tools()
    navigate_tool = next(t for t in tools if t.name == "navigate_folder")
    result = str(navigate_tool.invoke({"query": "What was the deficit?"}))
    assert "[h::1]" in result and "body" in result
    extracted = extract_navigation_trace(result)
    assert extracted is not None and extracted["dm_calls"] == 3
