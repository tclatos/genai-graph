# Decision-Model Navigation for the Document Graph (DecisionRAG Study)

**Status**: implemented (phases 1–4) — first subset run executed 2026-11-02, results & findings in §6
**Motivated by**: [Retriever-Free Retrieval: Navigating Documents with Decision Models](https://arxiv.org/abs/2610.35868) (Wan, Jünger et al., Cornell, Oct 2026)
**Building blocks**: `genai-tk` decision models (`get_decision_model`), genai-graph Document Graph tools, `genai_graph.bench` engine
**Target benchmark**: OfficeQA Pro (subset), sibling project `~/prj/officeqa`

---

## 1. Problem Statement

The current Document Graph agent (`type: deep`, profile `docgraph` / officeqa `default`) navigates a
corpus with an **LLM in the loop at every step**: `get_folder_toc` → `get_document_toc` →
`get_section_content` / `search_sections`. Every navigation decision is an autoregressive LLM
completion that regenerates the whole conversation context. On OfficeQA Pro this costs on average
**6.2 tool calls and ~248.5k input tokens per question** (glm_5.3_Flash baseline,
see [benchmarks_financebench_officeqa.md](../benchmarks_financebench_officeqa.md)), and failed runs
degenerate into search loops consuming >4M tokens.

The vast majority of those tokens buy *navigation* — deciding which document to open and which
section to read — not *answer synthesis*. This is exactly the inefficiency the DecisionRAG paper
attacks: replace token-generating agentic navigation with **decision models** (System-One models)
that jointly score all candidate branches in a **single forward pass, generating zero tokens**.

### 1.1 What the paper shows

DecisionRAG navigates a corpus hierarchy (topics → documents → passages) with a decision model.
At each level, every candidate branch becomes one yes/no question ("is this branch relevant to the
query?"); all questions of a level are answered **jointly in one forward pass against a shared
context** (the query), yielding comparable per-candidate probabilities P(yes). Only the top-k
branches are expanded; pruned branches are never visited.

Key results:

| Finding | Number |
|---|---|
| Accuracy vs dense retrieval + 8B reranker (no task training) | +13.8 R@1 (MultiDoc2Dial), +17.4 Hit@1 (NQ) |
| Joint (shared-context) scoring vs independent scoring | up to +7.7 accuracy points |
| Latency vs LLM generating navigation decisions over the same hierarchy | **34–145× lower** at comparable accuracy |
| Retrieval latency / cost per query (2.68M-paragraph corpus) | 0.7–1.8 s / $0.2–2.3 cents |

The mechanism transfers directly to our setting because our graph already stores the two things the
method needs: a **hierarchy** (`Folder ─CONTAINS→ Document ─HAS_SECTION→ MarkdownSection ─HAS_SUBSECTION→ …`)
and **per-node routing text** (one-line `description`, optional `summary`, generated at ingestion —
these play the role of the paper's branch descriptions/summaries).

### 1.2 Why this is a good fit for genai-graph

- Our navigation signal (`description` / `summary` per section) is exactly what decision models
  score; ingestion already pays for it (and the OfficeQA study showed summaries cut tokens 59%).
- `genai-tk` already ships a production decision-model stack: `get_decision_model()` factory with
  providers (Clef/Clef-flash ≤64 q/request, TypeSafe Jev ≤32, Perplexity Decider ≤128, OpenAI
  GPT-6 decisions ≤200, **Laya local sub-35 ms**, chat-model adapter, `fake` for tests), typed
  primitives (`Noul` boolean, `Choice` categorical, `Score` ordinal), shared-state multi-question
  `invoke`, and auto-chunking `batch_invoke` with usage/cost accounting.
- The benchmark engine (`genai_graph.bench`) already records tool calls, token usage
  (`runs.jsonl`) and judge verdicts (`scores.jsonl`, `scores_summary.json`), so a paired
  A/B comparison is mostly configuration, plus a few new navigation metrics.

---

## 2. Alternatives Analysis

Five architectures were considered for bringing decision models into Document Graph navigation.
They are not mutually exclusive; A, B and C stack naturally.

### Option A — Decision-model *middleware* (prune observations, keep the LLM agent)

A `DecisionRoutingMiddleware` intercepts `get_folder_toc` / `get_document_toc` tool results and
re-ranks/prunes the candidate list with the decision model **before** the observation reaches the
LLM (same interception pattern as the existing `MapBeforeSearchMiddleware` and
`DeduplicateToolCallsMiddleware`).

- ✅ Minimal change, composable, easy ablation (toggle the middleware); immediate token savings
  on large TOCs (OfficeQA bulletins have hundreds of sections).
- ❌ The LLM still *drives* navigation — one LLM completion per step. The 6.2 tool calls and the
  bulk of the 248k input tokens remain. Only shrinks observations; does not meet the
  "reduce LLM calls" goal.

### Option B — Deterministic *Decision Navigator* + one-shot synthesis (paper-faithful, recommended core)

A `DecisionNavigator` component replaces the LLM navigation loop entirely:

1. **L1 — pick documents**: one `Noul` question per document in the folder (candidate text =
   `filename` + `description`), all in one shared-context request → keep top-k documents (beam).
2. **L2 — pick sections**: per kept document, one `Noul` question per top-level section
   (candidate text = `title` + `description`/`summary`) → keep top-r sections.
3. **L3+ — optional descent**: while a section has subsections and depth < `max_depth`, repeat
   within the section's subtree.
4. **Fetch** selected sections (`get_section_content`), assemble an **evidence pack**.
5. Hand evidence pack + question to a small **synthesis agent** — the *only* LLM usage — which
   answers (with `python_interpreter` for arithmetic, `web_search` for external context, per
   current OfficeQA discipline). LLM calls per question: **~1–3 instead of ~6–10+**.

- ✅ Maximally reduces LLM calls/tokens; deterministic and replayable navigation; retrieval
  quality measurable in isolation (evidence hit rate vs the benchmark's gold `evidence` field);
  mirrors the paper's validated design; zero-navigation-LLM-cost.
- ❌ No iterative re-planning inside navigation; multi-part questions need beam width > 1 and
  possibly one query-decomposition step; exploratory "I'll know it when I read it" questions
  (rare in OfficeQA, common in free-form use) are weaker than an agentic loop.

### Option C — Hybrid with graceful fallback (recommended policy on top of B)

Option B as primary path; the synthesis agent keeps the **standard graph tools alongside the
navigator tools** and falls back to LLM-driven navigation when (a) no candidate clears
`min_confidence`, (b) the top score distribution is flat (uncertain), or (c) it judges the evidence
pack insufficient after ≤2 re-navigations with refined hints. Because the synthesis agent is a
deep agent, the fallback needs no second agent — it is prompt/skill discipline plus tool
availability. Bounded worst case ≈ today's agent; expected case keeps B's cost.

### Option D — Local decision model (Laya) as navigator

`laya@laya` runs in-process at sub-35 ms with zero API cost — attractive for production. Likely
lower accuracy than cloud decision models on nuanced financial phrasing; evaluate as a *variant*
of B (same code path, different `decision_model` tag), not a separate architecture.

### Option E — Retrieval-only reference point (no LLM at all)

`search_sections` (hybrid vector + BM25/RRF) already exists. A "top-m sections → synthesis" pipeline
with no TOC navigation is a useful **lower-bound baseline**: it isolates how much of the gain comes
from *hierarchical decision scoring* vs plain retrieval. Cheap to run (one config), worth having in
the comparison matrix.

### Recommendation

**Implement B** (`DecisionNavigator` + `decision` agent profile) **with C's escalation as a
configurable policy** (default on for the benchmark, off for pure cost runs), and run **E** as a
baseline row. **A** is a useful complementary middleware for the *existing* deep agent (large-TOC
token pruning) but is out of scope for the first iteration. **D** is a variant flag.

---

## 3. Proposed Architecture

### 3.1 `DecisionNavigator` (genai-graph, reusable)

New module `genai_graph/agent/decision_navigator.py`. Pydantic-configured, provider-agnostic via
`get_decision_model()`:

```python
class DecisionNavigatorConfig(BaseModel):
    decision_model: str = "fast"          # tag or 'model@provider' → get_decision_model()
    beam_width: int = 3                   # documents kept at L1
    top_r_sections: int = 6               # sections kept per document at L2 (total cap)
    min_confidence: float = 0.5           # Noul gate; below → low-confidence flag (fallback)
    max_depth: int = 2                    # section-tree levels to descend
    candidate_cap: int = 48               # max questions per level per parent
    prefilter: str = "off"                # off | hybrid  (search_sections prefilter above cap)
    include_summaries: bool = False       # use section summary instead of description
    domain_context: str | None = None     # domain routing knowledge injected into `state`
    answerable_gate: bool = False         # add a corpus-level 'answerable' Noul (MMLongBench)
```

Sketch of the level-score primitive (batch chunking delegated to `batch_invoke`, so per-provider
`max_questions` limits are respected automatically):

```python
def _score_candidates(self, state: str, candidates: list[Candidate]) -> list[tuple[Candidate, float]]:
    request = {
        "state": state,
        "questions": {
            c.id: Noul(instructions=f"Is '{c.label}' likely to contain information needed "
                                    f"to answer the question? {c.hint}")
            for c in candidates
        },
    }
    response = self.model.batch_invoke(request)
    ranked = sorted(((c, response.answers[c.id].noul) for c in candidates),
                    key=lambda t: t[1], reverse=True)
    return ranked
```

**Domain context in `state` (review remark 2)**: the domain knowledge currently in the
`officeqa-qa` skill's *Dynamic Structure Discovery & Domain Acronyms* and *Document Graph
Navigation & Revision Selection* sections (FFO/UST/FD/PDO/OFS/USCC/IFS/CM acronym expansions,
"figures are often revised in later bulletins — locate the specific month/year edition") is
**navigation** knowledge: the decision model needs it to score candidates, not just the synthesis
LLM. It is injected into the shared `state` verbatim via `domain_context` (inline text, or a file
path the factory loads — e.g. a `decision-routing.md` excerpt split out of `officeqa-qa`).
Statistical formula templates (§3–§4 of the skill) and citation rules are **synthesis** knowledge
and stay with the agent skill. Per-benchmark notes:

- **OfficeQA Pro**: `domain_context` = acronym/routing + revision-selection excerpt.
- **MMLongBench-Doc**: no domain skill exists today; questions are single-document (L1 document
  selection is skipped — the bench runner scopes `folder_id`/`document_id` to the question's
  document) and ~22.5% of gold answers are "Not answerable" → enable `answerable_gate`, which
  appends one corpus-level `Noul` ("can this question plausibly be answered from this document?")
  to the L1 request; below-threshold answerability short-circuits navigation and the synthesis
  agent is told to declare the information absent. Chart/plot questions route on section
  descriptions (which embed image descriptions) and stay answerable via the retained `query_image`.

Public surface:

- `rank_documents(query, folder_id) -> list[(document_id, p)]` — L1
- `rank_sections(query, document_id) -> list[(section_id, p)]` — L2 (+ L3 descent)
- `navigate(query, folder_id) -> EvidencePack` — beam search over levels, returns selected
  section ids + scores + a navigation trace (for logging/`runs.jsonl`)

**Comparability rule (from the paper's shared-context finding)**: all candidates sharing a parent
are scored in requests that share the same `state` text (query + `domain_context`). Chunks
introduced by `batch_invoke` are fine *within one parent's candidate list* (all chunks share the
state; we merge by probability and, for ranking stability, record chunk id in the trace).
Cross-parent comparisons happen only on probabilities of the same question wording.

**Graceful fallback (Option C)**: `navigate()` flags low-confidence beams in the `EvidencePack`;
the synthesis deep agent — which retains `get_document_toc` / `get_section_content` /
`search_sections` — then navigates LLM-driven for that question, per the profile skill's
discipline ("navigator first; graph tools on insufficient evidence").

### 3.2 New agent profile `decision`

One reusable tool factory in genai-graph (`create_decision_navigation_tools`, mirroring
`create_docgraph_tools`), then a thin profile per consuming project. In
`~/prj/officeqa/config/agents.yaml`:

```yaml
agents:
  decision:
    harness: langchain
    name: "OfficeQA Decision-Navigated Analyst"
    type: deep                     # deep: the ONLY type with the skill system (SkillsMiddleware);
                                   # react profiles ignore skill_directories — officeqa's domain
                                   # knowledge (formulas, revision rules) lives in `officeqa-qa`.
    llm: glm_5.3_Flash@openrouter
    description: "Synthesis agent over decision-model-navigated evidence packs."
    enable_planning: false         # navigation is deterministic; planning is dead weight
    system_prompt: |
      You answer OfficeQA questions from the EVIDENCE PACK fetched by `navigate_folder`
      (decision-model navigation over the Document Graph). Read the pack's sections, then
      answer: numbers first, citations `[hash::sequence]` for every claim, all arithmetic in
      `python_interpreter`. Call `navigate_folder` again with refined hints describing what is
      missing (max 2 re-navigations); only if it still returns low-confidence evidence, fall
      back to `get_document_toc` / `get_section_content` / `search_sections` directly.
    tools:
      # All navigator configuration lives HERE, as tool-factory kwargs (review remark 3):
      - factory: genai_graph.agent.decision_navigation.create_decision_navigation_tools
        decision_model: fast           # clef_flash@openrouter; variants: accurate=jev, laya, fake
        beam_width: 3
        top_r_sections: 6
        min_confidence: 0.5
        max_depth: 2
        prefilter: hybrid
        domain_context_file: ${paths.project}/skills/custom/officeqa-qa/decision-routing.md
      - officeqa.tools.calculator.create_calculator_tools
      - genai_tk.agents.tools.langchain.search_tools_factory.create_search_tool
    skill_directories: [${paths.project}/skills/custom, ${paths.project}/skills/community]
    middlewares: [ …same discipline middlewares as `default` (dedup, wrap-up, truncation)… ]
```

The profile is `type: deep` — the LangChain react engine has **no skill system**
(`AgentProfileConfig` treats `skill_directories`/`backend`/`subagents` as deep-only and ignores
them otherwise), and the `officeqa-qa` skill is required for answering discipline (statistical
formula templates, revision rules, citation format). Deep-agent overhead is trimmed with
`enable_planning: false`; navigation itself is deterministic and skill-free.

Tool surface of the new profile (deliberately smaller than the deep agent's):
`navigate_folder` (navigator; also returns `get_section_content` / `search_sections` /
`query_image` for re-reads, visual questions and LLM-driven fallback), `python_interpreter`,
`web_search`. A short `decision-navigation` skill excerpt documents the loop and the fallback
rule.

The same profile pattern drops into genai-graph's `config/agents/docgraph.yaml` as
`docgraph_decision` for interactive `cli docgraph agent` use — one profile, two projects,
navigator code owned by genai-graph.

### 3.3 Configuration & runtime wiring

- **All navigator configuration is tool-factory kwargs** (review remark 3): profile tool entries
  are `UnifiedToolSpec`/`FactoryToolSpec` dicts — every extra key becomes a constructor kwarg of
  the factory (`_load_tool_from_target` → `obj(**params)`), so `decision_model`, `beam_width`,
  `domain_context_file`, … flow straight into `create_decision_navigation_tools(db_path=None, …)`,
  which resolves `db_path` from the docgraph profile exactly like `create_docgraph_tools` does.
  **No new profile-YAML section and no change to `create_docgraph_agent`'s tool injection are
  needed** — the profile's `tools:` list already flows through `process_langchain_tools_from_config`.
- `get_decision_model(tag)` already resolves tags (`default`/`fast`/`accurate`/`fake`) from
  genai-tk config (`decision.models`), so switching the navigator's brain is a one-line change —
  including the `fake` model for deterministic unit tests and the chat-adapter for ablations.
- Bench-profile-level overrides (e.g. `decision_fast` vs `decision_accurate` variants) are separate
  bench profiles whose `agent_profile` points at sibling profiles differing only in the factory
  kwargs — no code branching.
- Decision-model usage is already returned in `ClassifierResponse.usage` (tokens + cost) — record
  it per question in `runs.jsonl` (new `navigation` block: dm_calls, dm_tokens, dm_cost,
  selected_sections, scores, low_confidence flag).

---

## 4. Benchmark Comparison Plan (OfficeQA Pro subset)

### 4.1 Why OfficeQA Pro — and MMLongBench-Doc as secondary

OfficeQA Pro is the hardest current suite (133 questions, 9 decades of Treasury Bulletins,
multi-document and multi-period questions) and has full baseline runs on disk
(`~/OneDrive/prj/officeqa/data/officeqa/{glm_5.3_Flash,mistral_glm}/`). The bar for the new
approach is **at least equivalent retrieval/answer quality at materially lower LLM cost**:
strict accuracy within **−5 points** of the same-LLM baseline on the subset, groundedness within
−3 points, and no regression in the `retrieval_or_lookup_error` category.

**MMLongBench-Doc (secondary, review remark 2)**: adapter already ships in genai-graph
(`genai_graph/bench/adapters/mmlongbench.py`, questions from `data/mmlongbench/questions.jsonl`).
It stress-tests the design where OfficeQA does not: single very long documents (skip L1 — the
runner scopes the question's document), multimodal chart/plot questions (routing via image
descriptions inside section descriptions), and ~22.5% unanswerable probes (`answerable_gate`).
Run a 25-question subset through the same matrix; primary metric addition: **unanswerable
detection rate** (declared-absent vs gold "Not answerable") alongside evidence hit rate.

### 4.2 Protocol

1. **Freeze a stratified subset** of 33 questions (25%): stratify by decade (9) and
   `source_files` cardinality (1 vs ≥2 documents); include the known-hard multi-period items
   (e.g. `UID0005`). Commit the frozen id list as `bench/officeqa_dm_subset.json` in the officeqa
   project and run via `cli bench run -p <profile> -q $(ids)`.
2. **Paired runs, same infrastructure** (identical docgraph profile, grader
   `DeepSeek-V4-Pro-0813@openrouter`, rubric, concurrency):
   - `default` — today's LLM-navigating deep agent (baseline; re-run on subset for pairing)
   - `decision_fast` — navigator `clef_flash@openrouter`
   - `decision_accurate` — navigator `jev@openrouter`
   - `decision_chat` — navigator = chat-adapter over `glm_5.3_Flash` (ablation: separates the
     *decision interface* from *model quality* — the paper's shared-context claim predicts this
     should approach the agentic baseline)
   - `retrieval_only` (Option E) — hybrid `search_sections` top-m → same synthesis prompt
   - optional `decision_laya` — local, zero-cost navigator
3. **Metrics** (from `scores_summary.json` + new `navigation` block):
   - Quality: strict / weighted accuracy, numeric match, groundedness, error taxonomy.
   - Cost: LLM completions per question, input/output tokens, end-to-end latency, $ per question
     (LLM + decision model separately).
   - Retrieval: **evidence hit rate** — gold `evidence` string found in selected sections
     (computable only for the navigator runs; measures retrieval quality decoupled from synthesis).
4. **Success criteria** (subset): strict accuracy ≥ baseline −5 pts, groundedness ≥ baseline −3 pts,
   **≥50% fewer LLM completions**, **≥40% fewer input tokens**, decision-model navigation adds
   <2 s per question. Report as an addendum to this study; promote to `docs/design/` if adopted.

### 4.3 Risks & mitigations

| Risk | Mitigation |
|---|---|
| Folder-level candidate list exceeds `max_questions` (hundreds of bulletins vs Clef's 64) | `candidate_cap` + `prefilter: hybrid` (`search_sections` top-48 → decision model re-ranks); paper shows decision scoring still adds accuracy over lexical ranking |
| Chunked `batch_invoke` weakens cross-candidate comparability | Keep one parent's candidates in requests sharing identical `state`; merge chunk results by probability; trace chunk ids |
| Multi-period / multi-document questions need >1 branch | `beam_width ≥ 3`; optional one-shot query decomposition (single LLM call, configurable `decompose: true`) before L1 |
| Decision model mis-scores archaic Treasury nomenclature | Escalation policy (Option C); `decision_chat` ablation diagnoses interface vs capability |
| Date/period filtering dominates navigation (paper's signal is semantic) | Candidate text always includes document date/period (parsed from filename metadata) alongside title+description |
| Navigator returns confident-but-wrong branch | `get_section_content` and `search_sections` remain available to the synthesis agent; ≤2 re-navigation calls with refined hints |

---

## 5. Implementation Plan

1. **genai-graph**: `genai_graph/agent/decision_navigator.py` — `DecisionNavigatorConfig`,
   `DecisionNavigator` (`rank_documents`, `rank_sections`, `navigate`), built on
   `get_decision_model()` + `batch_invoke`. Unit tests against `fake_decision@fake`
   (deterministic scores) covering beam selection, confidence gating, chunked levels,
   `domain_context` state composition, `answerable_gate`, trace.
2. **genai-graph**: `create_decision_navigation_tools(db_path=None, *, profile="default",
   **config)` factory — config arrives as tool-spec kwargs (§3.3); resolves `db_path` from the
   docgraph profile like `create_docgraph_tools`; emits `navigate_folder` plus the fallback graph
   tools. **No `create_docgraph_agent` changes** (tools flow through the profile `tools:` list).
3. **genai-graph**: extend `runs.jsonl` run records with the `navigation` block
   (dm calls/tokens/cost, selected sections + scores, low_confidence flag).
4. **officeqa**: `decision` agent profile — `type: deep`, factory-configured navigator tool
   (+ `decision-routing.md` domain-context excerpt split from `officeqa-qa`, + `decision-navigation`
   skill excerpt), `decision_fast` / `decision_accurate` / `decision_chat` / `retrieval_only`
   bench profiles, frozen subset file.
5. **Run the matrix** (§4.2) on OfficeQA Pro and the MMLongBench-Doc subset, aggregate with
   `cli bench report`, write results addendum below, and update the studies README index.

Phases 1–3 land in genai-graph (fully testable offline via the `fake` decision model); phases 4–5
in the officeqa project. No changes to ingestion or the graph schema are required.

---

## 6. Results & Findings — First Subset Run (2026-11-02)

All phases were implemented and the §4.2 matrix was started on OfficeQA Pro with **Clef-flash**
(`decision_model: fast` → `cloudflare/clef-flash` via OpenRouter Decisions API) as navigator and
`glm_5.3_Flash@openrouter` as synthesizer. 33/33 frozen-subset questions executed and 33/33 were graded
(`data/officeqa/decision_fast/{runs,scores}.jsonl`). §6.2–6.5 report the run **as configured then**;
§6.7 lists the fixes applied since (pack budgeting, tightened toolset, parallel L2) — a validation re-run
is the pending next step.

### 6.1 Level of achievement

| Phase | Status |
|---|---|
| 1. `DecisionNavigator` + unit tests (15, offline, scripted model) | done |
| 2. `create_decision_navigation_tools` factory (kwargs-configured) | done |
| 3. `runs.jsonl` navigation telemetry + summary aggregates | done |
| 4. officeqa `decision` profile, skills, `decision_fast` bench profile, frozen subset | done |
| 5. Run matrix + report | partial: `decision_fast` 33/33 run + 33/33 graded; other variants not run; §6.7 fixes await a validation re-run |

### 6.2 Outcome vs success criteria (§4.2)

| Criterion | Target | Observed (n=33, official judge) | Verdict |
|---|---|---|---|
| Strict accuracy | ≥ baseline −5 pts (baseline 69.7% strict on same ids) | **27.3%** strict, 34.8% weighted, 50% groundedness | miss |
| ≥50% fewer LLM completions | yes | Tool calls 19.9/q vs 21.8/q baseline (−9%) | miss |
| ≥40% fewer input tokens | yes | Mean 701k vs 614k (+14%); median +81%. Large savings on baseline's search-loop tail (e.g. 2.1M→187k, 4.0M→896k) | miss on mean; win on tail |
| Navigation overhead < 2 s | yes | 13.2 s/question (about 11 sequential joint calls) | miss |
| Navigation cost | — | $0.0007/question, 38k decision-model input tokens, zero generated tokens | pass |

Paired flips (baseline → decision): 8 correct→correct, 13 correct→incorrect, 5 incorrect→incorrect,
1 incorrect→partial, 1 partial→correct, 2 partial→partial, 2 correct→partial, 1 partial→incorrect.
Failure attribution on the 10 runs that ended with a recorded error: 6× deepagents
`_aintercept_large_tool_result` assertion, 3× recursion limit, 1× executor shutdown race — i.e. the 12
`halted_or_empty_response` verdicts are almost entirely infrastructural. Even the 23 runs that finished
without an error reached only 9/23 correct at 644k mean input tokens: the pack-paging failure mode (§6.5)
was the dominant cost driver, not navigation quality (the `low_confidence` flag split 5 correct / 4 incorrect
— the flag itself was not predictive of failure).

### 6.3 Trajectory analysis

- **Navigation worked and was precise.** Live probe for "federal debt held by the public, end of FY2011":
  the navigator ranked `treasury_bulletin_2012_03` first and scored `TABLE FD-1.—Summary of Federal Debt`
  at 0.944. Each parent is scored in one shared-context request, chunked by the 64-question cap.
- **The agent did not stay pack-first.** Tool mix over 33 questions: `read_file` 215, `get_section_content`
  116, `search_sections` 83, `get_document_toc` 52, `navigate_folder` 40 (1.2/question). The worst run
  (UID0117) made one navigation, flagged `low_confidence: true`, then 83 tool calls and 5.28M input tokens.
  The pack exceeded the observation limit, was evicted to `/large_tool_results/`, and the agent paged it
  back 30 lines per `read_file`, re-sending context every turn.
- **Telemetry is complete and cheap**: mean 11.2 decision-model calls, 38k tokens, $0.0007, 13.2 s per
  question, stored per run in the `navigation` block.
- **Low-confidence flag fired on 9/33 questions** (never `unanswerable`). The agent answered those by
  paging the pack instead of falling back to targeted graph tools.
- **Baseline on the same 33 ids** (official judge): strict 69.7%, weighted 75.8%, 21.8 tool calls,
  614k input tokens. The new profile capped the baseline's worst tails (1.3–4.0M tokens dropped to
  0.9M / 0.9M / 0.2M), so Option C's bounded worst case held.
- **Gold-ANSWER comparison works as before** (the judge grades against the `answer` column — that is the
  27.3% vs 69.7% figure above). What is not computable is the study's *evidence-hit-rate* retrieval metric
  (gold evidence snippet ∈ selected sections): the OfficeQA Pro split ships **no `evidence` column at all**
  (columns: uid, question, answer, source_docs, source_files, difficulty, doc_names, doc_name). Keep the
  evidence-hit-rate metric for MMLongBench-Doc only.

### 6.4 What went right

- The DecisionRAG mechanism transfers: shared-context joint scoring over `description`/`summary` routing text
  picked the right edition and table as top-1, at negligible cost and zero generated tokens.
- All configuration flows through tool-factory kwargs (§3.3). `create_docgraph_agent` was not changed; bench
  variants are sibling profiles.
- Per-run and aggregate telemetry (decision calls, tokens, cost, per-level scores, flags) is recorded.
- The stratified subset (9 decades × 1-vs-multi document) is frozen and reproducible.

### 6.5 What went wrong (root causes)

1. **Unbudgeted evidence packs** (up to ~100 KB of section text) exceed the observation limit, get evicted, and
   are paged back through `read_file`. This is the dominant failure: one navigation plus one synthesis
   becomes 80+ turns.
2. **deepagents SDK bug**: `_aintercept_large_tool_result` assertion ("Unreachable code") killed 6 of 33 runs
   on oversized tool results.
3. **`recursion_limit: 64`** is too low once the agent falls into pack paging plus fallback tools (3 runs hit it).
4. **Fallback policy too permissive**: the synthesizer used standard graph tools even when the pack was confident.
5. **Hybrid prefilter degraded to keyword-only**: the embeddings call fails here
   (`Embeddings.create() got an unexpected keyword argument 'model_name'`, a genai-tk issue), so the vector leg
   of `search_sections` is dead. The officeqa DB also predates the `content_hash` rename (sections and images key on
   `markdown_hash`). Tools now introspect this column (`_doc_key`), but a rebuild is still advisable.
6. **Operations**: a stale Prefect DB blocked the managed server (reset, stale `-wal`/`-shm` moved aside);
   Ladybug buffer pool exhausted at concurrency 6 ("buffer pool is full"); Langfuse OTLP egress errors (noise).

### 6.6 Recommendations

1. **Budget the evidence pack** (highest impact): cap each pack at roughly 15–20k characters, prefer leaf
   sections, truncate per section with a pointer to `get_section_content`, and use summaries as a fallback tier,
   so a pack never triggers eviction.
2. **Tighten the decision toolset**: remove `search_sections` and `get_folder_toc` from the decision profile and
   keep `get_section_content` for re-reads plus `query_image`. Make re-navigation with `hints` the primary recovery
   path, with graph-tool fallback gated on `low_confidence` only.
3. **Set `recursion_limit: 48`** (backstop ~2 steps/tool-call above the wrap-up hard limit of 20 calls);
   keep the wrap-up soft limit at 12.
4. **Fix the upstream bugs**: the genai-tk embeddings `model_name` kwarg (restores vector search) and the
   deepagents large-tool-result assertion (or chunk tool results above 10k chars).
5. **Grade the 32 pending runs** (`cli bench grade -p decision_fast`) before drawing accuracy conclusions. The
   unofficial heuristic (42% vs 69.7%) is pessimistic, but a real gap is likely.
6. **Parallelize the per-document L2 scoring** (about 11 sequential calls today) and consider one-shot query
   decomposition for multi-document questions.
7. **Then** run the remaining matrix rows (`decision_chat` ablation, `retrieval_only` lower bound). With the
   infrastructure in place, each row is a config-only change.

### 6.7 Recommendations implemented (2026-11-02, post-run)

Recommendations 1–5 were implemented immediately after grading; the design was promoted to
[docs/design/06_decision_model_navigation.md](../design/06_decision_model_navigation.md):

1. **Pack budgeting** — `pack_char_budget` (default 20000): sections are inlined best-score-first,
   all-or-nothing; over-budget sections render as a header plus a `get_section_content('<id>')` pointer
   instead of evicted text. OfficeQA profile uses 10000 (rendered pack ≈ 13k chars < 16000 truncation limit).
2. **Tightened toolset** — officeqa `decision` profile sets `excluded_tools: [get_folder_toc,
   search_sections, list_documents]` (stripped by deepagents' tool-exclusion middleware); recovery path is
   re-navigation with `hints` plus `get_document_toc` / `get_section_content`.
3. **Wrap-up / recursion pairing** — wrap-up soft 8 / hard 12 with cross-page extension capped at
   12 / 16, `recursion_limit: 64` (~3 LangGraph steps per tool call; see §6.8 for the arithmetic).
4. **Upstream fixes** — genai-tk `embeddings_factory` no longer passes `model_name`/`cache_dir` into
   pydantic embeddings classes (vector search restored on deepinfra; verified `embed_query` → dim 1024).
   The deepagents `_aintercept_large_tool_result` assertion is external; pack budgeting removes its trigger.
5. **Parallel L2** — `l2_workers` (default 1) scores the kept documents' sections concurrently with
   per-document traces merged into the run trace (OfficeQA profile: 3).

### 6.8 Validation re-run (3 questions, 2026-11-02)

Re-ran UID0015 (previously correct), UID0005 (hard multi-period, previously incorrect) and UID0068
(previously deepagents-crashed) with the §6.7 fixes:

- **UID0015**: correct → **correct** with 6 tool calls / 79k input tokens (was 13 calls / 207k) —
  pack-first now holds end-to-end; one navigation, no fallback tools.
- **UID0005**: incorrect-halted → **partial** (retrieval error) — pack budgeting + tool exclusion
  removed the crash; wrap-up's forced answer fired at 12 calls (242k tokens) instead of dying.
- **UID0068**: crashed → completed but still graded incorrect — now honestly a **retrieval miss**
  (the OCS rents line is absent from modern FFO-4; 2–4 low-confidence re-navigations never found it)
  instead of a crash.

Trajectory inspection answered "why did we crash at all — it should be the contrary":

1. **Per-step cost did NOT grow.** Both profiles consume ~3 LangGraph supersteps per tool call.
   The baseline never crashed only because its pairing was consistent: wrap-up hard 36 calls × ~4
   steps ≈ 144 < `recursion_limit: 160`. The first decision profiles broke that pairing
   (cross-page-extended wrap-up 28 calls ≈ 84 steps vs `recursion_limit: 48/64`).
2. **The decision agent flailed into the FILESYSTEM.** With the graph discovery tools excluded,
   a disappointing pack pushed glm into `grep`/`glob`/`ls` — mostly against `/officeqa/data`, which
   is fs-DENIED, plus an 8-variant `grep` loop over the skills workspace ("OCS", "royalt", "Shelf").
   The baseline flailed into the *graph* (useful tools); excluding them created a new failure mode.
   Fixed by excluding `ls`/`glob`/`grep` as well (`read_file` stays — SkillsMiddleware reads
   SKILL.md through it).

Final trio state (vs baseline on the same ids): UID0015 correct/79k (baseline correct/409k),
UID0005 partial/436k (baseline incorrect/1.6M), UID0068 incorrect-retrieval/257k (baseline
incorrect/2.1M) — **772k vs 4.1M tokens (−81%), no crashes**, tool sequences exactly the designed
loop (navigate → pointers → python → answer).

**Wrap-up / recursion arithmetic (learned the hard way)**: LangGraph counts steps (~3 per tool call
with deep agents + skills), WrapUpMiddleware counts tool calls and **auto-extends** limits for
cross-page conversations (defaults: hard 20 → extended 28 ≈ 84 steps). A `recursion_limit` below
the extended trajectory kills the run *before* wrap-up can force an answer — strictly worse than a
forced (possibly wrong) answer. Final pairing: wrap-up soft 8 / hard 12, cross-page extension capped
at 12 / 16, `recursion_limit: 64` (16 × ~3 + finalize < 64). The ACL warning was also de-duplicated
(warn once per backend — previously ~150 warnings per bench run).

**Remaining known issue**: the deepagents tool-rejection loop (model keeps emitting stripped tools)
ends an empty answer for glm; a downstream fix could make the rejection path inject the final-answer
prompt as a user turn.

### 6.9 Second full-subset attempt (2026-11-02 evening) — cost targets met, accuracy unmeasurable

Configuration: concurrency 2, `ladybug.buffer_pool_size: 4GB` (new config key), local python
executor, `pack_char_budget: 10000`, `excluded_tools: [get_folder_toc, search_sections,
list_documents, ls, glob, grep]`, wrap-up 8/12, `recursion_limit: 64`.

Outcome: **26/33 completed, 7 never ran**. From ~15:58 the run wedged in a retry loop (2,305
`buffer pool is full` tool errors over ~2 h) and was killed. The new fail-fast guard did **not**
fire — it inspects only *completed* records, and the wedged question never completed.

Measured on the 26 completed questions, against the baseline on the same ids:

| Metric | decision_fast | baseline (glm_5.3_Flash) |
|---|---|---|
| Strict accuracy | 1 correct + 3 partial (4%) | 16 correct + 4 partial (62%) |
| Avg tool calls | 11.3 | 22.5 (−50%) |
| Avg input tokens | 116k | 628k (−81%) |

**Attribution: the accuracy number measures a broken database, not the navigator.** 24 of the 26
completed runs contain at least one buffer-pool error (only `UID0001` and `UID0003` were clean), so
most agents were reading a failing graph for part of their investigation and correctly refused to
fabricate answers. The cost figures (−50% calls, −81% tokens) are real and consistent with the design.

**Root cause (new evidence): the exhaustion is cumulative, not RAM-bound.** The pool failed after
~26 questions at concurrency 2 with 12 GB of host RAM free, and after ~17 questions at concurrency 4
with 8 GB. Each question builds a fresh agent plus a thread-local Ladybug connection that is
deliberately kept alive for the process lifetime (`_KEEPALIVE_BACKENDS`, added to avoid a native
heap-corruption bug), so native memory grows monotonically with completed questions until the buffer
manager cannot allocate. Pool size only moves the failure point.

Secondary defects found in the trajectories:

1. **No discovery fallback.** With `list_documents`, `get_folder_toc`, `search_sections`, `ls`,
   `glob` and `grep` all excluded, a disappointing pack leaves no way to enumerate the corpus. One
   answer says it explicitly ("…or with a document-listing tool available"), and several then narrate
   what they still need instead of answering. The token win came from the pack budget, not from
   removing recovery paths.
2. **Wrap-up 8/12 converges too early.** Answers end mid-investigation ("I need a later bulletin…")
   and the judge correctly files them as `halted_or_empty_response` (22 of 33 verdicts).
3. **Fail-fast gap.** The guard must also watch the *running* question, and a per-question wall-clock
   timeout is missing — one wedged question consumed the whole budget.
4. **Navigation traces record `scores: {}`** at every level, so the cheap decision model cannot be
tuned from the recorded runs.
5. **Corpus gap.** `UID0003`'s gold table (1953 monthly detail, Table 6B) is genuinely absent from
   the graph — the agent's refusal was correct.
6. **Log noise.** Langfuse OTLP export fails every few seconds for the whole run (its container is
   restarting), burying real errors.

Proposed improvements, priority order:

1. **Release per-question native state** — close/evict the thread-local connection and drop agent
   references between questions (or run each question in a subprocess). This is the actual fix for
   the wedge; pool sizing only delays it.
2. **In-flight watchdog** — per-question wall-clock timeout (~20 min) plus a consecutive-DB-error
   circuit breaker that fails the question and aborts the run, instead of retrying a dead database.
3. **Restore a discovery fallback** — keep `list_documents` and `get_folder_toc` (both cheap); keep
   `ls`/`glob`/`grep` excluded (they hit fs-denied paths). Pack-first discipline stays in the prompt.
4. **Relax wrap-up to soft 16 / hard 24** with `recursion_limit ≈ 96`; 8/12 truncates legitimate
   multi-period investigations.
5. **Record per-candidate probabilities in the navigation trace** so the decision model can be tuned
   offline from `runs.jsonl`.
6. **Preflight the corpus** — assert each question's gold table is present and extractable before
   scoring, and report ingestion gaps separately from navigation failures.
7. **Refine the judge rubric** to distinguish "declared the evidence missing" from "crashed/empty",
   since the two are currently merged into `halted_or_empty_response`.
8. **Fix or disable Langfuse export** during bench runs.

## 7. References

- Wan, Jünger, Du, Liu, Krishnan, Weinberger — [Retriever-Free Retrieval: Navigating Documents with Decision Models](https://arxiv.org/abs/2610.35868), arXiv:2610.35868, Oct 2026 (DecisionRAG)
- genai-tk — Decision models (System One): `genai_tk/core/decision_models/`, `get_decision_model` factory, `skills/development/decision-models/SKILL.md`
- genai-graph — Document Graph tools: `genai_graph/kg/query/document_graph_tools.py`; agent wiring: `genai_graph/agent/docgraph_agent.py`
- OfficeQA Pro baseline results: [docs/benchmarks_financebench_officeqa.md](../benchmarks_financebench_officeqa.md)
- Related hierarchical navigation (LLM-based, token-generating): MemWalker (Chen et al. 2023), PageIndex (VectifyAI 2025), LATTICE (Gupta et al. 2026)
