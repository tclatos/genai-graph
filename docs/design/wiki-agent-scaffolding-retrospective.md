# Case Study — Building the `wiki` Agent from Scaffolding: What Broke and What to Fix

> Retrospective + playbook from building a complete Document-Graph agent project
> (`~/prj/wiki`) with a coding agent (Warp/Oz) using only the toolkit's README,
> scaffolded skills, and library surface — no local genai-graph access required.
> Goal of this document: a new user (or a new agent run) should be able to repeat
> the exercise from a single prompt, and the friction encountered here should no
> longer exist.

## 1. Where we are

Everything planned for the wiki exercise is implemented and validated:

- **Project scaffolded** at `~/prj/wiki` following the genai-graph README
  quick start (`uv init` → `uv add` genai-tk + genai_graph → `cli init --name
  Wiki --with-graph --extra harnessing` → `uv sync`).
- **Wiki CLI group** (`wiki/commands/wiki_commands.py`) with three commands:
  - `wiki add SOURCES...` — markdownize (docling) + ingest the Document Graph,
    with structure discovery, section summaries and VLM image descriptions.
  - `wiki search QUERY` — hybrid (vector + BM25) / vector / bm25 section search.
  - `wiki ask [QUESTION]` — deep agent over the Document Graph; one-shot,
    `--chat` REPL (shared `chat_repl`), or `--tui` Textual app (`wiki/tui.py`).
- **Configuration** mirrors the proven officeqa setup: docgraph `wiki` profile
  (docling, images, summaries, embeddings `qwen3_06b@deepinfra`), deep agent
  profile `wiki` (`glm_5.3_Flash@openrouter`, officeqa middleware stack),
  domain skill `skills/custom/wiki-qa`.
- **Validation done**: seeded corpus (Markdown + DOCX with an embedded chart)
  ingests (3 docs / 10 sections / FTS index); BM25 search ranks the correct
  sections first; the deep agent builds with 7 tools + 4 skills and reaches the
  LLM API boundary; `--chat` starts/exits cleanly; `--tui` passes a headless
  pilot test (mount → stream a turn → `/quit`); `cli docgraph/agent/trajectory/
  bench` all still work; `just lint` clean; 49/49 scaffolded skills valid.
- **Toolkit fixes landed** (see §4): scaffolder registers `DocGraphCommands`,
  chat-TUI recipe added to `cli-chat-interfaces`, README quick start fixed,
  models.dev fetch made non-fatal, dependency source pins untangled.

Known gap: live LLM answering could not be exercised from this workstation —
the corporate proxy only whitelists package hosts (pypi/github); openrouter,
deepinfra, LangSmith etc. time out. The pipeline is verified up to the network
boundary. First run on an open network should confirm `wiki ask` end to end.

## 2. What went wrong during scaffolding

In chronological order, each with the underlying cause:

1. **`uv add genai_graph` from git failed** — genai-graph's `pyproject.toml`
   declared `genai-tk` through a *local relative path* source
   (`../genai-tk`). A git checkout built anywhere else resolves that path
   against the wrong directory. *Fixed by pinning the git source in
   genai-graph — but see §4: that fix created the next two problems.*
2. **Scaffolded project missed CLI commands** — after `cli init --with-graph`,
   `cli docgraph` was not registered: the scaffolder's generated
   `app_conf.yaml` only added `BenchCommands`, not `DocGraphCommands`.
   *Fixed in `genai_tk/main/scaffolder.py` (adds `DocGraphCommands` when
   `--with-graph`).*
3. **Config merge warnings from example files** — the global config glob picked
   up bundled `examples/**` agent-profile YAMLs and merged them as if they were
   project config. *Fixed by excluding `!examples/**` in genai-tk's
   `config/app_conf.yaml` profile patterns.*
4. **Switching to local editable checkouts created dependency conflicts** —
   genai-graph's baked git source for genai-tk conflicted with the sibling
   checkout source ("Requirements contain conflicting URLs"). The workaround
   (consumer-level `override-dependencies`) *masked* the real problem and
   caused the worst bug of the exercise (§3.1).

## 3. What went wrong during development

1. **The `harnessing` extra silently never installed** (the big one). The wiki
   `pyproject.toml` had `override-dependencies = ["genai-tk @ file:///..."]`.
   uv applies overrides to *every* requirement for that package — including the
   project's own forwarded `genai-tk[harnessing]` extra — and an override
   without extras strips them. Result: `deepagents`/`agent-sandbox`/
   `opensandbox`/`deerflow-harness` were missing and every deep-agent command
   died with `ImportError: Optional feature 'harnessing' is not installed`
   even though `uv sync --extra harnessing` "succeeded" (it considered the
   already-wrong lockfile consistent). Diagnosis took a full traceback read;
   the fix was removing the override entirely.
2. **URL-source rejection for git-resolved packages.** With the override gone,
   uv then reported the conflicting-URLs error again, and separately refused
   `deerflow-harness` ("URL dependencies must be expressed as direct
   requirements or constraints"): genai-tk pinned deerflow via
   `[tool.uv.sources]`, and uv rejects URL *sources* declared by non-root
   packages when the package itself resolves from git. *Fixed by: (a)
   genai-graph keeps `genai-tk` bare and expresses its own git pin as a
   root-only `toolkit` dependency-group (consumers never inherit groups);
   (b) genai-tk expresses the deerflow pin as a direct URL requirement inside
   the `harnessing` extra, which is honored in every resolution context.*
3. **Prefect ephemeral server timeouts.** `wiki add` → `markdownize_flow`
   started Prefect's ephemeral server, whose `/health` check kept timing out.
   Root cause was twofold: the corporate proxy intercepted localhost traffic
   (`NO_PROXY` missing localhost), and the ephemeral server is fragile anyway.
   *Fixed in the wiki command: auto-start the managed server
   (`prefect_server().ensure_running()` + `configure_api_url()`, which also
   sets the localhost proxy bypass in-process).*
4. **models.dev fetch crashed agent startup.** With the network blocked, the
   LLM registry's hard `httpx.get("https://models.dev/api.json")` raised and
   took `wiki ask` down before any agent work. *Fixed in
   `genai_tk/core/models_db.py`: auto-fetch failure now logs a warning and
   continues with an empty index; alias resolution then fails with an explicit,
   actionable message ("add the model explicitly to llm.yaml").*
5. **Profiles referenced undeclared models.** The wiki agent profile used
   `glm_5.3_Flash@openrouter` etc., but the project's `config/providers/
   llm.yaml` declared none of them — resolution relied on the (unavailable)
   models.dev catalogue. *Fixed by adding explicit registry entries for
   `glm_5.3_Flash`, `deepseek-v4-flash-0731` and `gemini-2.5-flash`.*
6. **Stale middleware path copied from the reference project.** officeqa's
   agent profile references `genai_graph.agent.middleware.wrap_up.
   WrapUpMiddleware`, which does not exist in genai-graph (the class lives in
   `genai_tk.agents.langchain.middleware.wrap_up_middleware`). Copying the
   officeqa stack propagated the broken import. *Fixed in the wiki profile;
   officeqa should be fixed too (see §5).*
7. **Network policy vs. hybrid search.** The `wiki` profile enables embeddings,
   so ingestion failed at the embedding step (blocked API) and rolled back the
   whole graph build. *Fixed by supporting `wiki add --embeddings none`
   (vectorless graph, BM25-only) so ingestion completes offline.*

## 4. Fixes landed (by repo)

- **genai-tk**: `pyproject.toml` (deerflow direct-URL requirement in
  `harnessing`), `genai_tk/core/models_db.py` (non-fatal auto-fetch),
  `genai_tk/main/scaffolder.py` (register `DocGraphCommands` with
  `--with-graph`), `config/app_conf.yaml` (exclude `examples/**`),
  `skills/development/cli-chat-interfaces/SKILL.md` (new *Textual TUI Variant*
  section documenting the streaming chat-TUI pattern proven in `wiki/tui.py`).
- **genai-graph**: `pyproject.toml` (bare `genai-tk` + root-only `toolkit`
  group), `README.md` (quick start now passes `--extra harnessing`, with a note
  that deep agents need it), and this document.
- **wiki**: full agent implementation, profiles, skills, plus
  `wiki_commands.py` Prefect bootstrap and the `--embeddings none` option.

⚠️ The pure git-based flows (README quick start on a fresh machine, and
genai-graph's own `uv sync`) only resolve **after the genai-tk and
genai-graph changes are pushed**, because uv resolves `@main` from the remote.
Sibling-checkout consumers (wiki/officeqa/rfq_pricing) work locally and all
re-lock cleanly.

## 5. What to improve

### Skills
- **`cli-chat-interfaces`** now covers the Textual TUI pattern — but a
  follow-up should also document headless testing of chat TUIs
  (`App.run_test()` + pilot) so `--tui` commands are CI-testable.
- **`kg-docgraph-agent`** ("wiring a downstream project") should mention: add
  explicit `llm.yaml` entries for every model the profiles reference (never
  rely on models.dev fuzzy resolution); verify middleware `class:` paths
  against the actual module layout (`genai_tk.agents.langchain.middleware.*`,
  `genai_graph.agent.middleware.map_before_search`) instead of copying another
  project's YAML.
- **A new skill (or extension of `cli-and-scaffolding`) on uv sources for
  sibling-checkout development** — the three rules that cost the most time:
  (1) never `override-dependencies` a package whose extras you forward;
  (2) never bake `[tool.uv.sources]` for a dependency in a *library* that
  consumers resolve differently; (3) prefer direct URL requirements over
  URL sources when a package must resolve from git. Add the failing symptoms
  (stripped extras / conflicting URLs / "URL dependencies must be expressed…")
  as searchable anchors.
- **`optional-features`** should teach `require_feature` gating for deep-agent
  entry points and how to read its ImportError as a fix instruction.

### Docs / README
- **genai-graph README**: done (`--extra harnessing` in quick start). Still
  worth adding: an offline/firewalled note (proxy → `NO_PROXY=localhost`,
  models.dev cache, `--embeddings none` fallback) and a "what a deep agent
  needs" checklist (harnessing extra + declared models + valid middleware
  paths).
- **`docs/document-graph.md`**: document the managed-Prefect expectation for
  programmatic flow use (`prefect_server().ensure_running()` +
  `configure_api_url()` before invoking flows directly), not just CLI usage.
- **officeqa**: fix its stale `wrap_up.WrapUpMiddleware` reference (it will
  break the same way it broke wiki).

### Scaffolder
- Emit the **forwarding extras block + `[tool.uv.sources]` with a warning
  comment** into scaffolded pyprojects (already proven correct in wiki), and
  consider emitting `requires-python = ">=3.12,<3.13"` (matches genai-tk
  reality and avoids uv solving future-python splits that git-only deps cannot
  satisfy).
- Scaffold a **`.gitignore` that excludes `data/` runtime subdirs** (kg,
  markdown, traces, trajectories, kv_store) while keeping seed sources —
  the wiki project had to hand-patch this before its first commit.
- Add a **`cli doctor`** (or extend `cli info`) that pre-flights exactly the
  failure modes hit here: harnessing feature present, models declared for the
  active profiles, Prefect server reachable, model catalogue cached, proxy
  bypass for localhost. Every one of these was discovered reactively.

## 6. Reproduction playbook (prompt for a coding agent)

Given the current toolkit state, the whole exercise reduces to:

```text
Using the genai-graph README quick start, create a new project <name> that
wraps the Document Graph in a single `wiki` CLI group:
  1. uv init; uv add genai-tk and genai_graph (git or sibling editable checkouts
     per [tool.uv.sources] — never use override-dependencies).
  2. cli init --name <Name> --with-graph --extra harnessing; uv sync.
  3. Add a docgraph profile (docling, images, summaries, embeddings) and a
     `type: deep` agent profile mirroring officeqa's — declare every referenced
     model explicitly in config/providers/llm.yaml and verify every middleware
     `class:` path exists.
  4. Implement: `wiki add` (markdownize + ingest; auto-start the managed
     Prefect server first; support --embeddings none), `wiki search` (hybrid/
     vector/bm25 over document_graph_tools.search_sections), `wiki ask`
     (create_docgraph_agent + stream_turn/run_chat_repl/Textual TUI per the
     cli-chat-interfaces skill).
  5. Write a domain skill (navigation loop + citation rules) under
     skills/custom/, seed data/sources, and validate: ingest → search →
     one-shot ask → --chat → --tui, plus `just lint` and
     `cli skills validate --all`.
```

After the toolkit fixes are pushed, steps 1–2 should work verbatim from the
README with no local checkouts; before that, use sibling editable checkouts
for both libraries.

## 7. Open items

- [ ] Push genai-tk + genai-graph; then on a clean machine re-run the README
      quick start verbatim to confirm the git-based flow resolves.
- [ ] First `wiki ask` against a real LLM on an open network (validate
      `query_image` on the seeded chart too).
- [ ] Fix officeqa's stale middleware path; add the uv-sources skill;
      implement `cli doctor`.
- [ ] Consider `mistral_ocr` / document-size defaults for the `wiki` profile —
      only docling has been exercised so far.
