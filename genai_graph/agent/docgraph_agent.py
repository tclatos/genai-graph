"""Build and run a deep agent over the Document Graph.

The agent is a genai-tk ``type: deep`` profile (DeepAgents SDK) wired with the
read-only Document Graph navigation tools from
:mod:`genai_graph.kg.query.document_graph_tools` and the runtime skills co-located
under ``genai_graph/agent/skills/``. The tools and the target folder are injected
at runtime so ``--db`` / ``--folder`` / ``--llm`` overrides work without editing
the profile.

Skills are loaded via DeepAgents' ``SkillsMiddleware`` through a
``FilesystemBackend`` whose root also bounds everything the agent's file tools
(``ls``/``grep``/``glob``/``read_file``) can reach. Skill directories from
different project trees are therefore COPIED into a minimal per-question
workspace that becomes the backend root — rooting the backend at the skills'
common ancestor would expose every intermediate directory (other projects,
benchmark question banks, run records with gold answers) to the agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger

from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.query.document_graph_tools import (
    DocumentGraphError,
    create_document_graph_tools,
    get_folder_path,
    resolve_folder_id,
)

DEFAULT_LLM = "deepseek_v4flash@openrouter"
DEFAULT_PROFILE = "docgraph"

# Co-located runtime skills, resolved from the package location so they work
# no matter which project imports genai_graph.
_PACKAGE_SKILLS_DIR = str(Path(__file__).resolve().parent / "skills")


def find_docgraph_db_path(profile: str = "default") -> str | None:
    """Search for database path in docgraph_profiles, paths.kg_db, or graph_db.default.

    Checks:
    1. global_config(): ``docgraph_profiles.<profile>.paths.kg_db`` / ``path.kg_db`` / ``kg_db``
    2. config/docgraph.yaml: ``docgraph_profiles.<profile>.paths.kg_db`` / ``path.kg_db``
    3. global_config(): ``graph_db.default`` / ``paths.kg_db``
    """
    try:
        from genai_tk.config_mgmt.file_patterns import resolve_config_path
    except ImportError:

        def resolve_config_path(p: str) -> str:  # type: ignore[misc]
            return str(Path(p).expanduser())

    # 1. Try global_config()
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        cfg = global_config()

        # Dot-notation lookup
        for key in (
            f"docgraph_profiles.{profile}.paths.kg_db",
            f"docgraph_profiles.{profile}.path.kg_db",
            f"docgraph_profiles.{profile}.kg_db",
        ):
            val = cfg.get(key, None)
            if val:
                return resolve_config_path(str(val))

        # DictConfig / dict lookup
        profiles = cfg.get("docgraph_profiles", None)
        if isinstance(profiles, dict) or hasattr(profiles, "get"):
            p_data = profiles.get(profile, {})
            if isinstance(p_data, dict) or hasattr(p_data, "get"):
                paths = p_data.get("paths") if hasattr(p_data, "get") else getattr(p_data, "paths", None)
                if not paths and (isinstance(p_data, dict) or hasattr(p_data, "get")):
                    paths = p_data.get("path")
                if isinstance(paths, dict) or hasattr(paths, "get"):
                    kg_db = paths.get("kg_db")
                    if kg_db:
                        return resolve_config_path(str(kg_db))
                elif isinstance(p_data, dict) and p_data.get("kg_db"):
                    return resolve_config_path(str(p_data["kg_db"]))
    except Exception:
        pass

    # 2. Try loading config/docgraph.yaml directly
    try:
        from genai_graph.bench.config import load_raw_docgraph_yaml

        raw = load_raw_docgraph_yaml()
        profiles = raw.get("docgraph_profiles", {}) or {}
        if profile in profiles:
            p_data = profiles[profile] or {}
            paths = p_data.get("paths") or p_data.get("path") or {}
            if isinstance(paths, dict):
                kg_db = paths.get("kg_db")
                if kg_db:
                    return resolve_config_path(str(kg_db))
            if p_data.get("kg_db"):
                return resolve_config_path(str(p_data.get("kg_db")))
    except Exception:
        pass

    # 3. Fallbacks: graph_db.default or top-level paths.kg_db
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        cfg = global_config()
        default_db = cfg.get("graph_db.default", None)
        if default_db:
            return resolve_config_path(str(default_db))
        top_kg_db = cfg.get("paths.kg_db", None) or cfg.get("path.kg_db", None)
        if top_kg_db:
            return resolve_config_path(str(top_kg_db))
    except Exception:
        pass

    return None


def resolve_db_path(db_path: str | None = None, profile: str = "default") -> str:
    """Return *db_path* or the configured ``docgraph_profiles.<profile>.paths.kg_db`` / ``graph_db.default``.

    Raises:
        DocumentGraphError: When no path is given and no default is configured.
    """
    if db_path:
        return db_path

    resolved = find_docgraph_db_path(profile)
    if resolved:
        return resolved

    raise DocumentGraphError(
        f"No database path provided and no `docgraph_profiles.{profile}.paths.kg_db` or `graph_db.default` configured. "
        "Pass --db <path>, --profile <name>, or configure docgraph_profiles in config/docgraph.yaml."
    )


def create_document_graph_tools_from_config(
    db_path: str | None = None, *, profile: str = "default", embeddings_id: str | None = None
) -> list[BaseTool]:
    """Build the navigation tools, resolving *db_path* from config when omitted."""
    return create_document_graph_tools(resolve_db_path(db_path, profile=profile), embeddings_id=embeddings_id)


def build_docgraph_system_prompt(
    folder_id: str | None = None,
    folder_name: str | None = None,
    *,
    base: str | None = None,
) -> str:
    """Build the system prompt that frames the agent as a Document Graph analyst.

    When *folder_id* is given the agent is scoped to that folder's documents.
    """
    target = ""
    if folder_id:
        label = f"{folder_name!r} ({folder_id})" if folder_name else folder_id
        target = f"\n\n[Target folder: {label} — focus your search on this folder's documents.]"

    prompt = f"""\
You are a document-graph analyst. You answer questions by NAVIGATING the Document
Graph loaded into your tools — a Ladybug graph of Folders → Documents → Markdown
sections. You do NOT have the documents memorised; you must read them via tools.

Navigation loop (vectorless agentic RAG):
1. `get_folder_toc(folder_id)` — list the documents in the folder, each with an id
   and a one-line description. Pick the document(s) most likely to answer.
2. `get_document_toc(document_id)` — get one document's section tree: section ids,
   titles and one-line descriptions. This is the map; use it to pick sections.
3. `get_section_content(section_ids)` — read the raw Markdown of ONLY the sections
   whose description matches the question (comma-separated ids).
4. `search_sections(keyword, folder_id=...)` — when you do not know which document
   or section holds an answer, keyword-search titles and text across the folder.
5. Iterate: read more sections or search again with different keywords until you
   have grounded evidence, then answer.

Rules:
- Ground every claim in section text you actually read; cite section ids as
  `[hash::sequence]` and name the source document.
- If a tool returns "No ... found", try another tool or keyword — never guess.
- Never invent content that is not in the graph. If information is genuinely
  absent, say so explicitly.
- Return your analysis as your message. Do NOT use write_file/edit_file — the
  caller persists the report.{target}
"""
    if base:
        prompt = f"{prompt}\n\n{base}"
    return prompt


def _resolve_folder(backend: KuzuBackend, folder_ref: str | None) -> tuple[str | None, str | None]:
    """Resolve a folder reference to (folder_id, folder_name); (None, None) when omitted."""
    if not folder_ref:
        return None, None
    folder_id = resolve_folder_id(backend, folder_ref)
    if folder_id is None:
        raise DocumentGraphError(
            f"No folder found matching {folder_ref!r}. "
            "Use `cli docgraph folders` to list ingested folders, or omit --folder to search everything."
        )
    chain = get_folder_path(backend, folder_id)
    name = chain[-1]["name"] if chain else None
    return folder_id, name


def _stage_skills_workspace(resolved_skills: list[str]) -> tuple[str, list[str]]:
    """Copy skill directories into a minimal workspace and return (root, staged dirs).

    The ``FilesystemBackend`` root bounds the agent's whole file-tool surface
    (``ls``/``grep``/``glob``/``read_file``). Staging copies of the skills into a
    fresh workspace keeps that surface limited to the skill docs themselves;
    rooting at the skills' common ancestor instead exposes every intermediate
    directory — including unrelated projects and benchmark ground-truth files.
    """
    import shutil
    import tempfile

    workspace = Path(tempfile.mkdtemp(prefix="docgraph-ws-"))
    staged: list[str] = []
    for src in resolved_skills:
        src_path = Path(src).resolve()
        dst = workspace / src_path.name
        if dst.exists():
            # Skill-name collision across sources: merge into one directory.
            shutil.copytree(src_path, dst, dirs_exist_ok=True)
        else:
            shutil.copytree(src_path, dst)
        staged.append(str(dst))
    return str(workspace), staged


def prepare_docgraph_profile(
    profile: Any,
    *,
    db_path: str | None = None,
    docgraph_profile: str = "default",
    folder_id: str | None = None,
    extra_skill_dirs: list[str] | None = None,
) -> Any:
    """Mutate and return *profile* in place for a Document Graph run.

    Sets the system prompt (scoped to *folder_id* when given), the skill
    directories (package skills + caller extras + profile-listed), and a
    filesystem backend rooted at the common ancestor of those skill dirs.
    """
    resolved_db = resolve_db_path(db_path, profile=docgraph_profile)

    folder_resolved_id: str | None = None
    folder_name: str | None = None
    if folder_id:
        backend = KuzuBackend()
        backend.connect(resolved_db)
        try:
            folder_resolved_id, folder_name = _resolve_folder(backend, folder_id)
        finally:
            backend.close()

    profile.system_prompt = build_docgraph_system_prompt(folder_resolved_id, folder_name, base=profile.system_prompt)

    excluded = frozenset(getattr(profile, "excluded_tools", []) or [])
    if "read_file" in excluded:
        # Graph-only mode: file tools (incl. read_file) are stripped by the
        # tool-exclusion middleware, so SkillsMiddleware could not read any
        # SKILL.md anyway, and its progressive-disclosure prompt ("use
        # read_file ... for full instructions") would point the agent at a tool
        # it lacks. The navigation strategy must be inlined in the system prompt
        # (see the docgraph profile) instead. Skip skill loading and the
        # filesystem backend entirely for a clean, graph-only toolset.
        profile.skill_directories = []
        logger.info(
            "Document-graph agent: graph-only mode (read_file excluded) — "
            "skipping skill loading; navigation strategy is inlined in the system prompt."
        )
        return profile

    skill_dirs: list[str] = [_PACKAGE_SKILLS_DIR]
    if extra_skill_dirs:
        skill_dirs.extend(extra_skill_dirs)
    if getattr(profile, "skill_directories", None):
        skill_dirs.extend(profile.skill_directories)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for d in skill_dirs:
        if d not in seen:
            seen.add(d)
            deduped.append(d)
    profile.skill_directories = deduped

    # Resolve/expand (drops missing dirs, expands one level) so the backend root
    # is computed from the same dirs the SkillsMiddleware will actually scan.
    from genai_tk.agents.langchain.factory import _resolve_skill_dirs

    resolved_skills = _resolve_skill_dirs(deduped)
    if resolved_skills:
        workspace, staged_skills = _stage_skills_workspace(resolved_skills)
        profile.skill_directories = staged_skills
        if hasattr(profile, "backend"):
            from genai_tk.agents.langchain.config import BackendConfig

            profile.backend = BackendConfig(type="filesystem", root_dir=workspace)
        logger.info(
            "Document-graph agent skills staged in {} (workspace: {})",
            staged_skills,
            workspace,
        )
    else:
        logger.warning("No skill directories resolved for document-graph agent; running without skills.")

    return profile


def create_docgraph_agent(
    profile: Any,
    *,
    llm: str | None = None,
    db_path: str | None = None,
    docgraph_profile: str = "default",
    folder_id: str | None = None,
    extra_skill_dirs: list[str] | None = None,
    embeddings_id: str | None = None,
) -> Any:
    """Prepare *profile* and return a ready-to-stream :class:`LangChainHarness` or :class:`DeerFlowHarness`.

    The harness lazily compiles the agent on first use. The navigation tools
    are injected as ``extra_tools`` so they reflect the resolved ``db_path`` and
    ``folder_id`` without touching the profile YAML.

    Args:
        profile: A resolved ``AgentProfileConfig`` (``type: deep``) or ``DeerFlowProfile``,
            typically from :func:`genai_tk.agents.harness.profiles.load_agent_profiles`.
        llm: LLM identifier override (e.g. ``"deepseek_v4flash"``).
        db_path: Ladybug database path; resolved from ``docgraph_profiles.<docgraph_profile>.paths.kg_db`` when None.
        docgraph_profile: DocGraph profile name for database resolution (default: 'default').
        folder_id: Folder to scope the agent to (hash, prefix, or name).
        extra_skill_dirs: Additional runtime skill directories (e.g. a project's
            use-case skills).
        embeddings_id: Embeddings model id enabling the hybrid (vector + BM25)
            ``search_sections`` mode; None keeps keyword search only.

    Returns:
        A :class:`genai_tk.agents.harness.base.BaseHarness` instance (:class:`LangChainHarness` or :class:`DeerFlowHarness`).
    """
    prepare_docgraph_profile(
        profile,
        db_path=db_path,
        docgraph_profile=docgraph_profile,
        folder_id=folder_id,
        extra_skill_dirs=extra_skill_dirs,
    )
    tools = create_document_graph_tools_from_config(db_path, profile=docgraph_profile, embeddings_id=embeddings_id)
    if getattr(profile, "harness", "langchain") == "deerflow":
        from genai_tk.agents.harness.deerflow_harness import DeerFlowHarness

        return DeerFlowHarness(
            profile,
            llm_override=llm,
            extra_tools=tools,
        )

    from genai_tk.agents.harness.langchain_harness import LangChainHarness

    return LangChainHarness(
        profile,
        llm_override=llm,
        force_memory_checkpointer=True,
        extra_tools=tools,
    )


async def run_docgraph_agent(harness: Any, query: str, *, show_trace: bool = False) -> str:
    """Run one turn against *harness*, streaming events, and return the assistant text."""
    from genai_tk.agents.harness.chat_repl import astream_turn

    return await astream_turn(harness, query, show_trace=show_trace)
