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


def resolve_db_path(db_path: str | None) -> str:
    """Return *db_path* or the configured ``graph_db.default``.

    Raises:
        DocumentGraphError: When no path is given and no default is configured.
    """
    if db_path:
        return db_path
    from genai_tk.config_mgmt.config_mngr import global_config

    default_db = global_config().get("graph_db.default", None)
    if default_db:
        return str(default_db)
    raise DocumentGraphError(
        "No database path provided and no `graph_db.default` configured. "
        "Pass --db <path> or add graph_db.default to your config."
    )


def create_document_graph_tools_from_config(
    db_path: str | None = None, *, embeddings_id: str | None = None
) -> list[BaseTool]:
    """Build the navigation tools, resolving *db_path* from config when omitted."""
    return create_document_graph_tools(resolve_db_path(db_path), embeddings_id=embeddings_id)


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
    folder_id: str | None = None,
    extra_skill_dirs: list[str] | None = None,
) -> Any:
    """Mutate and return *profile* in place for a Document Graph run.

    Sets the system prompt (scoped to *folder_id* when given), the skill
    directories (package skills + caller extras + profile-listed), and a
    filesystem backend rooted at the common ancestor of those skill dirs.
    """
    resolved_db = resolve_db_path(db_path)

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
        from genai_tk.agents.langchain.config import BackendConfig

        workspace, staged_skills = _stage_skills_workspace(resolved_skills)
        profile.skill_directories = staged_skills
        profile.backend = BackendConfig(type="filesystem", root_dir=workspace)
        logger.info(
            "Document-graph agent skills staged in {} (backend root: {})",
            staged_skills,
            profile.backend.root_dir,
        )
    else:
        logger.warning("No skill directories resolved for document-graph agent; running without skills.")

    return profile


def create_docgraph_agent(
    profile: Any,
    *,
    llm: str | None = None,
    db_path: str | None = None,
    folder_id: str | None = None,
    extra_skill_dirs: list[str] | None = None,
    embeddings_id: str | None = None,
) -> Any:
    """Prepare *profile* and return a ready-to-stream :class:`LangChainHarness`.

    The harness lazily compiles the deep agent on first use. The navigation tools
    are injected as ``extra_tools`` so they reflect the resolved ``db_path`` and
    ``folder_id`` without touching the profile YAML.

    Args:
        profile: A resolved ``AgentProfileConfig`` (``type: deep``), typically from
            :func:`genai_tk.agents.harness.profiles.load_langchain_profiles`.
        llm: LLM identifier override (e.g. ``"deepseek_v4flash"``).
        db_path: Ladybug database path; resolved from ``graph_db.default`` when None.
        folder_id: Folder to scope the agent to (hash, prefix, or name).
        extra_skill_dirs: Additional runtime skill directories (e.g. a project's
            use-case skills).
        embeddings_id: Embeddings model id enabling the hybrid (vector + BM25)
            ``search_sections`` mode; None keeps keyword search only.

    Returns:
        A :class:`genai_tk.agents.harness.langchain_harness.LangChainHarness`.
    """
    from genai_tk.agents.harness.langchain_harness import LangChainHarness

    prepare_docgraph_profile(profile, db_path=db_path, folder_id=folder_id, extra_skill_dirs=extra_skill_dirs)
    tools = create_document_graph_tools_from_config(db_path, embeddings_id=embeddings_id)
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
