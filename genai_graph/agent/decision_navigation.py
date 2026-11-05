"""Tool factory for decision-model-navigated agents (DecisionRAG study).

Every extra kwarg passed by the profile's ``factory:`` tool entry (via
``UnifiedToolSpec``/``FactoryToolSpec`` → ``obj(**params)``) flows straight into
:class:`DecisionNavigatorConfig`, so all navigator tuning lives in the agent
profile YAML — no bespoke config section and no change to
:func:`genai_graph.agent.docgraph_agent.create_docgraph_agent` are needed:

```yaml
tools:
  - factory: genai_graph.agent.decision_navigation.create_decision_navigation_tools
    decision_model: fast
    beam_width: 3
    domain_context_file: ${paths.project}/skills/custom/officeqa-qa/decision-routing.md
```

The emitted toolset is ``navigate_folder`` (one-shot decision-model navigation
returning an evidence pack) plus the standard read-only graph tools, which the
synthesis agent keeps for re-reads, visual questions and the low-confidence
LLM-driven fallback (study Option C).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from genai_graph.agent.decision_navigator import DecisionNavigator, DecisionNavigatorConfig
from genai_graph.agent.docgraph_agent import resolve_db_path
from genai_graph.kg.query.document_graph_tools import create_document_graph_tools

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def create_decision_navigation_tools(
    db_path: str | None = None,
    *,
    profile: str = "default",
    embeddings_id: str | None = None,
    max_image_queries: int = 3,
    **navigator_config: Any,
) -> list[BaseTool]:
    """Build the decision-navigated toolset: ``navigate_folder`` + fallback graph tools.

    Args:
        db_path: Path to the Ladybug database. Resolved from
            ``docgraph_profiles.<profile>.paths.kg_db`` when None (same resolution
            order as :func:`genai_graph.agent.docgraph_agent.create_docgraph_tools`).
        profile: DocGraph profile name for database resolution (default: 'default').
        embeddings_id: Embeddings model id enabling the hybrid ``search_sections``
            fallback mode.
        max_image_queries: Maximum ``query_image`` VLM calls allowed per turn.
        **navigator_config: Everything else maps onto :class:`DecisionNavigatorConfig`
            fields (``decision_model``, ``beam_width``, ``top_r_sections``,
            ``min_confidence``, ``max_depth``, ``candidate_cap``, ``prefilter``,
            ``include_summaries``, ``domain_context``, ``answerable_gate``,
            ``retries``) plus ``domain_context_file`` — a file path whose content
            is loaded as ``domain_context`` (keeps long domain routing knowledge
            in a file).

    Returns:
        ``[navigate_folder, get_folder_toc, get_document_toc, get_section_content,
        search_sections, query_image, list_documents]`` tools.
    """
    from langchain_core.tools import tool

    known = DecisionNavigatorConfig.model_fields.keys()
    config = DecisionNavigatorConfig.model_validate({k: v for k, v in navigator_config.items() if k in known})
    if config.domain_context is None:
        context_file = navigator_config.get("domain_context_file")
        if context_file:
            config.domain_context = Path(str(context_file)).expanduser().read_text(encoding="utf-8")
            logger.info("Loaded navigator domain context from {}", context_file)

    resolved_db = resolve_db_path(db_path, profile=profile)
    navigator = DecisionNavigator(config, db_path=resolved_db)

    @tool("navigate_folder")
    def _navigate_folder(query: str, folder_id: str | None = None, hints: str | None = None) -> str:
        """Navigate the Document Graph with a decision model and return an evidence pack.

        Scores every document (and then every section of the top documents) in a
        single joint pass each, fetches the selected sections' full text, and
        returns them with relevance scores. No LLM is used for navigation.

        A transient failure is retried (``retries`` attempts beyond the first)
        before the tool gives up, so one flaky call does not end navigation for a
        whole run: the fallback instruction below is a last resort, not a first
        response.

        Args:
            query: The question to ground (state verbatim, not keywords).
            folder_id: Optional folder scope (defaults to the whole corpus).
            hints: Optional refinement hints describing what a previous pack was
                missing; they re-weight the next navigation pass.
        """
        attempts = max(1, navigator.config.retries + 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                pack = navigator.navigate(query, folder_id, hints=hints)
            except Exception as exc:
                # Retried, then reported to the agent as text.
                last_error = exc
                logger.warning(
                    "navigate_folder attempt {}/{} failed — {}: {}",
                    attempt,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
                continue
            if attempt > 1:
                logger.info("navigate_folder recovered on attempt {}/{}", attempt, attempts)
            # Best-score-first inclusion under the configured budget; oversized
            # sections become get_section_content pointers instead of evicted tool
            # results.
            return pack.render(navigator.config.pack_char_budget)

        error = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
        return (
            f"Error: navigation failed after {attempts} attempt(s) ({error}). "
            "Fall back to `get_folder_toc` / `get_document_toc` / `search_sections` directly."
        )

    fallback_tools = create_document_graph_tools(
        resolved_db, embeddings_id=embeddings_id, max_image_queries=max_image_queries
    )
    return [_navigate_folder, *fallback_tools]
