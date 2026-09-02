"""Middleware that nudges the agent to map a document before over-searching.

The Document Graph agent sometimes fires ``search_sections`` repeatedly without
ever looking at a document's table of contents. The FinanceBench Phase 2 eval
showed this blind-search loop drives both token cost and wrong answers. This
middleware counts consecutive ``search_sections`` calls and, once the streak
exceeds a threshold with no intervening orienting call, injects a
``SystemMessage`` nudge to call ``get_document_toc`` first.

Orienting tools (which reset the streak)::

    get_folder_toc, get_document_toc, get_section_content, list_documents

Example YAML config::

    middlewares:
      - class: genai_graph.agent.middleware.map_before_search.MapBeforeSearchMiddleware
        max_consecutive: 3
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from loguru import logger

_SEARCH_TOOL = "search_sections"
_ORIENTING_TOOLS = frozenset({"get_folder_toc", "get_document_toc", "get_section_content", "list_documents"})
_NUDGE = (
    "You have called search_sections several times in a row without orienting. "
    "Before searching again, call get_document_toc to map the most relevant "
    "document's sections, then read the specific section with get_section_content. "
    "Reserve search_sections for when you genuinely do not know where the answer "
    "lives, and prefer reading one grounded section over re-searching."
)


def _count_search_streak(messages: list[BaseMessage]) -> int:
    """Count trailing ``search_sections`` calls with no orienting tool between them."""
    streak = 0
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage):
            tool_calls = msg.tool_calls or []
            if not tool_calls:
                break
            names = {tc.get("name") for tc in tool_calls}
            if names & _ORIENTING_TOOLS:
                break
            streak += sum(1 for tc in tool_calls if tc.get("name") == _SEARCH_TOOL)
            continue
        break
    return streak


class MapBeforeSearchMiddleware(AgentMiddleware):
    """Nudge the agent to map a document after a run of blind search_sections calls.

    Args:
        max_consecutive: Trigger the nudge once the streak of consecutive
            ``search_sections`` calls (with no orienting tool in between) exceeds
            this many. Default 3.
    """

    def __init__(self, max_consecutive: int = 3) -> None:
        self._max_consecutive = max_consecutive

    def _maybe_nudge(self, request: ModelRequest) -> ModelRequest:
        streak = _count_search_streak(request.messages)
        if streak > self._max_consecutive:
            logger.warning(
                "[MapBeforeSearch] {} consecutive search_sections calls with no "
                "orienting tool — nudging get_document_toc",
                streak,
            )
            return request.override(messages=[*request.messages, SystemMessage(content=_NUDGE)])
        return request

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Call handler, injecting a map-first nudge when the search streak is too long."""
        return await handler(self._maybe_nudge(request))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync variant — same logic as async."""
        return handler(self._maybe_nudge(request))
