"""Middleware that forces convergence before the step limit is reached.

Deep agents occasionally enter runaway loops: after a hard rejection (e.g. the
``query_image`` budget) they keep firing *varied* tool calls — each with fresh
arguments, so duplicate-call middlewares never trigger — until the LangGraph
``recursion_limit`` kills the run. The MMLongBench smoke run showed a single
such question burning 2.9M input tokens (88% of a run) and ending in a
``GraphRecursionError`` with no answer at all.

This middleware escalates in two steps as total tool calls accumulate:

- **soft_limit**: appends a ``SystemMessage`` telling the agent to stop
  exploring and converge on its final answer.
- **hard_limit**: strips ALL tools from the model request (via
  ``request.override(tools=[])``) and injects a mandatory final-answer
  instruction. With no tools advertised, the model must emit text, so the run
  always produces an answer instead of dying at the recursion limit. Any tool
  call still emitted is rejected at the tool-call boundary.

Defaults (24 / 32) leave 3–8x headroom over observed healthy runs (3–16 tool
calls per question) while catching loops at roughly 1/5 of a
``recursion_limit=160`` budget.

Example YAML config::

    middlewares:
      - class: genai_graph.agent.middleware.wrap_up.WrapUpMiddleware
        soft_limit: 24
        hard_limit: 32
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from loguru import logger


def _count_tool_calls(messages: list[BaseMessage]) -> int:
    """Count the total number of tool calls recorded in the conversation so far."""
    total = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            total += len(msg.tool_calls or [])
    return total


class WrapUpMiddleware(AgentMiddleware):
    """Force the agent to converge before the recursion limit is reached.

    Args:
        soft_limit: Total tool-call count at which a "wrap up now" nudge is
            injected into every subsequent model request. Default 24.
        hard_limit: Total tool-call count at which all tools are stripped from
            the model request and a mandatory final-answer instruction is
            injected. Must be greater than ``soft_limit``. Default 32.
    """

    def __init__(self, soft_limit: int = 24, hard_limit: int = 32) -> None:
        if hard_limit <= soft_limit:
            raise ValueError(f"hard_limit ({hard_limit}) must be greater than soft_limit ({soft_limit})")
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit

    def _converge(self, request: ModelRequest) -> ModelRequest:
        """Return an overridden request once the soft/hard call threshold is hit."""
        total = _count_tool_calls(request.messages)
        if total < self._soft_limit:
            return request

        if total >= self._hard_limit:
            logger.warning(
                "[WrapUp] {} tool calls >= hard limit {} — stripping all tools, forcing final answer",
                total,
                self._hard_limit,
            )
            directive = SystemMessage(
                content=(
                    f"MANDATORY: The tool budget is exhausted ({total} tool calls). All tools are now "
                    "disabled. Based ONLY on the evidence already gathered in this conversation, "
                    "deliver your complete final answer to the original question in plain text now. "
                    "Do not mention the tool budget."
                )
            )
            return request.override(messages=[*request.messages, directive], tools=[])

        logger.warning(
            "[WrapUp] {} tool calls >= soft limit {} — nudging convergence",
            total,
            self._soft_limit,
        )
        nudge = SystemMessage(
            content=(
                f"WRAP UP: You have made {total} tool calls and are approaching the step limit "
                f"(hard cutoff at {self._hard_limit}). Stop exploring. Within at most a couple of "
                "calls, deliver your final answer based on the evidence already gathered. Do not "
                "repeat searches or re-read sections you have already seen."
            )
        )
        return request.override(messages=[*request.messages, nudge])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync variant — inject wrap-up nudges / strip tools near the limit."""
        return handler(self._converge(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async variant — same logic as sync."""
        return await handler(self._converge(request))
