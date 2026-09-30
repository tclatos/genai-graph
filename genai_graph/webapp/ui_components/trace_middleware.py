"""Streamlit trace middleware for LangChain agents.

This module provides reusable middleware and UI components for capturing and displaying
tool execution traces in Streamlit applications. It enables real-time visibility into
agent tool calls, their arguments, results, and timing information.

Key Components:
    - ToolCallRecord: Data class representing a single tool execution
    - LLMCallRecord: Data class representing a single LLM message
    - TraceMiddleware: Generic middleware for capturing tool execution details
    - display_tool_traces: UI function to render tool traces in Streamlit
    - display_llm_traces: UI function to render LLM traces in Streamlit
    - display_interleaved_traces: UI function to render both traces interleaved chronologically

Usage Example:
    ```
    from genai_graph.webapp.ui_components.trace_middleware import (
        TraceMiddleware,
        display_interleaved_traces,
    )

    # Create middleware instance (typically stored in session state)
    if "trace_middleware" not in st.session_state:
        st.session_state.trace_middleware = TraceMiddleware()

    # Use middleware when creating agent
    agent = create_agent(
        model=llm,
        tools=tools,
        middleware=[st.session_state.trace_middleware],
    )

    # Display traces (interleaved is recommended for better flow understanding)
    display_interleaved_traces(st.session_state.trace_middleware)
    ```

Design Notes:
    - The middleware is stateful and accumulates tool calls across agent runs
    - Call `clear()` to reset the trace history when needed
    - Both sync and async tool calls are supported
    - The UI component handles long results with truncation and expandable sections
"""

import contextlib
from collections.abc import Callable
from datetime import datetime
from typing import Any

import streamlit as st
from langchain.agents.middleware import AgentMiddleware
from pydantic import BaseModel, Field
from streamlit.delta_generator import DeltaGenerator


class ToolCallRecord(BaseModel):
    """Record of a single tool call execution.

    Captures all relevant information about a tool invocation including
    its arguments, result, timing, and any errors that occurred.

    Attributes:
        name: Name of the tool that was called
        arguments: String representation of the tool arguments/query
        result: String representation of the tool result (None if not completed or errored)
        error: Error message if the tool call failed (None if successful)
        start_time: Timestamp when the tool call started
        end_time: Timestamp when the tool call completed (None if still running)

    Example:
        ```
        record = ToolCallRecord(
            name="search_documents",
            arguments="{'query': 'machine learning'}",
        )
        # After execution
        record.result = "Found 5 documents..."
        record.end_time = datetime.now()
        ```
    """

    name: str
    arguments: str
    result: str | None = None
    error: str | None = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None

    @property
    def duration_ms(self) -> float | None:
        """Calculate execution duration in milliseconds.

        Returns:
            Duration in milliseconds, or None if tool is still running
        """
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000

    @property
    def is_error(self) -> bool:
        """Check if this tool call resulted in an error."""
        return self.error is not None

    @property
    def is_complete(self) -> bool:
        """Check if this tool call has completed (success or error)."""
        return self.end_time is not None

    @property
    def formatted_time(self) -> str:
        """Get formatted start time string (HH:MM:SS.mmm)."""
        return self.start_time.strftime("%H:%M:%S.%f")[:-3]


class LLMCallRecord(BaseModel):
    """Record of a single LLM message emitted during agent execution.

    This is intentionally lightweight: we focus on which graph node produced
    the message, its content, and when it was observed. Duration is typically
    less meaningful for streamed LLM outputs, so we omit it here.
    """

    node: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def formatted_time(self) -> str:
        """Get formatted timestamp string (HH:MM:SS.mmm)."""
        return self.timestamp.strftime("%H:%M:%S.%f")[:-3]


# Callback type for real-time trace updates
TraceCallback = Callable[[str, Any], None]


class TraceMiddleware(AgentMiddleware):
    """Middleware to capture tool and LLM execution details for display in Streamlit.

    This middleware intercepts tool calls made by LangChain agents, recording
    their arguments, results, timing, and any errors. The captured data can
    then be displayed using the `display_tool_traces` function.

    The middleware maintains a list of `ToolCallRecord` objects that persists
    across multiple agent invocations. Use `clear()` to reset the history.

    Attributes:
        tool_calls: List of completed tool call records
        current_call: The tool call currently being executed (None if idle)
        on_event: Optional callback for real-time UI updates

    Thread Safety:
        This middleware is NOT thread-safe. Each Streamlit session should have
        its own middleware instance stored in session state.

    Example:
        ```
        # Initialize in session state
        if "trace_middleware" not in st.session_state:
            st.session_state.trace_middleware = TraceMiddleware()

        middleware = st.session_state.trace_middleware

        # Create agent with middleware
        agent = create_agent(
            model=llm,
            tools=tools,
            middleware=[middleware],
        )

        # After agent run, access captured traces
        for call in middleware.tool_calls:
            print(f"{call.name}: {call.duration_ms}ms")
        ```
    """

    def __init__(self, on_event: TraceCallback | None = None) -> None:
        """Initialize the trace middleware with empty state.

        Args:
            on_event: Optional callback invoked on each trace event.
                      Signature: (event_type: str, data: Any) -> None
                      Event types: 'tool_start', 'tool_end', 'tool_error', 'llm_call'
        """
        self.tool_calls: list[ToolCallRecord] = []
        self.current_call: ToolCallRecord | None = None
        # Lightweight storage for LLM calls, populated explicitly by the UI layer.
        # We keep this generic to avoid coupling to any particular LangChain internals
        # for model tracing while still co-locating traces with tool calls.
        self.llm_calls: list[LLMCallRecord] = []
        self._on_event: TraceCallback | None = on_event

    def clear(self) -> None:
        """Clear all captured tool and LLM call records.

        Call this method to reset the trace history, typically when starting
        a new conversation or when the user requests to clear the trace display.
        """
        self.tool_calls = []
        self.current_call = None
        self.llm_calls = []

    def set_callback(self, callback: TraceCallback | None) -> None:
        """Set or update the real-time event callback.

        Args:
            callback: Callback function or None to disable
        """
        self._on_event = callback

    def _notify(self, event_type: str, data: Any) -> None:
        """Notify the callback of a trace event if registered."""
        if self._on_event is not None:
            with contextlib.suppress(Exception):
                # Don't let callback errors disrupt agent execution
                self._on_event(event_type, data)

    def add_llm_call(self, node: str, content: str) -> None:
        """Record a single LLM message associated with a graph node.

        The UI layer is responsible for calling this when it observes new
        `AIMessage` instances emitted from the graph while streaming.
        """
        record = LLMCallRecord(
            node=node,
            content=content,
        )
        self.llm_calls.append(record)
        self._notify("llm_call", record)

    def _extract_tool_info(self, request: Any) -> tuple[str, str]:
        """Extract tool name and arguments from a tool call request.

        Args:
            request: The tool call request object from LangChain

        Returns:
            Tuple of (tool_name, arguments_string)
        """
        tool_call = getattr(request, "tool_call", {})
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {})
        return tool_name, str(tool_args)

    def _extract_result(self, result: Any) -> str:
        """Extract a string representation from a tool result.

        Args:
            result: The tool execution result

        Returns:
            String representation of the result
        """
        if hasattr(result, "content") and result.content:
            return str(result.content)
        return str(result)

    def _start_call(self, request: Any) -> ToolCallRecord:
        """Create and register a new tool call record.

        Args:
            request: The tool call request object

        Returns:
            The newly created ToolCallRecord
        """
        tool_name, tool_args = self._extract_tool_info(request)
        self.current_call = ToolCallRecord(
            name=tool_name,
            arguments=tool_args,
        )
        self._notify("tool_start", self.current_call)
        return self.current_call

    def _complete_call(self, result: Any) -> None:
        """Mark the current call as successfully completed.

        Args:
            result: The tool execution result
        """
        if self.current_call is not None:
            self.current_call.result = self._extract_result(result)
            self.current_call.end_time = datetime.now()
            self.tool_calls.append(self.current_call)
            completed_call = self.current_call
            self.current_call = None
            self._notify("tool_end", completed_call)

    def _fail_call(self, error: Exception) -> None:
        """Mark the current call as failed with an error.

        Args:
            error: The exception that occurred during tool execution
        """
        if self.current_call is not None:
            self.current_call.error = str(error)
            self.current_call.end_time = datetime.now()
            self.tool_calls.append(self.current_call)
            failed_call = self.current_call
            self.current_call = None
            self._notify("tool_error", failed_call)

    def wrap_tool_call(self, request: Any, handler: Callable) -> Any:
        """Wrap a synchronous tool call to capture execution details.

        This method is called by the LangChain agent framework for each
        synchronous tool invocation. It records the tool call details
        before and after execution.

        Args:
            request: The tool call request containing tool name and arguments
            handler: The actual tool execution function to call

        Returns:
            The result from the tool execution

        Raises:
            Any exception raised by the tool is re-raised after being recorded
        """
        self._start_call(request)

        try:
            result = handler(request)
            self._complete_call(result)
            return result
        except Exception as e:
            self._fail_call(e)
            raise

    async def awrap_tool_call(self, request: Any, handler: Callable) -> Any:
        """Wrap an asynchronous tool call to capture execution details.

        This method is called by the LangChain agent framework for each
        asynchronous tool invocation. It records the tool call details
        before and after execution.

        Args:
            request: The tool call request containing tool name and arguments
            handler: The actual async tool execution function to call

        Returns:
            The result from the tool execution

        Raises:
            Any exception raised by the tool is re-raised after being recorded
        """
        self._start_call(request)

        try:
            result = await handler(request)
            self._complete_call(result)
            return result
        except Exception as e:
            self._fail_call(e)
            raise


def display_tool_traces(
    middleware: TraceMiddleware,
    container: DeltaGenerator | None = None,
    max_result_length: int = 2000,
    show_clear_button: bool = True,
    expand_latest: bool = True,
) -> None:
    """Display tool execution traces in a Streamlit UI.

    Renders a list of tool call records with expandable details including
    tool name, arguments, results, timing, and error information.

    Args:
        middleware: The TraceMiddleware instance containing tool call records
        container: Optional Streamlit container to render into (defaults to current context)
        max_result_length: Maximum characters to display for results before truncating
        show_clear_button: Whether to show a button to clear the trace history
        expand_latest: Whether to auto-expand the most recent tool call

    Example:
        ```
        # In sidebar
        with st.sidebar:
            display_tool_traces(st.session_state.trace_middleware)

        # In a column with custom settings
        col1, col2 = st.columns(2)
        with col1:
            display_tool_traces(
                middleware,
                max_result_length=500,
                show_clear_button=False,
            )
        ```
    """
    ctx = container or st

    if not middleware.tool_calls:
        ctx.info("No tool calls yet. Send a message to see tool interactions!")
        return

    # Header with optional clear button
    if show_clear_button:
        col1, col2 = ctx.columns([3, 1])
        with col1:
            ctx.subheader("Tool Execution Trace")
        with col2:
            if ctx.button("Clear", key="clear_traces", help="Clear execution traces"):
                middleware.clear()
                st.rerun()
    else:
        ctx.subheader("Tool Execution Trace")

    # Summary count
    count = len(middleware.tool_calls)
    ctx.caption(f"{count} tool call{'s' if count != 1 else ''}")

    # Display each tool call in an expandable section
    for i, call in enumerate(middleware.tool_calls):
        _render_tool_call_expander(
            ctx,  # pyright: ignore[reportArgumentType]
            call,
            index=i,
            total=count,
            expand_latest=expand_latest,
            max_result_length=max_result_length,
        )


def _render_tool_call_expander(
    ctx: DeltaGenerator,
    call: ToolCallRecord,
    index: int,
    total: int,
    expand_latest: bool,
    max_result_length: int,
) -> None:
    """Render a single tool call as an expandable section.

    Args:
        ctx: Streamlit context to render into
        call: The tool call record to display
        index: Index of this call in the list (0-based)
        total: Total number of tool calls
        expand_latest: Whether to auto-expand if this is the latest call
        max_result_length: Maximum characters before truncating results
    """
    status_emoji = "❌" if call.is_error else "✅"

    # Build expander title with truncated query
    max_query_len = 30
    truncated_query = call.arguments
    if len(truncated_query) > max_query_len:
        truncated_query = truncated_query[:max_query_len] + "..."

    title = f"{status_emoji} [{call.formatted_time}] {call.name}: {truncated_query}"
    is_latest = index == total - 1

    with ctx.expander(title, expanded=(expand_latest and is_latest)):
        # Tool name
        ctx.markdown(f"**Tool:** `{call.name}`")

        # Timing information
        if call.formatted_time:
            timing_text = f"**Time:** {call.formatted_time}"
            if call.duration_ms is not None:
                timing_text += f" (duration: {call.duration_ms:.1f}ms)"
            ctx.markdown(timing_text)

        # Arguments
        ctx.markdown("**Arguments:**")
        ctx.code(call.arguments, language="text")

        # Result or Error
        if call.is_error:
            ctx.markdown("**Error:**")
            ctx.error(call.error)
        else:
            ctx.markdown("**Result:**")
            result_text = call.result or "No result returned"

            if len(result_text) > max_result_length:
                truncated = result_text[:max_result_length] + "\n\n... (truncated)"
                ctx.code(truncated, language="text")
                ctx.caption(f"Result was {len(result_text)} characters (truncated for display)")
            else:
                ctx.code(result_text, language="text")


def display_interleaved_traces(
    middleware: TraceMiddleware,
    container: DeltaGenerator | None = None,
    key_prefix: str = "main",
    show_full_trace: bool = False,
    show_clear_button: bool = True,
    max_result_length: int = 500,
) -> None:
    """Display LLM and tool traces interleaved chronologically.

    Shows execution traces in chronological order, interleaving LLM calls
    and tool calls for better understanding of the agent's execution flow.

    Args:
        middleware: The TraceMiddleware instance containing trace records
        container: Optional Streamlit container to render into (defaults to current context)
        key_prefix: Unique prefix for Streamlit element keys to avoid duplicates
                   when this function is called multiple times in the same run.
        show_full_trace: Whether to show the collapsible full trace summary
        show_clear_button: Whether to show a button to clear traces
        max_result_length: Maximum characters before showing result in a text area

    Example:
        ```
        from genai_graph.webapp.ui_components.trace_middleware import (
            TraceMiddleware,
            display_interleaved_traces,
        )

        # Display traces in main area
        display_interleaved_traces(st.session_state.trace_middleware)

        # Or with custom options
        display_interleaved_traces(
            middleware,
            key_prefix="streaming",
            show_full_trace=False,
        )
        ```
    """
    ctx = container or st

    # Generate a unique call ID to avoid duplicate keys when called multiple times
    # Use a module-level counter stored in session state
    if "_trace_display_counter" not in st.session_state:
        st.session_state._trace_display_counter = 0
    st.session_state._trace_display_counter += 1
    call_id = st.session_state._trace_display_counter
    unique_prefix = f"{key_prefix}_{call_id}"

    # Collect all events
    events = []

    # Add LLM calls
    if hasattr(middleware, "llm_calls") and middleware.llm_calls:
        for call in middleware.llm_calls:
            events.append({"type": "llm", "timestamp": call.timestamp, "data": call})

    # Add tool calls
    if middleware.tool_calls:
        for call in middleware.tool_calls:
            events.append({"type": "tool", "timestamp": call.start_time, "data": call})

    if not events:
        ctx.info("No activity yet. Send a message to see LLM and tool interactions!")
        return

    # Sort by timestamp to interleave
    events.sort(key=lambda x: x["timestamp"])

    # Optional: collapsible section to view full trace
    if show_full_trace:
        with ctx.expander("📋 View Full Trace (All Events)", expanded=False):
            for i, event in enumerate(events):
                if event["type"] == "llm":
                    call = event["data"]
                    ctx.markdown(f"**{i + 1}. 🧠 LLM Call** - {call.formatted_time}")
                    ctx.markdown(f"   - **Node:** `{call.node}`")
                    with ctx.container(border=True):
                        ctx.markdown(call.content)
                elif event["type"] == "tool":
                    call = event["data"]
                    status = "✅ Success" if not call.is_error else "❌ Error"
                    duration = f" ({call.duration_ms:.0f}ms)" if call.duration_ms else ""
                    ctx.markdown(f"**{i + 1}. 🔧 Tool Call** - {call.formatted_time}{duration}")
                    ctx.markdown(f"   - **Tool:** `{call.name}` - {status}")
                    if call.arguments:
                        ctx.markdown(f"   - **Arguments:** `{call.arguments}`")
                    if call.result:
                        with ctx.container(border=True):
                            result_preview = call.result[:200] + "..." if len(call.result) > 200 else call.result
                            ctx.text(result_preview)
                    if call.error:
                        ctx.error(f"   - **Error:** {call.error}")
                ctx.divider()

    # Display individual events as separate expanders (latest expanded)
    for i, event in enumerate(events):
        if event["type"] == "llm":
            call = event["data"]
            with ctx.expander(
                f"🧠 LLM Call - {call.node} ({call.formatted_time})",
                expanded=(i == len(events) - 1 and event["type"] == "llm"),
            ):
                ctx.markdown(call.content)

        elif event["type"] == "tool":
            call = event["data"]
            status = "✅" if not call.is_error else "❌"
            with ctx.expander(
                f"🔧 {call.name} {status}",
                expanded=(i == len(events) - 1 and event["type"] == "tool"),
            ):
                ctx.markdown(f"**Status:** {'Error' if call.is_error else 'Success'}")

                if call.arguments:
                    ctx.markdown("**Arguments:**")
                    ctx.code(call.arguments, language="text")

                if call.result:
                    ctx.markdown("**Result:**")
                    result = call.result
                    if isinstance(result, str) and len(result) > max_result_length:
                        ctx.text_area(
                            "Output",
                            result,
                            height=150,
                            disabled=True,
                            key=f"{unique_prefix}_tool_result_{i}",
                        )
                    else:
                        ctx.code(str(result), language="text")

                if call.error:
                    ctx.markdown("**Error:**")
                    ctx.error(call.error)

    # Optional clear button
    if show_clear_button:
        _col1, col2 = ctx.columns([3, 1])
        with col2:
            if ctx.button("🗑️ Clear Traces", key=f"{unique_prefix}_clear_btn", width="stretch"):
                middleware.clear()
                st.rerun()


class StreamingTraceRenderer:
    """Real-time trace renderer for use inside st.status containers.

    This class provides a callback-based approach to render trace events
    as they occur during agent execution, giving users immediate visibility
    into what the agent is doing.

    Usage:
        ```
        middleware = TraceMiddleware()
        with st.status("🤖 Agent is thinking...", expanded=True) as status:
            renderer = StreamingTraceRenderer(status)
            middleware.set_callback(renderer.on_event)

            # Run agent - traces will appear in real-time
            async for step in agent.astream(inputs, config):
                # Process steps...
                if isinstance(latest_message, AIMessage) and latest_message.content:
                    middleware.add_llm_call(node, latest_message.content)

            status.update(label="✅ Complete!", state="complete")
        ```
    """

    def __init__(self, status_container: DeltaGenerator, max_result_preview: int = 300) -> None:
        """Initialize the streaming renderer.

        Args:
            status_container: The st.status container to render into
            max_result_preview: Max characters for result preview
        """
        self._status = status_container
        self._max_result_preview = max_result_preview
        self._event_count = 0

    def on_event(self, event_type: str, data: Any) -> None:
        """Handle trace events and render them to the status container.

        Args:
            event_type: One of 'tool_start', 'tool_end', 'tool_error', 'llm_call'
            data: The event data (ToolCallRecord or LLMCallRecord)
        """
        self._event_count += 1

        if event_type == "tool_start":
            self._render_tool_start(data)
        elif event_type == "tool_end":
            self._render_tool_end(data)
        elif event_type == "tool_error":
            self._render_tool_error(data)
        elif event_type == "llm_call":
            self._render_llm_call(data)

    def _render_tool_start(self, call: ToolCallRecord) -> None:
        """Render a tool call start event."""
        self._status.write(f"🔧 **Calling tool:** `{call.name}`")

        # Parse and display arguments, highlighting Cypher queries
        args_str = call.arguments
        cypher_displayed = False

        try:
            import ast

            args_dict = ast.literal_eval(args_str)
            if isinstance(args_dict, dict) and "cypher_query" in args_dict:
                self._status.code(args_dict["cypher_query"], language="cypher")
                cypher_displayed = True
        except (ValueError, SyntaxError):
            pass

        if not cypher_displayed and args_str:
            # Show truncated args if not a Cypher query
            preview = args_str[:150] + "..." if len(args_str) > 150 else args_str
            self._status.caption(f"Args: {preview}")

    def _render_tool_end(self, call: ToolCallRecord) -> None:
        """Render a tool call completion event."""
        duration = f" ({call.duration_ms:.0f}ms)" if call.duration_ms else ""
        self._status.write(f"✅ **Tool `{call.name}` completed**{duration}")

        if call.result:
            result_preview = call.result
            if len(result_preview) > self._max_result_preview:
                result_preview = result_preview[: self._max_result_preview] + f"... ({len(call.result)} chars)"
            self._status.caption(f"Result: {result_preview}")

    def _render_tool_error(self, call: ToolCallRecord) -> None:
        """Render a tool call error event."""
        duration = f" ({call.duration_ms:.0f}ms)" if call.duration_ms else ""
        self._status.write(f"❌ **Tool `{call.name}` failed**{duration}")
        if call.error:
            self._status.error(call.error)

    def _render_llm_call(self, call: LLMCallRecord) -> None:
        """Render an LLM response event."""
        self._status.write(f"🧠 **LLM response** from `{call.node}`")
        # Show a preview of the response
        content_preview = call.content
        if len(content_preview) > self._max_result_preview:
            content_preview = content_preview[: self._max_result_preview] + "..."
        self._status.caption(content_preview)
