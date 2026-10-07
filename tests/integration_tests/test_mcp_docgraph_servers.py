"""Integration tests for Document Graph MCP 2.0 servers.

Tests the two exposed MCP servers in genai-graph:
1. docgraph-tools: Document Graph navigation tools (list_documents, get_folder_toc, etc.)
2. docgraph-agent: Deep agent query tool (ask_docgraph_agent)
"""

from __future__ import annotations

import pytest
from genai_tk.mcp.config import get_mcp_server_definition, load_mcp_server_definitions
from genai_tk.mcp.server_builder import build_mcp_server
from mcp import Client


@pytest.mark.integration
def test_docgraph_mcp_server_definitions():
    """Verify both docgraph MCP servers are defined and loaded."""
    definitions = load_mcp_server_definitions()
    server_names = {d.name for d in definitions}
    assert "docgraph-tools" in server_names
    assert "docgraph-agent" in server_names


@pytest.mark.integration
@pytest.mark.anyio
async def test_docgraph_tools_mcp_server():
    """Verify Server 1 exposes all Document Graph navigation tools with valid MCP 2.0 schemas."""
    defn = get_mcp_server_definition("docgraph-tools")
    server = build_mcp_server(defn)

    async with Client(server) as client:
        # Check server info
        assert client.server_info is not None
        assert client.server_info.name == "docgraph-tools"

        # Check tools listing
        tools_res = await client.list_tools()
        tools_dict = {t.name: t for t in tools_res.tools}

        expected_tools = {
            "list_documents",
            "get_folder_toc",
            "get_document_toc",
            "get_section_content",
            "search_sections",
            "query_image",
        }
        assert expected_tools.issubset(set(tools_dict.keys()))

        # Verify all schemas have "type": "object" (MCP 2.0 protocol requirement)
        for name, tool_obj in tools_dict.items():
            schema = getattr(tool_obj, "input_schema", getattr(tool_obj, "inputSchema", {}))
            assert schema.get("type") == "object", f"Tool {name} schema must have type: object"

        # Execute list_documents tool call over MCP 2.0 interface
        result = await client.call_tool("list_documents", {})
        assert not result.is_error
        assert len(result.content) > 0


@pytest.mark.integration
@pytest.mark.anyio
async def test_docgraph_agent_mcp_server():
    """Verify Server 2 exposes the Document Graph deep agent as an MCP tool."""
    defn = get_mcp_server_definition("docgraph-agent")
    server = build_mcp_server(defn)

    async with Client(server) as client:
        # Check server info
        assert client.server_info is not None
        assert client.server_info.name == "docgraph-agent"

        # Check tools listing
        tools_res = await client.list_tools()
        names = [t.name for t in tools_res.tools]
        assert "ask_docgraph_agent" in names

        # Verify agent tool schema
        agent_tool = next(t for t in tools_res.tools if t.name == "ask_docgraph_agent")
        schema = getattr(agent_tool, "input_schema", getattr(agent_tool, "inputSchema", {}))
        assert schema.get("type") == "object"
        assert "query" in schema.get("properties", {})
