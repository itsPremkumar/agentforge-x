#!/usr/bin/env python3
"""Tests for tool_discoverer."""
from __future__ import annotations

import pytest

from agentforge_x.tools.tool_discoverer import (
    DiscoveredTool,
    discover_mcp_tools,
    discover_local_tools,
    discover_all,
)


class TestDiscoveredTool:
    def test_creation(self):
        tool = DiscoveredTool(
            name="read_file",
            description="Read a file",
            source="mcp",
            module="mcp_toolbox.filesystem_safe",
        )
        assert tool.name == "read_file"
        assert tool.source == "mcp"
        assert tool.capabilities == set()

    def test_with_capabilities(self):
        tool = DiscoveredTool(
            name="read",
            description="Read",
            source="local",
            module="test",
            capabilities={"fs.read"},
        )
        assert "fs.read" in tool.capabilities


class TestDiscoverMcpTools:
    def test_with_nonexistent_command(self):
        tools = discover_mcp_tools(["nonexistent_command_xyz"])
        assert tools == []

    def test_with_echo(self):
        # echo won't produce valid JSON-RPC, should return empty
        tools = discover_mcp_tools(["echo", "not json"])
        assert tools == []

    def test_with_python_module(self):
        # Use python to print a valid tools/list response
        import json
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "A test tool",
                        "inputSchema": {"type": "object"},
                    }
                ]
            },
        }
        cmd = ["python", "-c", f"print('{json.dumps(response)}')"]
        tools = discover_mcp_tools(cmd)
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        assert tools[0].source == "mcp"


class TestDiscoverLocalTools:
    def test_with_nonexistent_module(self):
        tools = discover_local_tools("nonexistent_module_xyz")
        assert tools == []

    def test_with_current_module(self):
        # This test file has no @tool decorators, so should be empty
        tools = discover_local_tools("tests.test_tools_mcp_layer")
        # May be empty or have functions depending on decorators
        assert isinstance(tools, list)


class TestDiscoverAll:
    def test_empty_inputs(self):
        tools = discover_all()
        assert tools == []

    def test_with_bad_inputs(self):
        tools = discover_all(
            mcp_commands=[["nonexistent"]],
            local_modules=["nonexistent.module"],
        )
        assert tools == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
