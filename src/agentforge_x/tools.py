"""AgentForge-X Tool/MCP: tool registry and MCP server.

Provides a unified tool registry with MCP (Model Context Protocol)
server support for exposing tools to LLM agents.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from agentforge_x.kernel.executor import ToolRegistry


@dataclass
class ToolSpec:
    """Specification for a tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    handler: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolServer:
    """MCP-compatible tool server.

    Exposes tools via the Model Context Protocol for LLM agents.
    Supports stdio and HTTP transports.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or ToolRegistry()
        self._tools: dict[str, ToolSpec] = {}
        self._server_info = {
            "name": "agentforge-x",
            "version": "0.1.0",
            "protocol_version": "2024-11-05",
        }

    def register_tool(self, spec: ToolSpec) -> None:
        """Register a tool with the server."""
        self._tools[spec.name] = spec
        if spec.handler:
            self.registry.register(spec.name, spec.handler)

    def list_tools(self) -> list[dict[str, Any]]:
        """List all registered tools in MCP format."""
        return [
            {
                "name": name,
                "description": spec.description,
                "inputSchema": spec.input_schema,
            }
            for name, spec in self._tools.items()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool by name."""
        if name not in self._tools:
            return {"error": f"Tool not found: {name}", "success": False}

        spec = self._tools[name]
        if not spec.handler:
            return {"error": f"Tool {name} has no handler", "success": False}

        try:
            if asyncio.iscoroutinefunction(spec.handler):
                result = await spec.handler(arguments)
            else:
                result = spec.handler(arguments)
            return {"result": str(result), "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def handle_initialize(self) -> dict[str, Any]:
        """Handle MCP initialize request."""
        return {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "result": {
                "protocolVersion": self._server_info["protocol_version"],
                "serverInfo": {
                    "name": self._server_info["name"],
                    "version": self._server_info["version"],
                },
                "capabilities": {
                    "tools": {"listChanged": True},
                },
            },
        }

    def handle_tools_list(self) -> dict[str, Any]:
        """Handle MCP tools/list request."""
        return {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "result": {"tools": self.list_tools()},
        }

    async def handle_tools_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP tools/call request."""
        result = await self.call_tool(name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "result": {"content": [{"type": "text", "text": json.dumps(result)}]},
        }


class MCPStdioServer:
    """MCP server over stdio transport."""

    def __init__(self, tool_server: ToolServer):
        self.tool_server = tool_server

    async def start(self) -> None:
        """Start the stdio server."""
        import sys
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            try:
                request = json.loads(line)
                response = await self._handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                continue

    async def _handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle an MCP request."""
        method = request.get("method", "")
        if method == "initialize":
            return self.tool_server.handle_initialize()
        elif method == "tools/list":
            return self.tool_server.handle_tools_list()
        elif method == "tools/call":
            params = request.get("params", {})
            return await self.tool_server.handle_tools_call(
                params.get("name", ""),
                params.get("arguments", {}),
            )
        return {"error": f"Unknown method: {method}"}


__all__ = [
    "ToolSpec",
    "ToolServer",
    "MCPStdioServer",
]
