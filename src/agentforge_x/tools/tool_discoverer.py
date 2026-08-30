"""tool_discoverer.py — discover tools from MCP servers and local modules.

Provides:
  - discover_mcp_tools: discover tools from MCP servers via stdio
  - discover_local_tools: discover tools from Python modules
  - discover_all: unified discovery from both sources
"""
from __future__ import annotations

import importlib
import inspect
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("agentforge_x.tools.discoverer")


@dataclass
class DiscoveredTool:
    """A tool discovered from any source."""
    name: str
    description: str
    source: str  # "mcp" or "local"
    module: str
    capabilities: set[str] = field(default_factory=set)
    input_schema: dict[str, Any] = field(default_factory=dict)


def discover_mcp_tools(
    command: list[str],
    env: Optional[dict[str, str]] = None,
    timeout: float = 10.0,
) -> list[DiscoveredTool]:
    """Discover tools from an MCP server via stdio JSON-RPC.

    Args:
        command: MCP server command (e.g., ["mcp-toolbox-filesystem-safe"])
        env: Environment variables for the subprocess
        timeout: Timeout in seconds

    Returns:
        List of DiscoveredTool instances
    """
    try:
        # Send tools/list request
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }
        req_str = json.dumps(req) + "\n"

        proc = subprocess.run(
            command,
            input=req_str,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**__import__("os").environ, **(env or {})},
        )

        if proc.returncode != 0:
            log.warning("MCP server exited with %d: %s", proc.returncode, proc.stderr[:500])
            return []

        # Parse response
        stdout = proc.stdout.strip()
        if not stdout:
            return []

        try:
            response = json.loads(stdout.split("\n")[0])
        except json.JSONDecodeError:
            log.warning("Non-JSON output from MCP server: %s", stdout[:200])
            return []

        tools = []
        for t in response.get("result", {}).get("tools", []):
            tools.append(DiscoveredTool(
                name=t.get("name", ""),
                description=t.get("description", ""),
                source="mcp",
                module=command[0] if command else "unknown",
                input_schema=t.get("inputSchema", {}),
            ))
        return tools

    except subprocess.TimeoutExpired:
        log.warning("MCP server timed out after %.1fs", timeout)
        return []
    except Exception as exc:
        log.warning("Failed to discover MCP tools: %s", exc)
        return []


def discover_local_tools(
    module_name: str,
    predicate: Optional[callable] = None,
) -> list[DiscoveredTool]:
    """Discover tools from a Python module.

    Finds functions decorated with @tool or matching a predicate.

    Args:
        module_name: Python module to inspect (e.g., "mcp_toolbox.filesystem_safe")
        predicate: Optional function to filter tools (func) -> bool

    Returns:
        List of DiscoveredTool instances
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        log.warning("Failed to import %s: %s", module_name, exc)
        return []

    tools = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        # Skip private/internal
        if name.startswith("_"):
            continue

        # Check for tool marker
        is_tool = getattr(obj, "_agentforge_tool", False)
        if predicate:
            is_tool = predicate(obj)
        if not is_tool:
            continue

        # Extract metadata
        meta = getattr(obj, "_agentforge_tool_meta", {})
        description = meta.get("description", obj.__doc__ or "").strip().split("\n")[0]
        capabilities = meta.get("capabilities", set())

        # Build schema from signature
        sig = inspect.signature(obj)
        properties = {}
        required = []
        for pname, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            ptype = "string"
            if param.annotation is int:
                ptype = "integer"
            elif param.annotation is float:
                ptype = "number"
            elif param.annotation is bool:
                ptype = "boolean"
            properties[pname] = {"type": ptype}
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        input_schema = {"type": "object", "properties": properties}
        if required:
            input_schema["required"] = required

        tools.append(DiscoveredTool(
            name=meta.get("name", name),
            description=description,
            source="local",
            module=module_name,
            capabilities=capabilities if isinstance(capabilities, set) else set(capabilities),
            input_schema=input_schema,
        ))

    return tools


def discover_all(
    mcp_commands: Optional[list[list[str]]] = None,
    local_modules: Optional[list[str]] = None,
) -> list[DiscoveredTool]:
    """Discover tools from all sources.

    Args:
        mcp_commands: List of MCP server commands
        local_modules: List of Python module names

    Returns:
        Combined list of DiscoveredTool instances
    """
    tools = []

    for cmd in mcp_commands or []:
        tools.extend(discover_mcp_tools(cmd))

    for mod in local_modules or []:
        tools.extend(discover_local_tools(mod))

    return tools
