"""adapters.py — LangGraph ToolNode-compatible bindings.

Provides:
  - ToolNodeCompatibleTool: wraps an MCP tool for LangGraph use
  - to_langgraph_tools: convert MCP tools to LangGraph-compatible list
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from langchain_core.tools import Tool as LCTool

from agentforge_x.tools.mcp_pool import MCPToolClientPool, MCPTool

log = logging.getLogger("agentforge_x.tools.adapters")


class ToolNodeCompatibleTool:
    """Wraps an MCP tool as a LangGraph-compatible tool.

    Implements the LangChain tool interface (name, description, args_schema,
    _run, _arun) so it can be added to a LangGraph ToolNode.
    """

    def __init__(self, mcp_tool: MCPTool, pool: MCPToolClientPool):
        self.mcp_tool = mcp_tool
        self.pool = pool
        self.name = mcp_tool.name
        self.description = mcp_tool.description

    def _run(self, **kwargs: Any) -> str:
        """Synchronous run."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                future = ex.submit(
                    lambda: asyncio.new_event_loop().run_until_complete(
                        self.pool.call_tool(self.name, kwargs)
                    )
                )
                return future.result().text
        else:
            result = asyncio.get_event_loop().run_until_complete(
                self.pool.call_tool(self.name, kwargs)
            )
            return result.text

    async def _arun(self, **kwargs: Any) -> str:
        """Asynchronous run."""
        result = await self.pool.call_tool(self.name, kwargs)
        return result.text


def to_langgraph_tools(
    pool: MCPToolClientPool,
    tool_filter: Optional[list[str]] = None,
) -> list[ToolNodeCompatibleTool]:
    """Convert MCP tools from a pool to LangGraph ToolNode-compatible tools.

    Args:
        pool: MCPToolClientPool instance
        tool_filter: Optional list of tool names to include

    Returns:
        List of ToolNodeCompatibleTool instances
    """
    async def _collect():
        all_tools = await pool.list_tools()
        result = []
        for t in all_tools:
            if tool_filter and t.name not in tool_filter:
                continue
            result.append(ToolNodeCompatibleTool(t, pool))
        return result

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            return ex.submit(lambda: asyncio.run(_collect())).result()
    else:
        return asyncio.run(_collect())
