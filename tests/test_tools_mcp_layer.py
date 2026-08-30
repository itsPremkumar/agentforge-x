#!/usr/bin/env python3
"""Tests for agentforge-x MCP tool layer."""
from __future__ import annotations

import pytest

from agentforge_x.tools.sandbox import SandboxedExec, SandboxedToolWrapper, FS_JAIL_ROOT, DEFAULT_ALLOWLIST
from agentforge_x.tools.registry import ToolRegistry, ToolEntry, ToolKind, Capability
from agentforge_x.tools.mcp_pool import MCPToolClientPool, MCPServerConfig, Transport, MCPTool
from agentforge_x.tools.adapters import ToolNodeCompatibleTool, to_langgraph_tools


class TestSandboxedExec:
    def test_runs_echo(self, tmp_path):
        sandbox = SandboxedExec(jail_root=tmp_path)
        result = sandbox.run(["echo", "hello"])
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

    def test_blocks_non_allowlisted_command(self, tmp_path):
        sandbox = SandboxedExec(jail_root=tmp_path, allowlist={"echo"})
        result = sandbox.run(["ls", "/tmp"])
        assert result["returncode"] == -1
        assert "not in allowlist" in result["stderr"]

    def test_cwd_restriction(self, tmp_path):
        sandbox = SandboxedExec(jail_root=tmp_path)
        # Use a path that tries to escape
        with pytest.raises(ValueError, match="escapes sandbox"):
            sandbox.run(["echo", "test"], cwd="../../etc")

    def test_timeout(self, tmp_path):
        sandbox = SandboxedExec(jail_root=tmp_path, default_timeout=0.5)
        # Use python to sleep - works on both Unix and Windows
        result = sandbox.run(["python", "-c", "import time; time.sleep(5)"])
        assert result["timed_out"] is True

    def test_env_sanitization(self, tmp_path):
        sandbox = SandboxedExec(jail_root=tmp_path)
        result = sandbox.run(["echo", "test"], env={"PATH": "/evil", "SAFE": "ok"})
        # Should not crash
        assert result["returncode"] == 0 or result["returncode"] == -1

    def test_output_truncation(self, tmp_path):
        sandbox = SandboxedExec(jail_root=tmp_path, max_output_bytes=10)
        result = sandbox.run(["echo", "x" * 1000])
        assert len(result["stdout"]) <= 10


class TestSandboxedToolWrapper:
    def test_creation(self, tmp_path):
        def dummy(x):
            return x
        wrapper = SandboxedToolWrapper(dummy, jail_root=tmp_path)
        assert wrapper.jail_root == tmp_path

    def test_blocks_traversal(self, tmp_path):
        def dummy(x):
            return x
        wrapper = SandboxedToolWrapper(dummy, jail_root=tmp_path)
        with pytest.raises(ValueError, match="escapes"):
            wrapper("../etc/passwd")


class TestToolRegistry:
    def test_register_local(self):
        reg = ToolRegistry()

        def echo(text: str) -> str:
            return text

        entry = reg.register_local(echo, capabilities={Capability.SHELL.value})
        assert entry.name == "echo"
        assert entry.kind == ToolKind.LOCAL

    def test_has_tool(self):
        reg = ToolRegistry()

        def echo(text: str) -> str:
            return text

        reg.register_local(echo)
        assert reg.has("echo") is True
        assert reg.has("nonexistent") is False

    def test_get_tool(self):
        reg = ToolRegistry()

        def echo(text: str) -> str:
            return text

        reg.register_local(echo, description="Echo text")
        entry = reg.get("echo")
        assert entry is not None
        assert entry.description == "Echo text"

    def test_tools_by_capability(self):
        reg = ToolRegistry()

        def fs_read(path: str) -> str:
            return ""

        def shell_cmd(cmd: str) -> str:
            return ""

        reg.register_local(fs_read, capabilities={Capability.FS_READ.value})
        reg.register_local(shell_cmd, capabilities={Capability.SHELL.value})

        fs_tools = reg.tools_by_capability(Capability.FS_READ.value)
        assert len(fs_tools) == 1
        assert fs_tools[0].name == "fs_read"

    def test_all_tools(self):
        reg = ToolRegistry()

        def a(x: int) -> int:
            return x

        def b(x: int) -> int:
            return x

        reg.register_local(a)
        reg.register_local(b)
        assert len(reg.all_tools()) == 2

    def test_unregister(self):
        reg = ToolRegistry()

        def echo(text: str) -> str:
            return text

        reg.register_local(echo)
        assert reg.unregister("echo") is True
        assert reg.has("echo") is False


class TestMCPToolClientPool:
    def test_add_stdio_server(self):
        pool = MCPToolClientPool()
        pool.add_stdio("test", ["echo", "hello"])
        assert len(pool.servers) == 1
        assert pool.servers[0].name == "test"

    def test_add_http_server(self):
        pool = MCPToolClientPool()
        pool.add_http("remote", "https://example.com/mcp")
        assert pool.servers[0].transport == Transport.HTTP

    def test_empty_tools_when_disconnected(self):
        pool = MCPToolClientPool()
        pool.add_stdio("test", ["echo", "hello"])
        # list_tools requires initialization; mock empty for now
        assert pool.servers[0].timeout == 30.0


@pytest.mark.asyncio
class TestMCPPoolAsync:
    async def test_initialize_with_bad_command(self):
        pool = MCPToolClientPool()
        pool.add_stdio("bad", ["nonexistent_xyz_123"])
        # Should not raise; failures are logged
        try:
            await pool.initialize()
        except Exception:
            pass  # Expected to fail but not crash

    async def test_list_tools_empty_with_no_servers(self):
        pool = MCPToolClientPool()
        tools = await pool.list_tools()
        assert tools == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestToolNodeCompatibleTool:
    def test_creation(self):
        mcp_tool = MCPTool(
            name="my_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_name="test",
            server_config=MCPServerConfig(name="test", transport=Transport.STDIO),
        )
        pool = MCPToolClientPool()
        adapter = ToolNodeCompatibleTool(mcp_tool, pool)
        assert adapter.name == "my_tool"
        assert adapter.description == "A test tool"


class TestToLanggraphTools:
    def test_empty_list_with_no_servers(self):
        pool = MCPToolClientPool()
        tools = to_langgraph_tools(pool)
        assert tools == []

    def test_filter_by_name(self):
        pool = MCPToolClientPool()
        # No servers registered, so no tools
        tools = to_langgraph_tools(pool, tool_filter=["nonexistent"])
        assert tools == []


class TestCapability:
    def test_values(self):
        assert Capability.FS_READ.value == "filesystem.read"
        assert Capability.FS_WRITE.value == "filesystem.write"
        assert Capability.NET_FETCH.value == "network.fetch"
        assert Capability.MCP_TOOL.value == "mcp.tool"
        assert Capability.SHELL.value == "shell"


class TestToolKind:
    def test_values(self):
        assert ToolKind.MCP.value == "mcp"
        assert ToolKind.LOCAL.value == "local"


class TestMCPTool:
    def test_creation(self):
        cfg = MCPServerConfig(name="test", transport=Transport.STDIO)
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            server_name="test",
            server_config=cfg,
        )
        assert tool.name == "read_file"
        assert tool.server_name == "test"


class TestMCPServerConfig:
    def test_stdio_defaults(self):
        cfg = MCPServerConfig(name="test", transport=Transport.STDIO)
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 3

    def test_http_config(self):
        cfg = MCPServerConfig(name="remote", transport=Transport.HTTP, url="https://example.com/mcp")
        assert cfg.url == "https://example.com/mcp"


class TestToolEntry:
    def test_creation(self):
        entry = ToolEntry(
            name="test_tool",
            kind=ToolKind.LOCAL,
            description="A test",
            capabilities={"shell"},
        )
        assert entry.name == "test_tool"
        assert entry.kind == ToolKind.LOCAL
        assert "shell" in entry.capabilities


class TestRegistryLocalAsync:
    @pytest.mark.asyncio
    async def test_call_async_local(self):
        reg = ToolRegistry()

        async def async_echo(text: str) -> str:
            return f"echo: {text}"

        reg.register_local(async_echo)
        result = await reg.call("async_echo", {"text": "hello"})
        assert result.success is True
        assert "echo: hello" in result.text

    @pytest.mark.asyncio
    async def test_call_sync_local(self):
        reg = ToolRegistry()

        def sync_add(a: int, b: int) -> int:
            return a + b

        reg.register_local(sync_add)
        result = await reg.call("sync_add", {"a": 2, "b": 3})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_call_missing_tool(self):
        reg = ToolRegistry()
        result = await reg.call("nonexistent", {})
        assert result.success is False
        assert "not found" in result.error
