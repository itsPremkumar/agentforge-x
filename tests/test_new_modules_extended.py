"""Extended tests for the 4 new AgentForge-X modules."""

import asyncio
import pytest

from agentforge_x.core import CoreKernel, AgentConfig, AgentInstance
from agentforge_x.agents import PlannerAgent, ExecutorAgent, CriticAgent, MultiAgentOrchestrator, AgentSpec
from agentforge_x.tools import ToolServer, ToolSpec, MCPStdioServer
from agentforge_x.deployment import DeploymentManager, DeploymentConfig, MetricsCollector, HealthStatus


class TestCoreExtended:
    """Extended tests for CoreKernel."""

    def test_agent_instance_creation(self):
        config = AgentConfig(name="test")
        agent = AgentInstance(config)
        assert agent.status == "idle"

    def test_agent_config_defaults(self):
        config = AgentConfig(name="test")
        assert config.max_retries == 3
        assert config.timeout == 60.0

    @pytest.mark.asyncio
    async def test_agent_run(self):
        config = AgentConfig(name="test")
        agent = AgentInstance(config)
        result = await agent.run("Do X")
        assert result is not None

    @pytest.mark.asyncio
    async def test_kernel_run_agent_not_found(self):
        kernel = CoreKernel()
        result = await kernel.run_agent("nonexistent", "task")
        assert result is None


class TestAgentsExtended:
    """Extended tests for agent library."""

    def test_agent_spec(self):
        spec = AgentSpec(name="planner", role="planner")
        assert spec.name == "planner"
        assert spec.role == "planner"

    @pytest.mark.asyncio
    async def test_planner_with_tools(self):
        agent = PlannerAgent(tools=["bash", "fs"])
        plan = await agent.plan("Step 1; Step 2")
        assert len(plan) >= 1

    @pytest.mark.asyncio
    async def test_executor_with_tools(self):
        from agentforge_x.kernel.executor import ToolRegistry
        registry = ToolRegistry()
        registry.register("echo", lambda args: args.get("text", ""))
        agent = ExecutorAgent(tools=registry)
        result = await agent.execute("echo hello")
        assert result is not None

    @pytest.mark.asyncio
    async def test_critic_with_output(self):
        agent = CriticAgent()
        critique = await agent.critique("Test output")
        assert "passed" in critique

    def test_orchestrator_register_and_list(self):
        orch = MultiAgentOrchestrator()
        orch.register("a", PlannerAgent())
        orch.register("b", ExecutorAgent())
        assert len(orch.agents) == 2

    @pytest.mark.asyncio
    async def test_orchestrator_parallel(self):
        orch = MultiAgentOrchestrator()
        planner = PlannerAgent()
        orch.register("planner", planner)
        # PlannerAgent doesn't have run(), so use execute_parallel with the correct method
        results = await orch.execute_parallel({"planner": "plan task"})
        assert "planner" in results


class TestToolsExtended:
    """Extended tests for tool/MCP module."""

    def test_tool_spec(self):
        spec = ToolSpec(name="test", description="A test tool")
        assert spec.name == "test"

    def test_tool_server_list_empty(self):
        server = ToolServer()
        assert server.list_tools() == []

    @pytest.mark.asyncio
    async def test_call_async_tool(self):
        async def async_handler(args):
            return f"async: {args.get('x', '')}"

        server = ToolServer()
        server.register_tool(ToolSpec(
            name="async_tool",
            description="Async tool",
            handler=async_handler,
        ))
        result = await server.call_tool("async_tool", {"x": "hello"})
        assert result["success"] is True
        assert "async: hello" in result["result"]

    @pytest.mark.asyncio
    async def test_call_tool_error(self):
        def bad_handler(args):
            raise ValueError("Tool error")

        server = ToolServer()
        server.register_tool(ToolSpec(
            name="bad",
            description="Bad tool",
            handler=bad_handler,
        ))
        result = await server.call_tool("bad", {})
        assert result["success"] is False
        assert "error" in result

    def test_mcp_initialize_format(self):
        server = ToolServer()
        response = server.handle_initialize()
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert "serverInfo" in response["result"]

    def test_mcp_tools_list_format(self):
        server = ToolServer()
        server.register_tool(ToolSpec(name="t", description="d", handler=lambda a: ""))
        response = server.handle_tools_list()
        assert "tools" in response["result"]

    @pytest.mark.asyncio
    async def test_mcp_tools_call_format(self):
        server = ToolServer()
        server.register_tool(ToolSpec(name="t", description="d", handler=lambda a: "ok"))
        response = await server.handle_tools_call("t", {})
        assert "result" in response
        assert "content" in response["result"]


class TestDeploymentExtended:
    """Extended tests for deployment module."""

    def test_deployment_config_defaults(self):
        config = DeploymentConfig(name="test")
        assert config.port == 8080
        assert config.host == "0.0.0.0"

    def test_health_status(self):
        status = HealthStatus(
            status="healthy",
            version="0.1.0",
            uptime_seconds=100.0,
            agents_count=2,
        )
        assert status.status == "healthy"

    def test_metrics_with_labels(self):
        metrics = MetricsCollector()
        metrics.increment("req", labels={"method": "GET"})
        assert metrics.get_counter("req", {"method": "GET"}) == 1.0

    def test_metrics_histogram_stats(self):
        metrics = MetricsCollector()
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            metrics.histogram("latency", v)
        hist = metrics.get_histogram("latency")
        assert len(hist) == 5

    def test_deployment_manager_metrics(self):
        manager = DeploymentManager(DeploymentConfig(name="test"))
        manager.metrics.increment("test_metric")
        assert manager.metrics.get_counter("test_metric") == 1.0

    def test_deployment_health_with_no_agents(self):
        manager = DeploymentManager(DeploymentConfig(name="test"))
        health = manager.health_check()
        assert health.agents_count == 0
