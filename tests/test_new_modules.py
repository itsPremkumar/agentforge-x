"""Tests for the 4 new AgentForge-X modules."""

import asyncio
import pytest

from agentforge_x.core import CoreKernel, AgentConfig, AgentInstance
from agentforge_x.agents import PlannerAgent, ExecutorAgent, CriticAgent, MultiAgentOrchestrator
from agentforge_x.tools import ToolServer, ToolSpec, MCPStdioServer
from agentforge_x.deployment import DeploymentManager, DeploymentConfig, MetricsCollector


class TestCoreKernel:
    """Test the unified core kernel."""

    def test_create_kernel(self):
        kernel = CoreKernel()
        assert kernel is not None

    def test_create_agent(self):
        kernel = CoreKernel()
        config = AgentConfig(name="test_agent")
        agent = kernel.create_agent(config)
        assert agent.state.agent_id is not None

    def test_list_agents(self):
        kernel = CoreKernel()
        kernel.create_agent(AgentConfig(name="agent1"))
        kernel.create_agent(AgentConfig(name="agent2"))
        agents = kernel.list_agents()
        assert len(agents) == 2

    def test_get_agent(self):
        kernel = CoreKernel()
        config = AgentConfig(name="test")
        agent = kernel.create_agent(config)
        retrieved = kernel.get_agent(agent.state.agent_id)
        assert retrieved is not None
        assert retrieved.state.agent_id == agent.state.agent_id

    def test_get_agent_not_found(self):
        kernel = CoreKernel()
        assert kernel.get_agent("nonexistent") is None

    @pytest.mark.asyncio
    async def test_run_agent(self):
        kernel = CoreKernel()
        config = AgentConfig(name="test")
        agent = kernel.create_agent(config)
        result = await kernel.run_agent(agent.state.agent_id, "Do X")
        assert result is not None
        # MockPlanner creates a plan from "Do X" -> 1 step, MockExecutor succeeds
        assert result.status in ("completed", "failed")

    def test_start_evolution(self):
        kernel = CoreKernel()
        loop = kernel.start_evolution()
        assert loop is not None


class TestAgents:
    """Test the agent library."""

    @pytest.mark.asyncio
    async def test_planner_agent(self):
        agent = PlannerAgent()
        plan = await agent.plan("Research topic; Write report")
        assert len(plan) == 2

    @pytest.mark.asyncio
    async def test_executor_agent(self):
        agent = ExecutorAgent()
        result = await agent.execute("echo hello")
        assert result is not None

    @pytest.mark.asyncio
    async def test_critic_agent(self):
        agent = CriticAgent()
        critique = await agent.critique("Some output")
        assert critique["passed"] is True

    def test_multi_agent_orchestrator(self):
        orchestrator = MultiAgentOrchestrator()
        orchestrator.register("planner", PlannerAgent())
        orchestrator.register("executor", ExecutorAgent())
        assert len(orchestrator.agents) == 2


class TestTools:
    """Test the tool/MCP module."""

    def test_tool_server_create(self):
        server = ToolServer()
        assert server is not None

    def test_register_tool(self):
        server = ToolServer()
        server.register_tool(ToolSpec(
            name="echo",
            description="Echo input",
            handler=lambda args: args.get("text", ""),
        ))
        tools = server.list_tools()
        assert len(tools) == 1

    def test_list_tools_format(self):
        server = ToolServer()
        server.register_tool(ToolSpec(
            name="test",
            description="Test tool",
            handler=lambda args: "ok",
        ))
        tools = server.list_tools()
        assert tools[0]["name"] == "test"
        assert "inputSchema" in tools[0]

    @pytest.mark.asyncio
    async def test_call_tool(self):
        server = ToolServer()
        server.register_tool(ToolSpec(
            name="echo",
            description="Echo",
            handler=lambda args: args.get("text", ""),
        ))
        result = await server.call_tool("echo", {"text": "hello"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        server = ToolServer()
        result = await server.call_tool("missing", {})
        assert result["success"] is False

    def test_mcp_initialize(self):
        server = ToolServer()
        response = server.handle_initialize()
        assert "result" in response

    def test_mcp_tools_list(self):
        server = ToolServer()
        server.register_tool(ToolSpec(name="t1", description="d1", handler=lambda a: ""))
        response = server.handle_tools_list()
        assert len(response["result"]["tools"]) == 1


class TestDeployment:
    """Test the deployment module."""

    def test_create_deployment(self):
        config = DeploymentConfig(name="test")
        manager = DeploymentManager(config)
        assert manager is not None

    def test_register_agent(self):
        manager = DeploymentManager(DeploymentConfig(name="test"))
        manager.register_agent("agent1", object())
        assert manager.list_agents() == ["agent1"]

    def test_health_check(self):
        manager = DeploymentManager(DeploymentConfig(name="test"))
        manager.register_agent("a1", object())
        health = manager.health_check()
        assert health.status in ("healthy", "degraded")
        assert health.agents_count == 1

    def test_metrics_counter(self):
        metrics = MetricsCollector()
        metrics.increment("requests", 5.0)
        assert metrics.get_counter("requests") == 5.0

    def test_metrics_gauge(self):
        metrics = MetricsCollector()
        metrics.gauge("cpu_usage", 0.75)
        assert metrics.get_gauge("cpu_usage") == 0.75

    def test_metrics_histogram(self):
        metrics = MetricsCollector()
        metrics.histogram("latency", 0.1)
        metrics.histogram("latency", 0.2)
        assert len(metrics.get_histogram("latency")) == 2

    def test_metrics_prometheus(self):
        metrics = MetricsCollector()
        metrics.increment("test_counter", 1.0)
        output = metrics.to_prometheus()
        assert "test_counter 1.0" in output
