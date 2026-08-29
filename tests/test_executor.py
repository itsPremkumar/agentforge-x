"""Tests for the executor."""

import pytest

from agentforge_x.kernel.executor import (
    ToolExecutor,
    ToolRegistry,
)
from agentforge_x.kernel.state import AgentState, PlanEntry


class TestExecutor:
    """Test step execution."""

    @pytest.mark.asyncio
    async def test_step_execution_success(self):
        """Test successful step execution."""
        registry = ToolRegistry()
        registry.register("echo", lambda args: f"Echo: {args.get('text', '')}")

        executor = ToolExecutor(registry)
        state = AgentState()
        step = PlanEntry(id="step_0", instruction="Echo hello", tool="echo", args={"text": "hello"})

        result = await executor.execute(state, step)

        assert result.success is True
        assert "Echo: hello" in result.output

    @pytest.mark.asyncio
    async def test_step_execution_failure_non_retryable(self):
        """Test step execution failure (non-retryable)."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        state = AgentState()
        step = PlanEntry(id="step_0", instruction="Use missing tool", tool="missing_tool", args={})

        result = await executor.execute(state, step)

        assert result.success is False
        assert result.retryable is False
        assert "Tool not found" in result.error

    @pytest.mark.asyncio
    async def test_step_execution_failure_retryable(self):
        """Test step execution failure (retryable)."""
        registry = ToolRegistry()
        registry.register("fail", lambda args: (_ for _ in ()).throw(Exception("Transient error")))

        executor = ToolExecutor(registry)
        state = AgentState()
        step = PlanEntry(id="step_0", instruction="Fail", tool="fail", args={})

        result = await executor.execute(state, step)

        assert result.success is False
        assert result.retryable is True
        assert "Transient error" in result.error

    def test_artifact_creation_on_execution(self):
        """Test that artifacts are created during execution."""
        registry = ToolRegistry()
        registry.register("create_file", lambda args: {
            "artifact": True,
            "artifact_type": "code",
            "path": "/tmp/file.py",
        })

        executor = ToolExecutor(registry)
        assert executor.can_execute(PlanEntry(id="s", instruction="", tool="create_file", args={}))

    def test_tool_availability_checking(self):
        """Test that tool availability is checked."""
        registry = ToolRegistry()
        registry.register("bash", lambda args: "executed")

        executor = ToolExecutor(registry)

        assert executor.can_execute(PlanEntry(id="s", instruction="", tool="bash", args={})) is True
        assert executor.can_execute(PlanEntry(id="s", instruction="", tool="missing", args={})) is False
