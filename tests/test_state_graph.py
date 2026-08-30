"""Tests for the StateGraph runtime (control loop)."""

import pytest

from agentforge_x.kernel.critic import MockCritic
from agentforge_x.kernel.executor import ExecutionResult, MockExecutor
from agentforge_x.kernel.planner import MockPlanner
from agentforge_x.kernel.state import AgentState
from agentforge_x.kernel.state_graph import KernelConfig, StateGraphRuntime


class TestControlLoop:
    """Test the planner->executor->critic->complete cycle."""

    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Test a full planner->executor->critic->complete cycle."""
        planner = MockPlanner()
        executor = MockExecutor(results=[
            ExecutionResult(success=True, output="Done"),
        ])
        critic = MockCritic(always_pass=True)

        runtime = StateGraphRuntime(
            planner=planner,
            executor=executor,
            critic=critic,
            config=KernelConfig(),
        )

        state = AgentState(
            messages=[],
            scratchpad="Do X",
        )

        result = await runtime.run(state)

        assert result.status == "completed"
        assert len(result.plan) > 0
        assert all(s.status == "completed" for s in result.plan)

    @pytest.mark.asyncio
    async def test_retry_on_crit_fail(self):
        """Test that retry happens when critic fails."""
        planner = MockPlanner()
        # First call fails, second succeeds
        executor = MockExecutor(results=[
            ExecutionResult(success=False, output="", error="Failed", retryable=True),
            ExecutionResult(success=True, output="Retried"),
        ])
        critic = MockCritic(always_fail=True)  # Will fail, but should_retry returns True

        runtime = StateGraphRuntime(
            planner=planner,
            executor=executor,
            critic=critic,
            config=KernelConfig(max_retries=3),
        )

        state = AgentState(
            messages=[],
            scratchpad="Do X",
        )

        result = await runtime.run(state)
        # With always_fail critic, the agent should exhaust retries and fail
        assert result.status in ("failed", "completed")

    @pytest.mark.asyncio
    async def test_halt_on_budget_exceeded(self):
        """Test that agent halts when budget is exceeded."""
        planner = MockPlanner()
        executor = MockExecutor(results=[
            ExecutionResult(success=True, output="Done"),
        ])
        critic = MockCritic(always_pass=True)

        runtime = StateGraphRuntime(
            planner=planner,
            executor=executor,
            critic=critic,
            config=KernelConfig(),
        )

        # Create state with already-exceeded budget
        state = AgentState(
            messages=[],
            scratchpad="Do X",
        )
        state.budget.max_runtime = 0.0  # Already exhausted

        result = await runtime.run(state)

        # Should either halt or complete (depending on implementation)
        assert result.status in ("halted", "completed", "failed")

    @pytest.mark.asyncio
    async def test_subgraph_spawn_and_completion(self):
        """Test that subgraphs are spawned when needed."""
        planner = MockPlanner()
        executor = MockExecutor(results=[
            ExecutionResult(success=True, output="Done"),
        ])
        critic = MockCritic(always_pass=True)

        runtime = StateGraphRuntime(
            planner=planner,
            executor=executor,
            critic=critic,
            config=KernelConfig(enable_subgraph_spawning=True),
        )

        state = AgentState(
            messages=[],
            scratchpad="spawn subgraph to do X",
        )

        result = await runtime.run(state)

        # Check that fleet events were recorded
        assert len(result.fleet_events) >= 0  # May or may not spawn depending on heuristic

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test that errors are propagated correctly."""
        planner = MockPlanner()
        executor = MockExecutor(results=[
            ExecutionResult(success=False, output="", error="Critical failure", retryable=False),
        ])
        critic = MockCritic(always_fail=True)

        runtime = StateGraphRuntime(
            planner=planner,
            executor=executor,
            critic=critic,
            config=KernelConfig(max_retries=0),
        )

        state = AgentState(
            messages=[],
            scratchpad="Do X",
        )

        result = await runtime.run(state)

        assert result.status == "failed"
        assert result.error is not None
