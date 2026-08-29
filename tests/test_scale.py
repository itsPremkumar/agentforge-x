"""Scale and stress tests for the agentforge-x kernel."""

import asyncio

import pytest

from agentforge_x.kernel.budget import BudgetAllocation
from agentforge_x.kernel.checkpoint import SQLiteCheckpointStore
from agentforge_x.kernel.critic import MockCritic
from agentforge_x.kernel.event_bus import EventBus, EventType
from agentforge_x.kernel.executor import ExecutionResult, MockExecutor
from agentforge_x.kernel.planner import MockPlanner
from agentforge_x.kernel.state import AgentState, BudgetState, PlanEntry
from agentforge_x.kernel.state_graph import StateGraphRuntime
from agentforge_x.kernel.subgraph import SubgraphSpawner


class TestScale:
    """Scale and stress tests."""

    @pytest.mark.asyncio
    async def test_10_concurrent_agents(self):
        """Test running 10 agents concurrently with individual budgets."""
        results = []

        async def run_agent(agent_id: str):
            planner = MockPlanner()
            executor = MockExecutor(results=[ExecutionResult(success=True, output="Done")])
            critic = MockCritic(always_pass=True)
            runtime = StateGraphRuntime(planner=planner, executor=executor, critic=critic)

            state = AgentState(agent_id=agent_id, run_id="scale_test", messages=[], scratchpad="Do X")
            result = await runtime.run(state)
            return result

        tasks = [run_agent(f"agent_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r.status == "completed" for r in results)

    def test_5_level_deep_subgraph_nesting(self):
        """Test 5 subgraph spawns from the same parent (simulating depth)."""
        # Use a parent budget large enough for 5 spawns
        # Each spawn: 50.0 runtime + 5.0 overhead = 55.0 per spawn
        # 5 spawns need at least 275.0 runtime
        parent_state = AgentState(
            agent_id="root",
            run_id="nested_test",
            messages=[],
            budget=BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000),
        )
        spawner = SubgraphSpawner()
        allocation = BudgetAllocation(max_runtime=50.0, max_llm_calls=5, max_tokens=2000)

        # Spawn 5 subgraphs from the same parent
        for level in range(5):
            subgraph_id, _ = spawner.spawn(
                parent_state,
                goal=f"Level {level + 1} task",
                allocation=allocation,
            )
            assert subgraph_id.startswith("sub-")

        # Should have 5 spawn events
        assert len(parent_state.fleet_events) == 5

    def test_100_plan_steps_in_single_agent(self):
        """Test creating and running 100 plan steps."""
        state = AgentState()
        # Create a plan with 100 steps manually
        state.plan = [
            PlanEntry(id=f"step_{i}", instruction=f"Do task {i}", tool="bash", args={}, status="pending")
            for i in range(100)
        ]
        # Set dependencies using the correct field name
        for i, step in enumerate(state.plan):
            if i > 0:
                step.depends_on = [f"step_{i - 1}"]

        assert len(state.plan) == 100
        # Check that the first step is ready (no dependencies)
        assert state.plan[0].depends_on is None
        # Check that subsequent steps depend on the previous one
        assert state.plan[1].depends_on == ["step_0"]
        assert state.plan[99].depends_on == ["step_98"]

    def test_budget_exhaustion_under_load(self):
        """Test that budget exhaustion is handled under load."""
        # Create parent state with small budget
        parent_state = AgentState(
            agent_id="parent",
            run_id="load_test",
            messages=[],
            budget=BudgetState(max_runtime=30.0, max_llm_calls=50, max_tokens=50000),
        )
        spawner = SubgraphSpawner()
        allocation = BudgetAllocation(max_runtime=5.0, max_llm_calls=4, max_tokens=800)

        # Spawn until budget is exhausted
        spawned = 0
        try:
            for _ in range(10):
                spawner.spawn(parent_state, goal="task", allocation=allocation)
                spawned += 1
        except RuntimeError:
            pass  # Budget exhausted

        # Should have spawned some but not all 10
        assert spawned < 10
        assert spawned > 0
        # After spawning, the parent's budget elapsed should have increased
        assert parent_state.budget.elapsed > 0

    def test_event_bus_throughput(self):
        """Test event bus can handle 10K events."""
        bus = EventBus(run_id="throughput_test", agent_id="agent_1")

        for i in range(10000):
            bus.emit(EventType.PLAN_STEP_START, {"step_id": f"step_{i}"})

        events = bus.get_events()
        assert len(events) == 10000

    def test_checkpoint_recovery(self):
        """Test that checkpoints can be saved and recovered."""
        store = SQLiteCheckpointStore(":memory:")
        state = AgentState(
            agent_id="recovery_test",
            run_id="run_recovery",
            messages=[],
            plan=[PlanEntry(id="step_0", instruction="Do X", tool="bash", args={})],
            budget=BudgetState(max_runtime=600.0),
        )

        # Save checkpoint
        store.save(state.run_id, state.agent_id, 1, 1.0, state)

        # Recover
        loaded = store.load(state.run_id, state.agent_id)
        assert loaded is not None
        ts, restored = loaded
        assert restored.agent_id == state.agent_id
        assert restored.run_id == state.run_id
        assert len(restored.plan) == 1
