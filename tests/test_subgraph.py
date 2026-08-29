"""Tests for subgraph spawning and management."""

import pytest

from agentforge_x.kernel.subgraph import SubgraphSpawner
from agentforge_x.kernel.budget import BudgetEnforcer, BudgetAllocation, RUNTIME_OVERHEAD_PER_SPAWN
from agentforge_x.kernel.event_bus import EventBus, EventType
from agentforge_x.kernel.state import AgentState, BudgetState


class TestSubgraph:
    """Test subgraph spawning and completion."""

    def test_subgraph_spawn_with_budget_allocation(self):
        """Test spawning a subgraph with budget allocation."""
        parent_state = AgentState(
            agent_id="parent_1",
            run_id="run_1",
            messages=[],
            budget=BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000),
        )
        spawner = SubgraphSpawner()

        allocation = BudgetAllocation(max_runtime=100.0, max_llm_calls=10, max_tokens=5000)
        subgraph_id, subgraph_state = spawner.spawn(
            parent_state,
            goal="Do subgraph task",
            allocation=allocation,
        )

        assert subgraph_id.startswith("sub-")
        assert subgraph_state.parent_agent_id == "parent_1"
        assert subgraph_state.run_id == "run_1"
        assert subgraph_state.budget.max_runtime == 100.0

    def test_subgraph_budget_exceeded_raises_error(self):
        """Test that exceeding budget raises an error."""
        parent_state = AgentState(
            agent_id="parent_1",
            run_id="run_1",
            messages=[],
            budget=BudgetState(max_runtime=10.0, max_llm_calls=5, max_tokens=1000),
        )
        spawner = SubgraphSpawner()

        allocation = BudgetAllocation(max_runtime=100.0, max_llm_calls=50, max_tokens=50000)

        with pytest.raises(RuntimeError, match="Insufficient budget"):
            spawner.spawn(parent_state, goal="Too expensive", allocation=allocation)

    def test_subgraph_completion_records_fleet_event(self):
        """Test that subgraph completion records a fleet event."""
        parent_state = AgentState(
            agent_id="parent_1",
            run_id="run_1",
            messages=[],
            budget=BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000),
        )
        spawner = SubgraphSpawner()

        allocation = BudgetAllocation(max_runtime=100.0, max_llm_calls=10, max_tokens=5000)
        subgraph_id, _ = spawner.spawn(parent_state, goal="Do task", allocation=allocation)

        spawner.complete_subgraph(parent_state, subgraph_id, "Task completed")

        assert len(parent_state.fleet_events) == 2  # spawn + complete
        assert parent_state.fleet_events[0].type == EventType.SUBGRAPH_SPAWN
        assert parent_state.fleet_events[1].type == EventType.SUBGRAPH_COMPLETE

    def test_subgraph_spawn_emits_events(self):
        """Test that subgraph spawn emits events to the bus."""
        parent_state = AgentState(
            agent_id="parent_1",
            run_id="run_1",
            messages=[],
            budget=BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000),
        )
        spawner = SubgraphSpawner()
        events = EventBus(run_id="run_1", agent_id="parent_1")

        allocation = BudgetAllocation(max_runtime=100.0, max_llm_calls=10, max_tokens=5000)
        spawner.spawn(parent_state, goal="Do task", allocation=allocation, events=events)

        spawn_events = events.get_events(EventType.SUBGRAPH_SPAWN)
        assert len(spawn_events) == 1

    def test_multiple_subgraph_spawns(self):
        """Test spawning multiple subgraphs."""
        parent_state = AgentState(
            agent_id="parent_1",
            run_id="run_1",
            messages=[],
            budget=BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000),
        )
        spawner = SubgraphSpawner()

        allocation = BudgetAllocation(max_runtime=50.0, max_llm_calls=5, max_tokens=2000)

        for i in range(3):
            spawner.spawn(parent_state, goal=f"Task {i}", allocation=allocation)

        assert len(parent_state.fleet_events) == 3
        # Each spawn deducts overhead
        assert parent_state.budget.elapsed == 3 * RUNTIME_OVERHEAD_PER_SPAWN
