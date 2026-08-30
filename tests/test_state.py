"""Tests for AgentState."""

import pytest

from agentforge_x.kernel.state import (
    AgentState,
    AIMessage,
    ArtifactRef,
    BudgetState,
    FleetEvent,
    PlanEntry,
)


class TestAgentState:
    """Test AgentState creation and transitions."""

    def test_state_creation_with_all_fields(self):
        """Test creating an AgentState with all fields populated."""
        msg = AIMessage(role="user", content="Hello")
        step = PlanEntry(id="step_0", instruction="Do something", tool="bash", args={"cmd": "echo"})
        artifact = ArtifactRef(id="art_1", type="code", path="/tmp/file.py")
        budget = BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000)
        fleet_event = FleetEvent(ts=0.0, run_id="run_1", agent_id="agent_1", type="test", payload={})

        state = AgentState(
            agent_id="agent_1",
            run_id="run_1",
            parent_agent_id="parent_1",
            messages=[msg],
            plan=[step],
            scratchpad="test scratchpad",
            artifacts=[artifact],
            fleet_events=[fleet_event],
            status="idle",
            budget=budget,
            created_at=0.0,
            updated_at=0.0,
        )

        assert state.agent_id == "agent_1"
        assert state.run_id == "run_1"
        assert state.parent_agent_id == "parent_1"
        assert len(state.messages) == 1
        assert len(state.plan) == 1
        assert state.scratchpad == "test scratchpad"
        assert len(state.artifacts) == 1
        assert len(state.fleet_events) == 1
        assert state.status == "idle"

    def test_state_transition_validation(self):
        """Test valid and invalid state transitions."""
        state = AgentState()

        # Valid transitions
        state.transition("planning")
        assert state.status == "planning"

        state.transition("executing")
        assert state.status == "executing"

        state.transition("critiquing")
        assert state.status == "critiquing"

        state.transition("retrying")
        assert state.status == "retrying"

        state.transition("completed")
        assert state.status == "completed"

        # Cannot transition from completed
        with pytest.raises(ValueError):
            state.transition("planning")

    def test_budget_state_enforcement(self):
        """Test budget enforcement logic."""
        state = AgentState()

        # Test runtime budget (method on AgentState, not BudgetState)
        assert state.check_budget_runtime(500.0) is True
        assert state.check_budget_runtime(700.0) is False

        # Consume runtime
        assert state.consume_runtime(100.0) is True
        assert state.budget.elapsed == 100.0

        # Consume LLM calls
        assert state.consume_llm_call() is True
        assert state.budget.llm_calls == 1

        # Consume tokens
        assert state.consume_tokens(500) is True
        assert state.budget.tokens_used == 500

        # Exceed budget
        state.budget.llm_calls = 100
        assert state.consume_llm_call() is False
        assert state.budget.halted is True
        assert state.budget.halt_reason == "llm_calls"

    def test_serialization_round_trip(self):
        """Test serialization and deserialization of AgentState."""
        state = AgentState(
            agent_id="agent_1",
            run_id="run_1",
            messages=[AIMessage(role="user", content="Hello")],
            plan=[PlanEntry(id="step_0", instruction="Do X", tool="bash", args={})],
            budget=BudgetState(max_runtime=300.0, max_llm_calls=50, max_tokens=50000),
        )

        data = state.to_dict()
        restored = AgentState.from_dict(data)

        assert restored.agent_id == state.agent_id
        assert restored.run_id == state.run_id
        assert len(restored.messages) == 1
        assert len(restored.plan) == 1
        assert restored.budget.max_runtime == 300.0
        assert restored.budget.max_llm_calls == 50
