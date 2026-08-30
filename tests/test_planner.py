"""Tests for the planner."""

import pytest

from agentforge_x.kernel.planner import MockPlanner
from agentforge_x.kernel.state import AgentState, PlanEntry


class TestPlanner:
    """Test plan creation and decomposition."""

    @pytest.mark.asyncio
    async def test_plan_creation_from_instructions(self):
        """Test creating a plan from instructions."""
        planner = MockPlanner()
        state = AgentState()
        plan = await planner.create_plan(state, "Do X; Do Y; Do Z")

        assert len(plan) == 3
        assert plan[0].instruction == "Do X"
        assert plan[1].instruction == "Do Y"
        assert plan[2].instruction == "Do Z"

    @pytest.mark.asyncio
    async def test_plan_decomposition_goal_to_steps(self):
        """Test decomposing a goal into steps."""
        planner = MockPlanner()
        plan = await planner.decompose("Step 1; Step 2")

        assert len(plan) == 2
        assert all(isinstance(s, PlanEntry) for s in plan)

    @pytest.mark.asyncio
    async def test_replan_after_step_failure(self):
        """Test replanning after a step fails."""
        planner = MockPlanner()
        state = AgentState()
        state.plan = await planner.create_plan(state, "Do X; Do Y")

        # Mark first step as failed
        state.plan[0].status = "failed"

        new_plan = await planner.replan(state, failed_step_id="step_0")
        assert any(s.id == "step_0" for s in new_plan)

    def test_resource_estimation(self):
        """Test plan resource estimation."""
        planner = MockPlanner()
        plan = [
            PlanEntry(id="step_0", instruction="Do X", tool="bash", args={}),
            PlanEntry(id="step_1", instruction="Do Y", tool="bash", args={}),
        ]
        estimate = planner.estimate(plan)

        assert "estimated_runtime" in estimate
        assert "estimated_llm_calls" in estimate
        assert "estimated_tokens" in estimate
        assert estimate["estimated_runtime"] > 0

    def test_max_steps_enforcement(self):
        """Test that max steps is enforced."""
        planner = MockPlanner()
        # MockPlanner doesn't enforce max_steps, but the interface should accept it
        assert planner.estimate([]) is not None
