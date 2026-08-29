"""Tests for the critic."""

import pytest

from agentforge_x.kernel.critic import Critic, MockCritic, Critique
from agentforge_x.kernel.executor import ExecutionResult
from agentforge_x.kernel.state import AgentState, PlanEntry


class TestCritic:
    """Test critique evaluation."""

    @pytest.mark.asyncio
    async def test_crit_pass_when_score_above_threshold(self):
        """Test that critic passes when score >= threshold."""
        critic = MockCritic(pass_threshold=0.8, always_pass=True)
        state = AgentState()
        step = PlanEntry(id="step_0", instruction="Do X", tool="bash", args={})
        result = ExecutionResult(success=True, output="Done")

        critique = await critic.evaluate(state, step, result)

        assert critique.passed is True
        assert critique.score >= 0.8

    @pytest.mark.asyncio
    async def test_crit_fail_when_score_below_threshold(self):
        """Test that critic fails when score < threshold."""
        critic = MockCritic(pass_threshold=0.8, always_fail=True)
        state = AgentState()
        step = PlanEntry(id="step_0", instruction="Do X", tool="bash", args={})
        result = ExecutionResult(success=False, output="", error="Failed")

        critique = await critic.evaluate(state, step, result)

        assert critique.passed is False
        assert critique.score < 0.8

    def test_retry_decision_logic(self):
        """Test that retry decision respects retry count."""
        critic = MockCritic(always_pass=True)
        state = AgentState()
        step = PlanEntry(id="step_0", instruction="Do X", tool="bash", args={}, retry_count=0)

        assert critic.should_retry(state, step, "error") is True

        step.retry_count = 5
        assert critic.should_retry(state, step, "error") is False

    @pytest.mark.asyncio
    async def test_progress_evaluation(self):
        """Test plan progress evaluation."""
        critic = MockCritic()
        state = AgentState()
        state.plan = [
            PlanEntry(id="step_0", instruction="A", tool="bash", args={}, status="completed"),
            PlanEntry(id="step_1", instruction="B", tool="bash", args={}, status="completed"),
            PlanEntry(id="step_2", instruction="C", tool="bash", args={}, status="pending"),
        ]

        critique = await critic.evaluate_progress(state)

        assert critique.score == pytest.approx(0.67, abs=0.1)
        assert critique.passed is False  # Below 0.8 threshold

    @pytest.mark.asyncio
    async def test_feedback_quality(self):
        """Test that critiques include meaningful feedback."""
        critic = MockCritic(always_pass=True)
        state = AgentState()
        step = PlanEntry(id="step_0", instruction="Do X", tool="bash", args={})
        result = ExecutionResult(success=True, output="Done")

        critique = await critic.evaluate(state, step, result)

        assert critique.feedback is not None
        assert len(critique.feedback) > 0
