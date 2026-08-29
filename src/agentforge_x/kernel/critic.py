"""Critic interface and MockCritic implementation.

The Critic evaluates execution results against criteria and decides
whether to pass, fail, or retry steps.

Interface matches the architectural spec:
  evaluate(state, step, result) -> Critique
  evaluate_progress(state) -> Critique
  should_retry(state, step, failure) -> bool
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from agentforge_x.kernel.state import AgentState, PlanEntry
from agentforge_x.kernel.executor import ExecutionResult
from agentforge_x.kernel.event_bus import EventBus, EventType


class Critique(BaseModel):
    """A critique of a step execution."""

    passed: bool
    score: float = 0.0  # 0.0 to 1.0
    feedback: str = ""
    suggested_fix: Optional[str] = None
    criteria: list[dict[str, Any]] = Field(default_factory=list)


class Critic:
    """Interface for evaluating execution results."""

    async def evaluate(
        self,
        state: AgentState,
        step: PlanEntry,
        result: ExecutionResult,
        events: Optional[EventBus] = None,
    ) -> Critique:
        raise NotImplementedError

    async def evaluate_progress(self, state: AgentState) -> Critique:
        raise NotImplementedError

    def should_retry(self, state: AgentState, step: PlanEntry, failure: str) -> bool:
        raise NotImplementedError


class MockCritic(Critic):
    """Deterministic critic for testing."""

    def __init__(
        self,
        pass_threshold: float = 0.8,
        always_pass: bool = False,
        always_fail: bool = False,
    ):
        self.pass_threshold = pass_threshold
        self.always_pass = always_pass
        self.always_fail = always_fail

    async def evaluate(
        self,
        state: AgentState,
        step: PlanEntry,
        result: ExecutionResult,
        events: Optional[EventBus] = None,
    ) -> Critique:
        if self.always_pass:
            critique = Critique(
                passed=True,
                score=0.95,
                feedback="Step passed evaluation.",
                criteria=[{"name": "correctness", "score": 0.95, "weight": 1.0}],
            )
        elif self.always_fail:
            critique = Critique(
                passed=False,
                score=0.3,
                feedback="Step failed evaluation.",
                suggested_fix="Retry with adjusted parameters.",
                criteria=[{"name": "correctness", "score": 0.3, "weight": 1.0}],
            )
        else:
            score = 1.0 if result.success else 0.3
            critique = Critique(
                passed=score >= self.pass_threshold,
                score=score,
                feedback="Auto-evaluated based on success.",
            )

        if events:
            events.emit(
                EventType.CRITIC_EVAL,
                {
                    "step_id": step.id,
                    "criteria": [c["name"] for c in critique.criteria],
                    "threshold": self.pass_threshold,
                },
                agent_id=state.agent_id,
            )
            if critique.passed:
                events.emit(
                    EventType.CRITIC_PASS,
                    {"step_id": step.id, "score": critique.score, "feedback": critique.feedback},
                    agent_id=state.agent_id,
                )
            else:
                events.emit(
                    EventType.CRITIC_FAIL,
                    {
                        "step_id": step.id,
                        "score": critique.score,
                        "feedback": critique.feedback,
                        "reason": "below_threshold",
                    },
                    agent_id=state.agent_id,
                )

        return critique

    async def evaluate_progress(self, state: AgentState) -> Critique:
        completed = sum(1 for s in state.plan if s.status == "completed")
        total = len(state.plan)
        score = completed / total if total > 0 else 0.0
        return Critique(
            passed=score >= 0.8,
            score=score,
            feedback=f"{completed}/{total} steps completed.",
        )

    def should_retry(self, state: AgentState, step: PlanEntry, failure: str) -> bool:
        if step.retry_count >= 3:
            return False
        if self.always_fail:
            return False
        return True
