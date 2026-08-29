"""Planner interface and LLM-backed implementation.

The Planner creates and updates execution plans from instructions/goals.
The default implementation (LLMPlanner) can be swapped for a mock in tests.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agentforge_x.kernel.state import AgentState, PlanEntry
from agentforge_x.kernel.event_bus import EventBus, EventType


class Planner:
    """Interface for creating and updating plans."""

    async def create_plan(self, state: AgentState, instructions: str) -> list[PlanEntry]:
        """Create a plan from the agent's current state + instructions."""
        raise NotImplementedError

    async def replan(self, state: AgentState, failed_step_id: Optional[str] = None) -> list[PlanEntry]:
        """Re-plan when the current plan has failed or needs adjustment."""
        raise NotImplementedError

    async def decompose(self, goal: str, context: list) -> list[PlanEntry]:
        """Default plan creation from a high-level goal."""
        raise NotImplementedError

    def estimate(self, plan: list[PlanEntry]) -> dict[str, Any]:
        """Estimate resource requirements for a plan."""
        return {
            "estimated_runtime": len(plan) * 10.0,
            "estimated_llm_calls": len(plan) * 2,
            "estimated_tokens": len(plan) * 500,
        }


class MockPlanner(Planner):
    """Deterministic planner for testing."""

    async def create_plan(self, state: AgentState, instructions: str) -> list[PlanEntry]:
        steps = self._decompose(instructions)
        return steps

    async def replan(self, state: AgentState, failed_step_id: Optional[str] = None) -> list[PlanEntry]:
        steps = list(state.plan)
        if failed_step_id:
            for step in steps:
                if step.id == failed_step_id:
                    step.status = "failed"
        return steps

    async def decompose(self, goal: str, context: list = None) -> list[PlanEntry]:
        return self._decompose(goal)

    def _decompose(self, text: str) -> list[PlanEntry]:
        parts = re.split(r"[;\n]", text)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            parts = [text.strip()]
        return [
            PlanEntry(
                id=f"step_{i}",
                instruction=part,
                tool="mock_tool",
                args={"instruction": part},
                status="pending",
            )
            for i, part in enumerate(parts)
        ]
