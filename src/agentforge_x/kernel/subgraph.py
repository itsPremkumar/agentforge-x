"""Subgraph spawning and management."""

from __future__ import annotations

import uuid

from agentforge_x.kernel.budget import RUNTIME_OVERHEAD_PER_SPAWN, BudgetAllocation, BudgetEnforcer
from agentforge_x.kernel.event_bus import EventBus, EventType
from agentforge_x.kernel.state import AgentState, AIMessage, BudgetState, FleetEvent


class SubgraphSpawner:
    """Manages spawning of subgraph agents with budget enforcement."""

    def __init__(self, budget: BudgetEnforcer | None = None):
        self.budget = budget

    def can_spawn(self, state: AgentState, allocation: BudgetAllocation) -> bool:
        """Check if spawning is allowed under the parent state's budget."""
        remaining_runtime = state.budget.max_runtime - state.budget.elapsed - RUNTIME_OVERHEAD_PER_SPAWN
        remaining_llm = state.budget.max_llm_calls - state.budget.llm_calls
        remaining_tokens = state.budget.max_tokens - state.budget.tokens_used

        return (
            remaining_runtime >= allocation.max_runtime
            and remaining_llm >= allocation.max_llm_calls
            and remaining_tokens >= allocation.max_tokens
        )

    def spawn(
        self,
        state: AgentState,
        goal: str,
        allocation: BudgetAllocation,
        events: EventBus | None = None,
    ) -> tuple[str, AgentState]:
        """Spawn a new subgraph agent.

        Args:
            state: Parent agent state.
            goal: Goal for the subgraph agent.
            allocation: Budget allocation for the subgraph.
            events: Optional event bus.

        Returns: (subgraph_agent_id, subgraph_state)
        """
        if not self.can_spawn(state, allocation):
            if events:
                events.emit(
                    EventType.BUDGET_EXCEEDED,
                    {"reason": "subgraph_spawn_denied", "goal": goal},
                    agent_id=state.agent_id,
                )
            raise RuntimeError("Insufficient budget to spawn subgraph")

        subgraph_agent_id = f"sub-{uuid.uuid4().hex[:8]}"

        # Deduct overhead from parent state's budget
        state.budget.elapsed += RUNTIME_OVERHEAD_PER_SPAWN

        subgraph_budget = BudgetState(
            max_runtime=allocation.max_runtime,
            max_llm_calls=allocation.max_llm_calls,
            max_tokens=allocation.max_tokens,
        )

        subgraph_state = AgentState(
            agent_id=subgraph_agent_id,
            run_id=state.run_id,
            parent_agent_id=state.agent_id,
            messages=[AIMessage(role="user", content=goal)],
            budget=subgraph_budget,
        )

        # Record fleet event in parent
        fleet_event = FleetEvent(
            ts=state.budget.elapsed,
            run_id=state.run_id,
            agent_id=state.agent_id,
            type=EventType.SUBGRAPH_SPAWN,
            payload={
                "subgraph_agent_id": subgraph_agent_id,
                "goal": goal,
                "budget_allocation": allocation.model_dump(),
            },
        )
        state.fleet_events.append(fleet_event)

        if events:
            events.emit(
                EventType.SUBGRAPH_SPAWN,
                {
                    "subgraph_agent_id": subgraph_agent_id,
                    "parent_agent_id": state.agent_id,
                    "goal": goal,
                    "budget_allocation": allocation.model_dump(),
                },
                agent_id=state.agent_id,
            )

        return subgraph_agent_id, subgraph_state

    def complete_subgraph(
        self,
        parent_state: AgentState,
        subgraph_agent_id: str,
        result: str,
        events: EventBus | None = None,
    ) -> AgentState:
        """Record a subgraph's completion in the parent state."""
        fleet_event = FleetEvent(
            ts=parent_state.budget.elapsed,
            run_id=parent_state.run_id,
            agent_id=parent_state.agent_id,
            type=EventType.SUBGRAPH_COMPLETE,
            payload={
                "subgraph_agent_id": subgraph_agent_id,
                "status": "completed",
                "exit_state": {"result": result[:500]},
            },
        )
        parent_state.fleet_events.append(fleet_event)

        if events:
            events.emit(
                EventType.SUBGRAPH_COMPLETE,
                {
                    "subgraph_agent_id": subgraph_agent_id,
                    "status": "completed",
                    "exit_state": {"result": result[:500]},
                },
                agent_id=parent_state.agent_id,
            )

        return parent_state
