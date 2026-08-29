"""StateGraph runtime — the main control loop.

Orchestrates the planner -> executor -> critic -> retry loop.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentforge_x.kernel.budget import RUNTIME_OVERHEAD_PER_SPAWN
from agentforge_x.kernel.checkpoint import SQLiteCheckpointStore
from agentforge_x.kernel.critic import Critic, MockCritic
from agentforge_x.kernel.event_bus import EventBus, EventType
from agentforge_x.kernel.executor import Executor, ToolExecutor, ToolRegistry
from agentforge_x.kernel.planner import MockPlanner, Planner
from agentforge_x.kernel.state import AgentState, PlanEntry
from agentforge_x.kernel.subgraph import SubgraphSpawner


class KernelConfig(BaseModel):
    """Configuration for the StateGraph runtime."""

    max_retries: int = 3
    max_plan_steps: int = 100
    enable_subgraph_spawning: bool = True
    checkpoint_interval: int = 1
    budget_overhead_per_spawn: float = RUNTIME_OVERHEAD_PER_SPAWN

    model_config = {"arbitrary_types_allowed": True}


class StateGraphRuntime:
    """The main agent execution runtime."""

    def __init__(
        self,
        planner: Planner | None = None,
        executor: Executor | None = None,
        critic: Critic | None = None,
        tools: ToolRegistry | None = None,
        config: KernelConfig | None = None,
        checkpoint_store: SQLiteCheckpointStore | None = None,
    ):
        self.config = config or KernelConfig()
        self.planner = planner or MockPlanner()
        self.executor = executor or ToolExecutor(tools=tools)
        self.critic = critic or MockCritic()
        self.checkpoint_store = checkpoint_store or SQLiteCheckpointStore(":memory:")
        self.subgraph_spawner = SubgraphSpawner()

    async def run(
        self,
        state: AgentState,
        events: EventBus | None = None,
    ) -> AgentState:
        """Run the full agent execution loop."""
        if events:
            events.emit(EventType.AGENT_START, {"status": "planning"}, agent_id=state.agent_id)

        # 1. Plan
        state.status = "planning"
        goal = state.scratchpad or (state.messages[-1].content if state.messages else "No goal")
        state.plan = await self.planner.create_plan(state, goal)

        if events:
            events.emit(
                EventType.PLAN_CREATED,
                {"steps": [s.id for s in state.plan], "total_steps": len(state.plan)},
                agent_id=state.agent_id,
            )

        # 2. Execute loop
        state.status = "executing"
        seq = 0
        while not self._all_steps_done(state) and not state.budget.halted:
            step = self._get_next_step(state)
            if step is None:
                break

            # Check budget before step
            if not state.check_budget_runtime(10.0):
                state.budget.halted = True
                state.budget.halt_reason = "runtime"
                if events:
                    events.emit(
                        EventType.BUDGET_EXCEEDED,
                        {"field": "runtime", "limit": state.budget.max_runtime, "actual": state.budget.elapsed},
                        agent_id=state.agent_id,
                    )
                break

            # Execute step
            step.status = "in_progress"
            if events:
                events.emit(
                    EventType.PLAN_STEP_START,
                    {"step_id": step.id, "instruction": step.instruction, "tool": step.tool, "args": step.args},
                    agent_id=state.agent_id,
                )

            result = await self.executor.execute(state, step)
            state.budget.elapsed += result.duration_ms / 1000.0

            if events:
                events.emit(
                    EventType.EXECUTOR_INVOKE,
                    {"step_id": step.id, "tool": step.tool, "args": step.args, "result_type": "string"},
                    agent_id=state.agent_id,
                )

            # 3. Critique
            state.status = "critiquing"
            critique = await self.critic.evaluate(state, step, result, events=events)

            if critique.passed:
                step.status = "completed"
                step.result = result.output
                if events:
                    events.emit(
                        EventType.PLAN_STEP_END,
                        {"step_id": step.id, "success": True, "result": result.output[:200], "duration_ms": result.duration_ms},
                        agent_id=state.agent_id,
                    )
            else:
                step.status = "failed"
                step.error = result.error
                if events:
                    events.emit(
                        EventType.PLAN_STEP_FAIL,
                        {"step_id": step.id, "error": result.error, "retry_count": step.retry_count},
                        agent_id=state.agent_id,
                    )

                # 4. Retry decision
                if self.critic.should_retry(state, step, result.error or ""):
                    if step.retry_count < self.config.max_retries:
                        step.retry_count += 1
                        step.status = "pending"
                        if events:
                            events.emit(
                                EventType.PLAN_STEP_RETRY,
                                {"step_id": step.id, "retry_count": step.retry_count},
                                agent_id=state.agent_id,
                            )
                        continue
                    else:
                        state.status = "failed"
                        state.error = f"Step {step.id} exhausted retries"
                        if events:
                            events.emit(
                                EventType.AGENT_FAIL,
                                {"error": state.error},
                                agent_id=state.agent_id,
                            )
                        return state
                else:
                    state.status = "failed"
                    state.error = f"Step {step.id} failed: {result.error}"
                    if events:
                        events.emit(
                            EventType.AGENT_FAIL,
                            {"error": state.error},
                            agent_id=state.agent_id,
                        )
                    return state

            # Checkpoint
            seq += 1
            if self.config.checkpoint_interval > 0 and seq % self.config.checkpoint_interval == 0:
                self.checkpoint_store.save(state.run_id, state.agent_id, seq, state.budget.elapsed, state)

        # 5. Complete
        if not state.budget.halted:
            state.status = "completed"
            if events:
                events.emit(
                    EventType.AGENT_COMPLETE,
                    {"status": "completed", "final_plan_status": "completed"},
                    agent_id=state.agent_id,
                )
        else:
            state.status = "halted"
            if events:
                events.emit(
                    EventType.AGENT_HALT,
                    {"reason": state.budget.halt_reason, "budget": state.budget.model_dump()},
                    agent_id=state.agent_id,
                )

        return state

    def _get_next_step(self, state: AgentState) -> PlanEntry | None:
        """Get the next pending step whose dependencies are all completed."""
        completed_ids = {s.id for s in state.plan if s.status == "completed"}
        for step in state.plan:
            if step.status == "pending":
                deps_met = all(dep in completed_ids for dep in (step.depends_on or []))
                if deps_met:
                    return step
        return None

    def _all_steps_done(self, state: AgentState) -> bool:
        """Check if all non-skipped steps are completed or failed."""
        return all(s.status in ("completed", "failed") for s in state.plan)
