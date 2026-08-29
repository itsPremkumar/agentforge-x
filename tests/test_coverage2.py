"""Additional coverage tests to reach 85%."""

import pytest

from agentforge_x.kernel.budget import BudgetEnforcer, BudgetAllocation
from agentforge_x.kernel.checkpoint import SQLiteCheckpointStore
from agentforge_x.kernel.critic import MockCritic
from agentforge_x.kernel.event_bus import EventBus, EventType
from agentforge_x.kernel.executor import MockExecutor, ExecutionResult
from agentforge_x.kernel.planner import MockPlanner
from agentforge_x.kernel.retry import RetryScheduler
from agentforge_x.kernel.state_graph import StateGraphRuntime, KernelConfig
from agentforge_x.kernel.state import AgentState, BudgetState, PlanEntry


class TestMoreCoverage:
    """Additional coverage tests."""

    def test_budget_check_runtime_false(self):
        """Test check_runtime returns False when exceeding."""
        budget = BudgetState(max_runtime=10.0, elapsed=5.0)
        enforcer = BudgetEnforcer(budget)
        # 5 + 10 = 15 > 10, should be False
        assert enforcer.check_runtime(10.0) is False

    def test_budget_consume_runtime_ok(self):
        """Test consume_runtime within budget."""
        budget = BudgetState(max_runtime=100.0)
        enforcer = BudgetEnforcer(budget)
        assert enforcer.consume_runtime(50.0) is True
        assert budget.elapsed == 50.0

    def test_budget_consume_llm_call_ok(self):
        """Test consume_llm_call within budget."""
        budget = BudgetState(max_llm_calls=10)
        enforcer = BudgetEnforcer(budget)
        assert enforcer.consume_llm_call() is True
        assert budget.llm_calls == 1

    def test_budget_consume_tokens_ok(self):
        """Test consume_tokens within budget."""
        budget = BudgetState(max_tokens=1000)
        enforcer = BudgetEnforcer(budget)
        assert enforcer.consume_tokens(500) is True
        assert budget.tokens_used == 500

    def test_checkpoint_save_and_load(self):
        """Test save and load checkpoint."""
        store = SQLiteCheckpointStore(":memory:")
        state = AgentState(agent_id="test", run_id="run_1", messages=[])
        store.save("run_1", "test", 1, 1.0, state)
        loaded = store.load("run_1", "test", 1)
        assert loaded is not None
        ts, restored = loaded
        assert ts == 1.0

    def test_checkpoint_load_latest(self):
        """Test load latest checkpoint."""
        store = SQLiteCheckpointStore(":memory:")
        state = AgentState(agent_id="test", run_id="run_1", messages=[])
        store.save("run_1", "test", 1, 1.0, state)
        store.save("run_1", "test", 2, 2.0, state)
        loaded = store.load("run_1", "test")
        assert loaded is not None
        ts, _ = loaded
        assert ts == 2.0

    def test_checkpoint_load_nonexistent(self):
        """Test load nonexistent checkpoint."""
        store = SQLiteCheckpointStore(":memory:")
        loaded = store.load("nonexistent", "test")
        assert loaded is None

    def test_critic_evaluate_auto_pass(self):
        """Test critic evaluate with auto pass."""
        critic = MockCritic()  # Not always_pass, not always_fail
        state = AgentState()
        step = PlanEntry(id="step_0", instruction="Do X", tool="bash", args={})
        result = ExecutionResult(success=True, output="Done")
        import asyncio
        critique = asyncio.run(critic.evaluate(state, step, result))
        assert critique.passed is True

    def test_critic_evaluate_auto_fail(self):
        """Test critic evaluate with auto fail."""
        critic = MockCritic()  # Not always_pass, not always_fail
        state = AgentState()
        step = PlanEntry(id="step_0", instruction="Do X", tool="bash", args={})
        result = ExecutionResult(success=False, output="", error="Failed")
        import asyncio
        critique = asyncio.run(critic.evaluate(state, step, result))
        assert critique.passed is False

    def test_event_bus_emit_with_custom_ts(self):
        """Test emit with custom timestamp."""
        bus = EventBus(run_id="test", agent_id="agent_1")
        event = bus.emit(EventType.AGENT_START, {}, ts=42.0)
        assert event.ts == 42.0

    def test_event_bus_get_events_filtered(self):
        """Test get_events with filter."""
        bus = EventBus(run_id="test", agent_id="agent_1")
        bus.emit(EventType.AGENT_START, {})
        bus.emit(EventType.PLAN_CREATED, {})
        bus.emit(EventType.AGENT_START, {})
        events = bus.get_events(EventType.AGENT_START)
        assert len(events) == 2

    def test_executor_can_execute_false(self):
        """Test can_execute returns False for missing tool."""
        from agentforge_x.kernel.executor import ToolExecutor, ToolRegistry
        executor = ToolExecutor(ToolRegistry())
        step = PlanEntry(id="step_0", instruction="Do X", tool="missing", args={})
        assert executor.can_execute(step) is False

    def test_planner_estimate_empty(self):
        """Test estimate with empty plan."""
        planner = MockPlanner()
        estimate = planner.estimate([])
        assert estimate["estimated_runtime"] == 0.0

    def test_retry_decide_with_events(self):
        """Test decide with event bus."""
        scheduler = RetryScheduler()
        step = PlanEntry(id="step_0", instruction="Do X", tool="bash", args={}, retry_count=0)
        budget = BudgetEnforcer(BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000))
        events = EventBus(run_id="test", agent_id="test")
        decision = scheduler.decide(step, budget, events=events, agent_id="test")
        assert decision.should_retry is True

    def test_state_graph_run_with_events(self):
        """Test run with event bus."""
        import asyncio
        planner = MockPlanner()
        executor = MockExecutor(results=[ExecutionResult(success=True, output="Done")])
        critic = MockCritic(always_pass=True)
        runtime = StateGraphRuntime(planner=planner, executor=executor, critic=critic)
        state = AgentState(messages=[], scratchpad="Do X")
        events = EventBus(run_id=state.run_id, agent_id=state.agent_id)
        result = asyncio.run(runtime.run(state, events=events))
        assert result.status == "completed"
        assert len(events.get_events()) > 0

    def test_subgraph_complete(self):
        """Test complete_subgraph method."""
        from agentforge_x.kernel.subgraph import SubgraphSpawner
        spawner = SubgraphSpawner()
        parent_state = AgentState(
            agent_id="parent_1",
            run_id="run_1",
            messages=[],
            budget=BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000),
        )
        allocation = BudgetAllocation(max_runtime=100.0, max_llm_calls=10, max_tokens=5000)
        subgraph_id, _ = spawner.spawn(parent_state, goal="Do task", allocation=allocation)
        spawner.complete_subgraph(parent_state, subgraph_id, "Task completed")
        assert len(parent_state.fleet_events) == 2
