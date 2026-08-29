"""Additional tests to boost coverage."""

import pytest

from agentforge_x.kernel.state import AgentState, BudgetState, PlanEntry
from agentforge_x.kernel.budget import BudgetEnforcer, BudgetAllocation, RUNTIME_OVERHEAD_PER_SPAWN
from agentforge_x.kernel.checkpoint import SQLiteCheckpointStore
from agentforge_x.kernel.critic import MockCritic
from agentforge_x.kernel.event_bus import EventBus, EventType, read_jsonl
from agentforge_x.kernel.executor import ToolExecutor, ToolRegistry, ExecutionResult, MockExecutor
from agentforge_x.kernel.planner import MockPlanner
from agentforge_x.kernel.retry import RetryScheduler
from agentforge_x.kernel.state_graph import StateGraphRuntime, KernelConfig


class TestStateCoverage:
    """Cover remaining state.py lines."""

    def test_transition_from_failed(self):
        """Test that transitioning from failed raises error."""
        state = AgentState()
        state.status = "failed"
        with pytest.raises(ValueError):
            state.transition("planning")

    def test_transition_from_halted(self):
        """Test that transitioning from halted raises error."""
        state = AgentState()
        state.status = "halted"
        with pytest.raises(ValueError):
            state.transition("planning")

    def test_transition_invalid_status(self):
        """Test that transitioning to invalid status raises error."""
        state = AgentState()
        with pytest.raises(ValueError):
            state.transition("invalid_status")

    def test_consume_runtime_exceeds(self):
        """Test consume_runtime when it exceeds budget."""
        state = AgentState()
        state.budget.max_runtime = 50.0
        assert state.consume_runtime(60.0) is False
        assert state.budget.halted is True
        assert state.budget.halt_reason == "runtime"

    def test_consume_tokens_exceeds(self):
        """Test consume_tokens when it exceeds budget."""
        state = AgentState()
        state.budget.max_tokens = 100
        assert state.consume_tokens(150) is False
        assert state.budget.halted is True
        assert state.budget.halt_reason == "tokens"


class TestBudgetCoverage:
    """Cover remaining budget.py lines."""

    def test_can_spawn_subgraph(self):
        """Test can_spawn_subgraph method."""
        budget = BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000)
        enforcer = BudgetEnforcer(budget)
        allocation = BudgetAllocation(max_runtime=100.0, max_llm_calls=10, max_tokens=5000)
        assert enforcer.can_spawn_subgraph(allocation) is True

    def test_allocate_subgraph_budget(self):
        """Test allocate_subgraph_budget method."""
        budget = BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000)
        enforcer = BudgetEnforcer(budget)
        allocation = BudgetAllocation(max_runtime=100.0, max_llm_calls=10, max_tokens=5000)
        child_budget = enforcer.allocate_subgraph_budget(allocation)
        assert child_budget.max_runtime == 100.0

    def test_is_halted(self):
        """Test is_halted method."""
        budget = BudgetState()
        enforcer = BudgetEnforcer(budget)
        assert enforcer.is_halted() is False
        budget.halted = True
        assert enforcer.is_halted() is True

    def test_remaining(self):
        """Test remaining method."""
        budget = BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000)
        enforcer = BudgetEnforcer(budget)
        remaining = enforcer.remaining()
        assert remaining["runtime_remaining"] == 600.0
        assert remaining["llm_calls_remaining"] == 100
        assert remaining["tokens_remaining"] == 100000

    def test_to_dict(self):
        """Test to_dict method."""
        budget = BudgetState()
        enforcer = BudgetEnforcer(budget)
        data = enforcer.to_dict()
        assert "max_runtime" in data

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "max_runtime": 600.0,
            "elapsed": 0.0,
            "max_llm_calls": 100,
            "llm_calls": 0,
            "max_tokens": 100000,
            "tokens_used": 0,
            "halted": False,
            "halt_reason": None,
        }
        enforcer = BudgetEnforcer.from_dict(data)
        assert enforcer.budget.max_runtime == 600.0


class TestCheckpointCoverage:
    """Cover remaining checkpoint.py lines."""

    def test_list_checkpoints(self):
        """Test list_checkpoints method."""
        store = SQLiteCheckpointStore(":memory:")
        state = AgentState(agent_id="test", run_id="run_1", messages=[])
        store.save("run_1", "test", 1, 1.0, state)
        store.save("run_1", "test", 2, 2.0, state)
        store.save("run_1", "test", 3, 3.0, state)
        checkpoints = store.list_checkpoints("run_1", "test")
        assert checkpoints == [1, 2, 3]

    def test_latest_seq(self):
        """Test latest_seq method."""
        store = SQLiteCheckpointStore(":memory:")
        state = AgentState(agent_id="test", run_id="run_1", messages=[])
        store.save("run_1", "test", 1, 1.0, state)
        store.save("run_1", "test", 5, 5.0, state)
        assert store.latest_seq("run_1", "test") == 5

    def test_delete_run(self):
        """Test delete_run method."""
        store = SQLiteCheckpointStore(":memory:")
        state = AgentState(agent_id="test", run_id="run_1", messages=[])
        store.save("run_1", "test", 1, 1.0, state)
        deleted = store.delete_run("run_1")
        assert deleted == 1
        assert store.latest_seq("run_1", "test") is None


class TestCriticCoverage:
    """Cover remaining critic.py lines."""

    @pytest.mark.asyncio
    async def test_evaluate_progress_all_completed(self):
        """Test evaluate_progress when all steps completed."""
        critic = MockCritic()
        state = AgentState()
        state.plan = [
            PlanEntry(id="step_0", instruction="A", tool="bash", args={}, status="completed"),
            PlanEntry(id="step_1", instruction="B", tool="bash", args={}, status="completed"),
        ]
        critique = await critic.evaluate_progress(state)
        assert critique.passed is True
        assert critique.score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_progress_empty_plan(self):
        """Test evaluate_progress with empty plan."""
        critic = MockCritic()
        state = AgentState()
        critique = await critic.evaluate_progress(state)
        assert critique.passed is False
        assert critique.score == 0.0


class TestEventBusCoverage:
    """Cover remaining event_bus.py lines."""

    def test_clear(self):
        """Test clear method."""
        bus = EventBus(run_id="test", agent_id="agent_1")
        bus.emit(EventType.AGENT_START, {})
        assert len(bus.get_events()) == 1
        bus.clear()
        assert len(bus.get_events()) == 0

    def test_context_manager(self):
        """Test context manager."""
        with EventBus(run_id="test", agent_id="agent_1") as bus:
            bus.emit(EventType.AGENT_START, {})
            assert len(bus.get_events()) == 1


class TestExecutorCoverage:
    """Cover remaining executor.py lines."""

    @pytest.mark.asyncio
    async def test_async_tool(self):
        """Test executing an async tool."""
        async def async_echo(args):
            return f"Async: {args.get('text', '')}"

        registry = ToolRegistry()
        registry.register("async_echo", async_echo)
        executor = ToolExecutor(registry)
        state = AgentState()
        step = PlanEntry(id="step_0", instruction="Echo", tool="async_echo", args={"text": "hello"})
        result = await executor.execute(state, step)
        assert result.success is True
        assert "Async: hello" in result.output


class TestPlannerCoverage:
    """Cover remaining planner.py lines."""

    @pytest.mark.asyncio
    async def test_create_plan_empty_goal(self):
        """Test create_plan with empty goal."""
        planner = MockPlanner()
        state = AgentState()
        plan = await planner.create_plan(state, "")
        assert len(plan) >= 1

    @pytest.mark.asyncio
    async def test_replan_without_failed_step(self):
        """Test replan without specifying failed step."""
        planner = MockPlanner()
        state = AgentState()
        state.plan = await planner.create_plan(state, "Do X; Do Y")
        new_plan = await planner.replan(state)
        assert len(new_plan) == len(state.plan)


class TestRetryCoverage:
    """Cover remaining retry.py lines."""

    def test_compute_backoff(self):
        """Test compute_backoff method."""
        scheduler = RetryScheduler(base_delay=1.0, max_delay=30.0, backoff_factor=2.0, jitter=False)
        assert scheduler.compute_backoff(0) == 1.0
        assert scheduler.compute_backoff(1) == 2.0
        assert scheduler.compute_backoff(2) == 4.0
        assert scheduler.compute_backoff(10) == 30.0  # capped at max_delay

    def test_record_retry(self):
        """Test record_retry method."""
        scheduler = RetryScheduler()
        step = PlanEntry(id="step_0", instruction="Do X", tool="bash", args={}, retry_count=0)
        new_count = scheduler.record_retry(step)
        assert new_count == 1


class TestStateGraphCoverage:
    """Cover remaining state_graph.py lines."""

    @pytest.mark.asyncio
    async def test_run_with_multiple_steps(self):
        """Test run with multiple steps."""
        planner = MockPlanner()
        # Provide enough results for all steps
        executor = MockExecutor(results=[
            ExecutionResult(success=True, output="Step 1 done"),
            ExecutionResult(success=True, output="Step 2 done"),
            ExecutionResult(success=True, output="Step 3 done"),
            ExecutionResult(success=True, output="Step 4 done"),
        ])
        critic = MockCritic(always_pass=True)
        runtime = StateGraphRuntime(planner=planner, executor=executor, critic=critic)
        state = AgentState(messages=[], scratchpad="Do X; Do Y")
        result = await runtime.run(state)
        assert result.status == "completed"
        assert len(result.plan) == 2
