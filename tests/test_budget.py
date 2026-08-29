"""Tests for budget caps and enforcement."""

import pytest

from agentforge_x.kernel.budget import (
    BudgetEnforcer,
    BudgetAllocation,
    RUNTIME_OVERHEAD_PER_SPAWN,
)
from agentforge_x.kernel.state import BudgetState


class TestBudgetCaps:
    """Test budget enforcement and allocation."""

    def test_runtime_cap_enforcement(self):
        """Test that runtime budget is enforced."""
        budget = BudgetState(max_runtime=100.0)
        enforcer = BudgetEnforcer(budget)

        assert enforcer.check_runtime(50.0) is True
        assert enforcer.check_runtime(150.0) is False

        assert enforcer.consume_runtime(80.0) is True
        assert enforcer.consume_runtime(30.0) is False  # Would exceed
        assert budget.halted is True
        assert budget.halt_reason == "runtime"

    def test_llm_call_cap_enforcement(self):
        """Test that LLM call budget is enforced."""
        budget = BudgetState(max_llm_calls=5)
        enforcer = BudgetEnforcer(budget)

        for _ in range(5):
            assert enforcer.consume_llm_call() is True

        assert enforcer.consume_llm_call() is False
        assert budget.halted is True
        assert budget.halt_reason == "llm_calls"

    def test_token_cap_enforcement(self):
        """Test that token budget is enforced."""
        budget = BudgetState(max_tokens=1000)
        enforcer = BudgetEnforcer(budget)

        assert enforcer.consume_tokens(500) is True
        assert enforcer.consume_tokens(600) is False  # Would exceed
        assert budget.halted is True
        assert budget.halt_reason == "tokens"

    def test_subgraph_budget_allocation_from_parent(self):
        """Test that subgraph budget is allocated from parent."""
        parent_budget = BudgetState(max_runtime=600.0, max_llm_calls=100, max_tokens=100000)
        parent_enforcer = BudgetEnforcer(parent_budget)

        allocation = BudgetAllocation(max_runtime=100.0, max_llm_calls=10, max_tokens=5000)
        child_budget = parent_enforcer.allocate_subgraph_budget(allocation)

        assert child_budget.max_runtime == 100.0
        assert child_budget.max_llm_calls == 10
        assert child_budget.max_tokens == 5000
        # Parent should have overhead deducted
        assert parent_budget.elapsed == RUNTIME_OVERHEAD_PER_SPAWN

    def test_budget_overhead_deduction(self):
        """Test that spawn overhead is deducted from parent budget."""
        parent_budget = BudgetState(max_runtime=600.0)
        parent_enforcer = BudgetEnforcer(parent_budget)

        # Overhead should be deducted on each spawn
        allocation = BudgetAllocation(max_runtime=50.0, max_llm_calls=5, max_tokens=1000)
        parent_enforcer.allocate_subgraph_budget(allocation)

        assert parent_budget.elapsed == RUNTIME_OVERHEAD_PER_SPAWN  # 5.0 seconds

    def test_hard_cap_vs_soft_cap_behavior(self):
        """Test hard cap (halt) vs soft cap (retry) behavior."""
        # Hard cap: budget exceeded -> halt
        budget = BudgetState(max_runtime=10.0)
        enforcer = BudgetEnforcer(budget)
        enforcer.consume_runtime(15.0)
        assert budget.halted is True

        # After halt, no more actions allowed
        assert enforcer.check_runtime(1.0) is False
