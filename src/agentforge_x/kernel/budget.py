"""Budget cap model for the agentforge-x kernel.

Hierarchical budget management: parent agents allocate budgets to subgraphs,
and each agent enforces its own caps. Subgraph spawning incurs a fixed
overhead (5s runtime per spawn).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentforge_x.kernel.state import BudgetState

# Fixed overhead per subgraph spawn (spec §3.4)
RUNTIME_OVERHEAD_PER_SPAWN = 5.0  # seconds
LLM_OVERHEAD_PER_SPAWN = 0
TOKEN_OVERHEAD_PER_SPAWN = 0


class BudgetAllocation(BaseModel):
    """Budget allocated from parent to subgraph."""

    max_runtime: float = 600.0
    max_llm_calls: int = 100
    max_tokens: int = 100000
    hard_cap: bool = False  # If True, subgraph failure halts parent


class BudgetEnforcer:
    """Enforces budget caps and decides whether to allow actions.

    Each agent has its own BudgetEnforcer wrapping its BudgetState.
    """

    def __init__(self, budget: BudgetState | None = None):
        self.budget = budget or BudgetState()

    def can_spawn_subgraph(self, allocation: BudgetAllocation) -> bool:
        """Check if a subgraph can be spawned with the given allocation.

        Deducts overhead from parent budget before checking.
        """
        remaining_runtime = self.budget.max_runtime - self.budget.elapsed - RUNTIME_OVERHEAD_PER_SPAWN
        remaining_llm = self.budget.max_llm_calls - self.budget.llm_calls
        remaining_tokens = self.budget.max_tokens - self.budget.tokens_used

        return (
            remaining_runtime >= allocation.max_runtime
            and remaining_llm >= allocation.max_llm_calls
            and remaining_tokens >= allocation.max_tokens
        )

    def allocate_subgraph_budget(self, allocation: BudgetAllocation) -> BudgetState:
        """Allocate budget for a subgraph. Deducts overhead from parent.

        Returns the new BudgetState for the subgraph.
        Raises RuntimeError if budget is insufficient.
        """
        if not self.can_spawn_subgraph(allocation):
            self.budget.halted = True
            self.budget.halt_reason = "subgraph_budget"
            raise RuntimeError("Insufficient budget to spawn subgraph")

        # Deduct overhead from parent
        self.budget.elapsed += RUNTIME_OVERHEAD_PER_SPAWN

        # Create subgraph budget
        return BudgetState(
            max_runtime=allocation.max_runtime,
            max_llm_calls=allocation.max_llm_calls,
            max_tokens=allocation.max_tokens,
        )

    def check_runtime(self, additional_seconds: float) -> bool:
        """Check if adding runtime would exceed the cap."""
        return (self.budget.elapsed + additional_seconds) <= self.budget.max_runtime

    def consume_runtime(self, seconds: float) -> bool:
        """Consume runtime budget. Returns False if exceeded."""
        self.budget.elapsed += seconds
        if self.budget.elapsed > self.budget.max_runtime:
            self.budget.halted = True
            self.budget.halt_reason = "runtime"
            return False
        return True

    def consume_llm_call(self) -> bool:
        """Consume an LLM call budget. Returns False if exceeded."""
        self.budget.llm_calls += 1
        if self.budget.llm_calls > self.budget.max_llm_calls:
            self.budget.halted = True
            self.budget.halt_reason = "llm_calls"
            return False
        return True

    def consume_tokens(self, count: int) -> bool:
        """Consume token budget. Returns False if exceeded."""
        self.budget.tokens_used += count
        if self.budget.tokens_used > self.budget.max_tokens:
            self.budget.halted = True
            self.budget.halt_reason = "tokens"
            return False
        return True

    def is_halted(self) -> bool:
        """Check if any budget is exceeded."""
        return self.budget.halted

    def remaining(self) -> dict[str, Any]:
        """Return remaining budget."""
        return {
            "runtime_remaining": max(0, self.budget.max_runtime - self.budget.elapsed),
            "llm_calls_remaining": max(0, self.budget.max_llm_calls - self.budget.llm_calls),
            "tokens_remaining": max(0, self.budget.max_tokens - self.budget.tokens_used),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpointing."""
        return self.budget.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetEnforcer:
        """Deserialize from checkpoint."""
        return cls(budget=BudgetState(**data))
