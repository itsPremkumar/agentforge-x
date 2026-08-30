"""AgentState: core state model for the agentforge-x kernel.

AgentState is a TypedDict-like Pydantic model that serves as the single
source of truth for an agent's runtime state. It is checkpointed to SQLite
at every state transition.

Uses simulated timestamps (seconds as floats) for deterministic testing.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class AIMessage(BaseModel):
    """A message in the agent's conversation history."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    ts: float | None = None  # Simulated timestamp (seconds)


class PlanEntry(BaseModel):
    """A single step in the execution plan."""

    id: str  # e.g., "step_0"
    instruction: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] | None = None
    status: str = "pending"  # pending | in_progress | completed | failed
    result: str | None = None
    error: str | None = None
    retry_count: int = 0


class ArtifactRef(BaseModel):
    """Reference to a produced artifact."""

    id: str
    type: str  # e.g., "code", "data", "report"
    path: str
    metadata: dict[str, Any] | None = None


class BudgetState(BaseModel):
    """Per-agent budget tracking state."""

    max_runtime: float = 600.0      # Max simulated seconds
    elapsed: float = 0.0            # Elapsed simulated seconds
    max_llm_calls: int = 100        # Max LLM calls in this agent's lifetime
    llm_calls: int = 0             # LLM calls made so far
    max_tokens: int = 100000        # Max total tokens (in + out)
    tokens_used: int = 0           # Tokens consumed so far
    max_retries: int = 3           # Max retry attempts per step
    halted: bool = False            # True if any budget was exceeded
    halt_reason: str | None = None  # runtime | llm_calls | tokens | subgraph_budget


class FleetEvent(BaseModel):
    """A fleet-level event in the audit trail."""

    ts: float
    run_id: str
    agent_id: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """The single source of truth for an agent's runtime state.

    Uses simulated timestamps (float seconds) for deterministic testing.
    All time values are in simulated seconds, not wall-clock.
    """

    # Identity & routing
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_agent_id: str | None = None

    # Core working memory
    messages: list[AIMessage] = Field(default_factory=list)
    plan: list[PlanEntry] = Field(default_factory=list)
    scratchpad: str = ""
    artifacts: list[ArtifactRef] = Field(default_factory=list)

    # Fleet-level events (audit trail)
    fleet_events: list[FleetEvent] = Field(default_factory=list)

    # Lifecycle
    status: str = "idle"  # idle | planning | executing | critiquing | retrying | completed | failed | halted
    error: str | None = None

    # Budget tracking
    budget: BudgetState = Field(default_factory=BudgetState)

    # Metadata (simulated timestamps)
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: float | None = None

    model_config = {"arbitrary_types_allowed": True}

    def transition(self, new_status: str) -> AgentState:
        """Transition to a new status. Returns self for chaining."""
        valid_statuses = {
            "idle", "planning", "executing", "critiquing",
            "retrying", "completed", "failed", "halted"
        }
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status: {new_status}. Valid: {valid_statuses}")
        if self.status == "completed" and new_status != "completed":
            raise ValueError("Cannot transition from 'completed'")
        if self.status == "failed" and new_status not in ("failed", "completed"):
            raise ValueError("Cannot transition from 'failed'")
        if self.status == "halted":
            raise ValueError("Cannot transition from 'halted'")
        self.status = new_status
        return self

    def check_budget_runtime(self, additional_seconds: float) -> bool:
        """Check if adding runtime would exceed the cap. Returns True if OK."""
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for checkpointing."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Deserialize from a checkpoint dict."""
        return cls(**data)
