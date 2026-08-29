"""GEPA PromptGenome representation."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class ToolPrompt(BaseModel):
    """Per-tool prompt enhancement."""

    tool_name: str
    description: str
    examples: list[str] = Field(default_factory=list)
    error_recovery: str = ""


class PlanStrategy(BaseModel):
    """Planner behavior parameters."""

    decomposition_style: str = "broad_first"  # broad_first | narrow_first | dependency_first
    max_depth: int = 5
    parallelization: int = 1
    lookahead: int = 3


class CriticCriterion(BaseModel):
    """A single critic evaluation criterion."""

    name: str
    weight: float = 1.0
    evaluator: str = "rule"  # llm | rule | hybrid
    prompt: str = ""


class CriticRubric(BaseModel):
    """Critic evaluation criteria."""

    criteria: list[CriticCriterion] = Field(default_factory=list)
    aggregation: str = "weighted_mean"  # weighted_mean | min | majority
    pass_threshold: float = 0.8


class RetryPolicy(BaseModel):
    """When/how to retry failed steps."""

    max_retries: int = 3
    backoff: str = "exponential"  # linear | exponential | constant
    initial_delay: float = 1.0
    max_delay: float = 30.0
    escalation: str = "replan"  # replan | subgraph | fail


class MutationRecord(BaseModel):
    """Record of a mutation applied to a genome."""

    type: str
    target: str
    description: str
    timestamp: float = 0.0


class PromptGenome(BaseModel):
    """A evolvable prompt genome.

    Contains all evolvable components: system prompt, tool descriptions,
    plan strategy, critic rubric, and retry policy.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generation: int = 0
    parent_ids: list[str] = Field(default_factory=list)
    fitness: float | None = None

    # Evolvable components
    system_prompt: str = "You are a helpful assistant."
    tool_descriptions: list[ToolPrompt] = Field(default_factory=list)
    plan_strategy: PlanStrategy = Field(default_factory=PlanStrategy)
    critic_rubric: CriticRubric = Field(default_factory=CriticRubric)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)

    # Metadata
    created_at: float = 0.0
    evaluated_at: float | None = None
    benchmark_results: dict[str, Any] | None = None
    mutation_log: list[MutationRecord] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptGenome:
        """Deserialize from dict."""
        return cls(**data)

    def clone(self) -> PromptGenome:
        """Create a deep copy with new ID."""
        new = self.model_copy(deep=True)
        new.id = str(uuid.uuid4())
        new.parent_ids = [self.id]
        new.fitness = None
        new.evaluated_at = None
        new.benchmark_results = None
        new.mutation_log = []
        return new
