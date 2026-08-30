"""Rollback manager for evolution safety."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from agentforge_x.evolution.genome import PromptGenome


class Checkpoint(BaseModel):
    """A checkpoint in the evolution history."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generation: int = 0
    timestamp: float = 0.0
    population: list[PromptGenome] = Field(default_factory=list)
    best_fitness: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    safety_status: str = "clean"  # clean | warning | violation


class RollbackManager:
    """Manages checkpoints and rollback for evolution safety."""

    def __init__(self):
        self.checkpoints: list[Checkpoint] = []

    def save(
        self,
        generation: int,
        population: list[PromptGenome],
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Save a checkpoint."""
        best = max(population, key=lambda g: g.fitness or 0.0)
        checkpoint = Checkpoint(
            generation=generation,
            timestamp=0.0,
            population=[g.clone() for g in population],
            best_fitness=best.fitness or 0.0,
            metadata=metadata or {},
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def rollback(self, target_generation: int) -> list[PromptGenome]:
        """Rollback to a specific generation."""
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.generation <= target_generation:
                return [g.clone() for g in checkpoint.population]
        return []

    def rollback_to_last_safe(self) -> list[PromptGenome]:
        """Rollback to the last generation without safety violations."""
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.safety_status == "clean":
                return [g.clone() for g in checkpoint.population]
        return []

    def find_safe_generation(self) -> int:
        """Find the last generation without safety violations."""
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.safety_status == "clean":
                return checkpoint.generation
        return 0

    def partial_rollback(self, target: PromptGenome, reference: PromptGenome) -> PromptGenome:
        """Gradually revert changes from target toward reference."""
        result = target.clone()
        # Revert system prompt if too different
        if len(result.system_prompt) > len(reference.system_prompt) * 1.5:
            result.system_prompt = reference.system_prompt
        # Revert critic threshold if too low
        if result.critic_rubric.pass_threshold < reference.critic_rubric.pass_threshold * 0.8:
            result.critic_rubric.pass_threshold = reference.critic_rubric.pass_threshold
        return result
