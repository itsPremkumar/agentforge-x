"""Evolutionary operators: mutation, crossover, selection."""

from __future__ import annotations

import random
from collections.abc import Callable

from agentforge_x.evolution.genome import (
    CriticRubric,
    MutationRecord,
    PlanStrategy,
    PromptGenome,
    RetryPolicy,
)

# Type alias for RNG
RNG = Callable[[], float]


def mutate(genome: PromptGenome, rng: RNG = random.random, rate: float = 0.3) -> PromptGenome:
    """Random mutation: randomly modify components with given probability."""
    mutant = genome.clone()
    mutant.generation = genome.generation + 1

    if rng() < rate:
        mutant.system_prompt = _mutate_text(mutant.system_prompt, rng)
        mutant.mutation_log.append(MutationRecord(
            type="system_prompt_rewrite",
            target="system_prompt",
            description="Random text mutation",
            timestamp=0.0,
        ))

    if rng() < rate:
        mutant.plan_strategy = _mutate_plan_strategy(mutant.plan_strategy, rng)
        mutant.mutation_log.append(MutationRecord(
            type="plan_strategy_adjust",
            target="plan_strategy",
            description="Plan strategy adjusted",
            timestamp=0.0,
        ))

    if rng() < rate:
        mutant.critic_rubric = _mutate_critic_rubric(mutant.critic_rubric, rng)
        mutant.mutation_log.append(MutationRecord(
            type="critic_rubric_tighten",
            target="critic_rubric",
            description="Critic rubric adjusted",
            timestamp=0.0,
        ))

    if rng() < rate:
        mutant.retry_policy = _mutate_retry_policy(mutant.retry_policy, rng)
        mutant.mutation_log.append(MutationRecord(
            type="retry_policy_change",
            target="retry_policy",
            description="Retry policy changed",
            timestamp=0.0,
        ))

    return mutant


def _mutate_text(text: str, rng: RNG) -> str:
    """Simple text mutation: add/remove a sentence."""
    sentences = text.split(". ")
    if len(sentences) > 1 and rng() < 0.5:
        idx = int(rng() * len(sentences))
        sentences.pop(idx)
    else:
        additions = [
            " Be thorough in your analysis.",
            " Consider edge cases.",
            " Verify your work.",
            " Think step by step.",
        ]
        sentences.append(additions[int(rng() * len(additions))])
    return ". ".join(sentences)


def _mutate_plan_strategy(strategy: PlanStrategy, rng: RNG) -> PlanStrategy:
    """Mutate plan strategy."""
    new = strategy.model_copy()
    if rng() < 0.5:
        styles = ["broad_first", "narrow_first", "dependency_first"]
        new.decomposition_style = styles[int(rng() * len(styles))]
    else:
        new.max_depth = max(1, min(10, new.max_depth + int(rng() * 3) - 1))
    return new


def _mutate_critic_rubric(rubric: CriticRubric, rng: RNG) -> CriticRubric:
    """Mutate critic rubric."""
    new = rubric.model_copy()
    new.pass_threshold = max(0.5, min(0.99, new.pass_threshold + (rng() - 0.5) * 0.1))
    return new


def _mutate_retry_policy(policy: RetryPolicy, rng: RNG) -> RetryPolicy:
    """Mutate retry policy."""
    new = policy.model_copy()
    if rng() < 0.5:
        new.max_retries = max(0, min(5, new.max_retries + int(rng() * 3) - 1))
    else:
        backoff_types = ["linear", "exponential", "constant"]
        new.backoff = backoff_types[int(rng() * len(backoff_types))]
    return new


def crossover(parent1: PromptGenome, parent2: PromptGenome, rng: RNG = random.random) -> PromptGenome:
    """Uniform crossover: randomly mix components from both parents."""
    child = parent1.clone()
    child.parent_ids = [parent1.id, parent2.id]
    child.generation = max(parent1.generation, parent2.generation) + 1

    if rng() < 0.5:
        child.system_prompt = parent2.system_prompt
    if rng() < 0.5:
        child.plan_strategy = parent2.plan_strategy.model_copy()
    if rng() < 0.5:
        child.critic_rubric = parent2.critic_rubric.model_copy()
    if rng() < 0.5:
        child.retry_policy = parent2.retry_policy.model_copy()

    child.mutation_log.append(MutationRecord(
        type="crossover_merge",
        target="multiple",
        description=f"Crossover from {parent1.id[:8]} and {parent2.id[:8]}",
        timestamp=0.0,
    ))

    return child


def tournament_select(pool: list[PromptGenome], tournament_size: int = 3, rng: RNG = random.random) -> PromptGenome:
    """Tournament selection: pick the best from a random subset."""
    size = min(tournament_size, len(pool))
    candidates = []
    for _ in range(size):
        candidates.append(pool[int(rng() * len(pool))])
    return max(candidates, key=lambda g: g.fitness or 0.0)


def roulette_select(pool: list[PromptGenome], rng: RNG = random.random) -> PromptGenome:
    """Roulette wheel selection: probability proportional to fitness."""
    fitnesses = [g.fitness or 0.0 for g in pool]
    total = sum(fitnesses)
    if total == 0:
        return pool[int(rng() * len(pool))]
    r = rng() * total
    cumulative = 0.0
    for genome, fitness in zip(pool, fitnesses, strict=False):
        cumulative += fitness
        if cumulative >= r:
            return genome
    return pool[-1]


def diversity(pool: list[PromptGenome]) -> float:
    """Measure population diversity: 0.0 = identical, 1.0 = maximally diverse."""
    if len(pool) < 2:
        return 0.0
    total_diff = 0
    comparisons = 0
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            total_diff += _genome_diff(pool[i], pool[j])
            comparisons += 1
    return total_diff / comparisons if comparisons > 0 else 0.0


def similarity(a: PromptGenome, b: PromptGenome) -> float:
    """Measure similarity between two genomes: 0.0 = different, 1.0 = identical."""
    return 1.0 - _genome_diff(a, b)


def _genome_diff(a: PromptGenome, b: PromptGenome) -> float:
    """Fraction of components that differ between two genomes."""
    diffs = 0
    total = 4

    if a.system_prompt != b.system_prompt:
        diffs += 1
    if a.plan_strategy.decomposition_style != b.plan_strategy.decomposition_style:
        diffs += 1
    if abs(a.critic_rubric.pass_threshold - b.critic_rubric.pass_threshold) > 0.05:
        diffs += 1
    if a.retry_policy.max_retries != b.retry_policy.max_retries:
        diffs += 1

    return diffs / total
