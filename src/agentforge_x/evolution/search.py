"""Search algorithms for strategy optimization."""

from __future__ import annotations

import random
from collections.abc import Callable

from agentforge_x.evolution.genome import PromptGenome
from agentforge_x.evolution.operators import crossover, diversity, mutate, tournament_select


class SearchSpace:
    """Defines the search space for strategy optimization."""

    def __init__(self):
        self.dimensions = {
            "decomposition_style": ["broad_first", "narrow_first", "dependency_first"],
            "max_depth": list(range(1, 11)),
            "parallelization": list(range(1, 9)),
            "critic_threshold": [i / 100 for i in range(50, 100)],
            "max_retries": list(range(0, 6)),
            "retry_backoff": ["linear", "exponential", "constant"],
            "system_prompt_length": list(range(50, 2001, 50)),
            "tool_prompt_examples": list(range(0, 6)),
            "plan_lookahead": list(range(1, 11)),
            "mutation_rate": [i / 100 for i in range(1, 51)],
        }


class SearchConfig:
    """Configuration for search algorithms."""

    def __init__(
        self,
        algorithm: str = "genetic",
        population_size: int = 20,
        generations: int = 10,
        elite_ratio: float = 0.1,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.3,
        early_stopping_patience: int = 5,
        early_stopping_min_improvement: float = 0.01,
    ):
        self.algorithm = algorithm
        self.population_size = population_size
        self.generations = generations
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_improvement = early_stopping_min_improvement


class SearchProgress:
    """Tracks search progress."""

    def __init__(self):
        self.generation = 0
        self.best_fitness = 0.0
        self.mean_fitness = 0.0
        self.diversity = 0.0
        self.genomes_evaluated = 0
        self.elapsed_time = 0.0
        self.estimated_remaining = 0.0
        self.pareto_front: list[PromptGenome] = []


def genetic_search(
    initial_population: list[PromptGenome],
    evaluate: Callable[[PromptGenome], float],
    config: SearchConfig,
    rng: random.Random = None,
) -> tuple[PromptGenome, list[PromptGenome], SearchProgress]:
    """Standard genetic algorithm search."""
    if rng is None:
        rng = random.Random()

    population = initial_population[:]
    best_ever = max(population, key=lambda g: g.fitness or 0.0)
    progress = SearchProgress()
    no_improvement_count = 0

    for gen in range(config.generations):
        # Evaluate
        for genome in population:
            if genome.fitness is None:
                genome.fitness = evaluate(genome)
                progress.genomes_evaluated += 1

        # Sort by fitness
        population.sort(key=lambda g: g.fitness or 0.0, reverse=True)

        # Update best
        current_best = population[0]
        if (current_best.fitness or 0.0) > best_ever.fitness:
            best_ever = current_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Update progress
        progress.generation = gen
        progress.best_fitness = best_ever.fitness or 0.0
        progress.mean_fitness = sum(g.fitness or 0.0 for g in population) / len(population)
        progress.diversity = diversity(population)

        # Early stopping
        if no_improvement_count >= config.early_stopping_patience:
            break

        # Elitism
        elite_count = max(1, int(config.elite_ratio * len(population)))
        elites = population[:elite_count]

        # Create offspring
        offspring = []
        while len(offspring) < len(population) - elite_count:
            parent1 = tournament_select(population, tournament_size=3, rng=rng.random)
            parent2 = tournament_select(population, tournament_size=3, rng=rng.random)

            if rng.random() < config.crossover_rate:
                child = crossover(parent1, parent2, rng=rng.random)
            else:
                child = parent1.clone()

            child = mutate(child, rng=rng.random, rate=config.mutation_rate)
            offspring.append(child)

        population = elites + offspring

    return best_ever, population, progress


def random_search(
    initial_population: list[PromptGenome],
    evaluate: Callable[[PromptGenome], float],
    config: SearchConfig,
    rng: random.Random = None,
) -> tuple[PromptGenome, list[PromptGenome], SearchProgress]:
    """Random search baseline."""
    if rng is None:
        rng = random.Random()

    population = initial_population[:]
    best_ever = max(population, key=lambda g: g.fitness or 0.0)
    progress = SearchProgress()

    for gen in range(config.generations):
        # Evaluate
        for genome in population:
            if genome.fitness is None:
                genome.fitness = evaluate(genome)
                progress.genomes_evaluated += 1

        population.sort(key=lambda g: g.fitness or 0.0, reverse=True)
        current_best = population[0]
        if (current_best.fitness or 0.0) > best_ever.fitness:
            best_ever = current_best

        progress.generation = gen
        progress.best_fitness = best_ever.fitness or 0.0
        progress.mean_fitness = sum(g.fitness or 0.0 for g in population) / len(population)
        progress.diversity = diversity(population)

        # Generate new random population
        offspring = []
        for _ in range(len(population)):
            parent = tournament_select(population, tournament_size=3, rng=rng.random)
            child = mutate(parent, rng=rng.random, rate=config.mutation_rate)
            offspring.append(child)
        population = offspring

    return best_ever, population, progress
