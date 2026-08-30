"""Evolution loop: the main evolution cycle."""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from agentforge_x.evolution.genome import PromptGenome
from agentforge_x.evolution.operators import crossover, diversity, mutate, tournament_select
from agentforge_x.evolution.rollback import RollbackManager
from agentforge_x.evolution.safety import SafetyGate, SafetyViolation


class EvolutionConfig(BaseModel):
    """Configuration for the evolution loop."""

    population_size: int = 20
    max_generations: int = 10
    elite_ratio: float = 0.1
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    max_retries: int = 3
    enable_safety: bool = True
    checkpoint_interval: int = 5


class GenerationResult(BaseModel):
    """Result of a single generation."""

    generation: int = 0
    population: list[PromptGenome] = Field(default_factory=list)
    best: PromptGenome | None = None
    mean_fitness: float = 0.0
    diversity: float = 0.0
    elapsed_time: float = 0.0
    safety_violations: list[SafetyViolation] = Field(default_factory=list)
    action: str = "continue"  # continue | rollback | stop | pause


class EvolutionResult(BaseModel):
    """Result of the full evolution run."""

    success: bool = False
    best_genome: PromptGenome | None = None
    total_generations: int = 0
    total_evaluations: int = 0
    total_runtime: float = 0.0
    fitness_curve: list[float] = Field(default_factory=list)
    final_diversity: float = 0.0
    safety_violations: list[SafetyViolation] = Field(default_factory=list)
    rollback_count: int = 0


class EvolutionLoop:
    """Main evolution loop: iterate generations with safety gates and rollback."""

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        evaluate: Callable[[PromptGenome], float] | None = None,
        safety_gate: SafetyGate | None = None,
        rollback_manager: RollbackManager | None = None,
    ):
        self.config = config or EvolutionConfig()
        self.evaluate = evaluate or (lambda g: 0.0)
        self.safety_gate = safety_gate or SafetyGate()
        self.rollback_manager = rollback_manager or RollbackManager()
        self.generation = 0
        self.population: list[PromptGenome] = []
        self.best_genome: PromptGenome | None = None
        self.history: list[GenerationResult] = []

    def initialize(self, seed_population: list[PromptGenome] | None = None):
        """Initialize the population."""
        if seed_population:
            self.population = seed_population[:self.config.population_size]
        else:
            self.population = [PromptGenome() for _ in range(self.config.population_size)]

    def step(self) -> GenerationResult:
        """Run one generation of evolution."""
        start = time.time()

        # 1. Evaluate
        for genome in self.population:
            if genome.fitness is None:
                genome.fitness = self.evaluate(genome)

        # 2. Sort by fitness
        self.population.sort(key=lambda g: g.fitness or 0.0, reverse=True)

        # 3. Update best
        current_best = self.population[0]
        current_best_fitness = current_best.fitness or 0.0
        if self.best_genome is None:
            self.best_genome = current_best
        else:
            best_fitness = self.best_genome.fitness or 0.0
            if current_best_fitness > best_fitness:
                self.best_genome = current_best

        # 4. Safety check
        violations = []
        if self.config.enable_safety:
            for genome in self.population:
                violations.extend(self.safety_gate.evaluate(genome))

        # 5. Elitism
        elite_count = max(1, int(self.config.elite_ratio * len(self.population)))
        elites = self.population[:elite_count]

        # 6. Create offspring
        offspring = []
        while len(offspring) < len(self.population) - elite_count:
            parent1 = tournament_select(self.population, tournament_size=3)
            parent2 = tournament_select(self.population, tournament_size=3)

            if parent2 is None or parent1.id == parent2.id:
                child = parent1.clone()
            else:
                child = crossover(parent1, parent2)

            child = mutate(child, rate=self.config.mutation_rate)
            offspring.append(child)

        self.population = elites + offspring
        self.generation += 1

        # 7. Checkpoint
        if self.generation % self.config.checkpoint_interval == 0:
            self.rollback_manager.save(self.generation, self.population)

        elapsed = time.time() - start

        result = GenerationResult(
            generation=self.generation,
            population=self.population[:],
            best=current_best,
            mean_fitness=sum(g.fitness or 0.0 for g in self.population) / len(self.population),
            diversity=diversity(self.population),
            elapsed_time=elapsed,
            safety_violations=violations,
        )
        self.history.append(result)
        return result

    def run(self, seed_population: list[PromptGenome] | None = None) -> EvolutionResult:
        """Run the full evolution loop."""
        start = time.time()
        self.initialize(seed_population)

        total_violations = []
        rollback_count = 0

        for _gen in range(self.config.max_generations):
            result = self.step()
            total_violations.extend(result.safety_violations)

            # Check for critical violations
            critical = [v for v in result.safety_violations if v.severity == "critical"]
            if critical:
                # Rollback
                rollback_count += 1
                self.population = self.rollback_manager.rollback_to_last_safe()

        return EvolutionResult(
            success=True,
            best_genome=self.best_genome,
            total_generations=self.generation,
            total_evaluations=sum(1 for g in self.population if g.fitness is not None),
            total_runtime=time.time() - start,
            fitness_curve=[r.mean_fitness for r in self.history],
            final_diversity=diversity(self.population),
            safety_violations=total_violations,
            rollback_count=rollback_count,
        )
