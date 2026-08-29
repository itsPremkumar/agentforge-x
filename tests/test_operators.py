"""Tests for evolutionary operators."""

import random

from agentforge_x.evolution.genome import PlanStrategy, PromptGenome
from agentforge_x.evolution.operators import (
    crossover,
    diversity,
    mutate,
    roulette_select,
    similarity,
    tournament_select,
)


class TestMutation:
    """Test mutation operators."""

    def test_random_mutation(self):
        """Test random mutation."""
        rng = random.Random(42).random
        genome = PromptGenome(system_prompt="Original prompt.")
        mutant = mutate(genome, rng=rng, rate=1.0)
        assert mutant.id != genome.id
        assert mutant.generation == 1

    def test_mutation_preserves_id(self):
        """Test that mutation creates new ID."""
        rng = random.Random(42).random
        genome = PromptGenome()
        mutant = mutate(genome, rng=rng)
        assert mutant.id != genome.id

    def test_mutation_adds_log_entry(self):
        """Test that mutation adds log entries."""
        rng = random.Random(42).random
        genome = PromptGenome()
        mutant = mutate(genome, rng=rng, rate=1.0)
        assert len(mutant.mutation_log) > 0

    def test_guided_mutation(self):
        """Test guided mutation."""
        rng = random.Random(42).random
        genome = PromptGenome(system_prompt="Do X. Do Y.")
        mutant = mutate(genome, rng=rng, rate=1.0)
        # With rate=1.0, at least one mutation should occur
        assert len(mutant.mutation_log) > 0


class TestCrossover:
    """Test crossover operators."""

    def test_uniform_crossover(self):
        """Test uniform crossover."""
        rng = random.Random(42).random
        parent1 = PromptGenome(system_prompt="Parent 1")
        parent2 = PromptGenome(system_prompt="Parent 2")
        child = crossover(parent1, parent2, rng=rng)
        assert child.id != parent1.id
        assert child.id != parent2.id
        assert len(child.parent_ids) == 2

    def test_crossover_preserves_lineage(self):
        """Test that crossover tracks parent IDs."""
        rng = random.Random(42).random
        parent1 = PromptGenome()
        parent2 = PromptGenome()
        child = crossover(parent1, parent2, rng=rng)
        assert parent1.id in child.parent_ids
        assert parent2.id in child.parent_ids


class TestSelection:
    """Test selection operators."""

    def test_tournament_selection(self):
        """Test tournament selection."""
        rng = random.Random(42).random
        pool = [PromptGenome() for _ in range(10)]
        for i, g in enumerate(pool):
            g.fitness = float(i)
        selected = tournament_select(pool, tournament_size=3, rng=rng)
        assert selected in pool

    def test_roulette_selection(self):
        """Test roulette wheel selection."""
        rng = random.Random(42).random
        pool = [PromptGenome() for _ in range(10)]
        for i, g in enumerate(pool):
            g.fitness = float(i)
        selected = roulette_select(pool, rng=rng)
        assert selected in pool

    def test_tournament_selects_best(self):
        """Test that tournament selection tends to pick better genomes."""
        rng = random.Random(42).random
        pool = [PromptGenome() for _ in range(20)]
        for i, g in enumerate(pool):
            g.fitness = float(i)
        selected = tournament_select(pool, tournament_size=5, rng=rng)
        # With tournament size 5 out of 20, should usually pick top half
        assert selected.fitness >= 10.0


class TestDiversity:
    """Test diversity measurement."""

    def test_diversity_identical(self):
        """Test diversity of identical genomes."""
        pool = [PromptGenome() for _ in range(5)]
        div = diversity(pool)
        assert div == 0.0

    def test_diversity_different(self):
        """Test diversity of different genomes."""
        pool = []
        for i in range(5):
            g = PromptGenome(system_prompt=f"Prompt {i}")
            g.plan_strategy = PlanStrategy(decomposition_style=["broad_first", "narrow_first", "dependency_first"][i % 3])
            pool.append(g)
        div = diversity(pool)
        assert div > 0.0

    def test_similarity(self):
        """Test similarity measurement."""
        a = PromptGenome(system_prompt="Same")
        b = PromptGenome(system_prompt="Same")
        c = PromptGenome(system_prompt="Different")

        assert similarity(a, b) > similarity(a, c)

    def test_single_element_diversity(self):
        """Test diversity with single element."""
        pool = [PromptGenome()]
        assert diversity(pool) == 0.0
