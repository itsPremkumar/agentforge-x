"""Tests for search algorithms."""

import random

from agentforge_x.evolution.genome import PromptGenome
from agentforge_x.evolution.search import (
    SearchConfig,
    SearchSpace,
    genetic_search,
    random_search,
)


class TestSearch:
    """Test search algorithms."""

    def test_search_space_definition(self):
        """Test search space definition."""
        space = SearchSpace()
        assert "decomposition_style" in space.dimensions
        assert "max_depth" in space.dimensions
        assert len(space.dimensions) == 10

    def test_genetic_algorithm_convergence(self):
        """Test genetic algorithm convergence."""
        rng = random.Random(42)
        population = [PromptGenome() for _ in range(10)]
        config = SearchConfig(algorithm="genetic", generations=5, population_size=10)
        best, final_pop, progress = genetic_search(
            population,
            evaluate=lambda g: rng.random(),
            config=config,
            rng=rng,
        )
        assert best is not None
        assert progress.generation > 0

    def test_random_search(self):
        """Test random search."""
        rng = random.Random(42)
        population = [PromptGenome() for _ in range(10)]
        config = SearchConfig(algorithm="random", generations=5, population_size=10)
        best, final_pop, progress = random_search(
            population,
            evaluate=lambda g: rng.random(),
            config=config,
            rng=rng,
        )
        assert best is not None

    def test_early_stopping(self):
        """Test early stopping."""
        rng = random.Random(42)
        population = [PromptGenome() for _ in range(10)]
        config = SearchConfig(
            algorithm="genetic",
            generations=20,
            population_size=10,
            early_stopping_patience=2,
            early_stopping_min_improvement=0.01,
        )
        best, final_pop, progress = genetic_search(
            population,
            evaluate=lambda g: 0.5,  # Constant fitness -> should stop early
            config=config,
            rng=rng,
        )
        # Should stop before 20 generations
        assert progress.generation < 20
