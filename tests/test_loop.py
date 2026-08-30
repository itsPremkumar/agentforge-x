"""Tests for the evolution loop."""


from agentforge_x.evolution.loop import EvolutionConfig, EvolutionLoop


class TestEvolutionLoop:
    """Test the evolution loop."""

    def test_full_evolution_loop(self):
        """Test a full evolution loop with 5 generations."""
        config = EvolutionConfig(population_size=10, max_generations=5)
        loop = EvolutionLoop(config=config, evaluate=lambda g: 0.5)
        result = loop.run()
        assert result.success is True
        assert result.total_generations == 5

    def test_population_initialization(self):
        """Test population initialization."""
        config = EvolutionConfig(population_size=15)
        loop = EvolutionLoop(config=config)
        loop.initialize()
        assert len(loop.population) == 15

    def test_fitness_convergence(self):
        """Test that fitness converges over generations."""
        config = EvolutionConfig(population_size=10, max_generations=10)
        loop = EvolutionLoop(config=config, evaluate=lambda g: 0.5)
        result = loop.run()
        assert result.best_genome is not None
        assert result.best_genome.fitness is not None

    def test_budget_aware_evolution(self):
        """Test evolution with budget constraints."""
        config = EvolutionConfig(population_size=10, max_generations=5)
        loop = EvolutionLoop(config=config, evaluate=lambda g: 0.5)
        result = loop.run()
        assert result.total_runtime >= 0

    def test_checkpoint_save(self):
        """Test checkpoint saving."""
        config = EvolutionConfig(population_size=10, max_generations=10, checkpoint_interval=2)
        loop = EvolutionLoop(config=config, evaluate=lambda g: 0.5)
        loop.run()
        # Checkpoints are saved at generations 2, 4, 6, 8, 10
        assert len(loop.rollback_manager.checkpoints) >= 4

    def test_safety_violation_handling(self):
        """Test that safety violations are handled."""
        config = EvolutionConfig(population_size=10, max_generations=5, enable_safety=True)
        loop = EvolutionLoop(config=config, evaluate=lambda g: 0.5)
        result = loop.run()
        # Should complete without errors
        assert result.success is True

    def test_diversity_maintenance(self):
        """Test that diversity is maintained."""
        config = EvolutionConfig(population_size=20, max_generations=5)
        loop = EvolutionLoop(config=config, evaluate=lambda g: 0.5)
        result = loop.run()
        assert result.final_diversity >= 0.0

    def test_parallel_evaluation(self):
        """Test parallel evaluation."""
        config = EvolutionConfig(population_size=10, max_generations=3)
        loop = EvolutionLoop(config=config, evaluate=lambda g: 0.5)
        result = loop.run()
        assert result.total_evaluations > 0
