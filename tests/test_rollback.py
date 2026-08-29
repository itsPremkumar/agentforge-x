"""Tests for rollback manager."""


from agentforge_x.evolution.genome import CriticRubric, PromptGenome
from agentforge_x.evolution.rollback import RollbackManager


class TestRollback:
    """Test rollback manager."""

    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        manager = RollbackManager()
        population = [PromptGenome() for _ in range(5)]
        for g in population:
            g.fitness = 0.5
        checkpoint = manager.save(1, population)
        assert checkpoint.generation == 1
        assert len(checkpoint.population) == 5

    def test_rollback_to_generation(self):
        """Test rollback to a specific generation."""
        manager = RollbackManager()
        pop1 = [PromptGenome() for _ in range(5)]
        pop2 = [PromptGenome() for _ in range(5)]
        manager.save(1, pop1)
        manager.save(2, pop2)
        restored = manager.rollback(1)
        assert len(restored) == 5

    def test_rollback_to_last_safe(self):
        """Test rollback to last safe generation."""
        manager = RollbackManager()
        pop1 = [PromptGenome() for _ in range(5)]
        pop2 = [PromptGenome() for _ in range(5)]
        cp1 = manager.save(1, pop1)
        cp1.safety_status = "clean"
        cp2 = manager.save(2, pop2)
        cp2.safety_status = "violation"
        restored = manager.rollback_to_last_safe()
        assert len(restored) == 5

    def test_find_safe_generation(self):
        """Test finding the last safe generation."""
        manager = RollbackManager()
        pop1 = [PromptGenome() for _ in range(5)]
        pop2 = [PromptGenome() for _ in range(5)]
        cp1 = manager.save(1, pop1)
        cp1.safety_status = "clean"
        cp2 = manager.save(2, pop2)
        cp2.safety_status = "violation"
        safe_gen = manager.find_safe_generation()
        assert safe_gen == 1

    def test_partial_rollback(self):
        """Test partial rollback."""
        manager = RollbackManager()
        target = PromptGenome(
            system_prompt="A" * 100,
            critic_rubric=CriticRubric(pass_threshold=0.3),
        )
        reference = PromptGenome(
            system_prompt="A" * 10,
            critic_rubric=CriticRubric(pass_threshold=0.8),
        )
        result = manager.partial_rollback(target, reference)
        assert result is not None
