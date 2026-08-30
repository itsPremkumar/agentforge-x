"""Tests for GEPA PromptGenome."""


from agentforge_x.evolution.genome import (
    CriticRubric,
    MutationRecord,
    PlanStrategy,
    PromptGenome,
    RetryPolicy,
    ToolPrompt,
)


class TestGenome:
    """Test PromptGenome creation and manipulation."""

    def test_genome_creation_with_all_components(self):
        """Test creating a genome with all components."""
        genome = PromptGenome(
            system_prompt="You are a helpful assistant.",
            tool_descriptions=[
                ToolPrompt(tool_name="bash", description="Run shell commands"),
            ],
            plan_strategy=PlanStrategy(decomposition_style="broad_first", max_depth=5),
            critic_rubric=CriticRubric(pass_threshold=0.8),
            retry_policy=RetryPolicy(max_retries=3, backoff="exponential"),
        )

        assert genome.system_prompt == "You are a helpful assistant."
        assert len(genome.tool_descriptions) == 1
        assert genome.plan_strategy.decomposition_style == "broad_first"
        assert genome.critic_rubric.pass_threshold == 0.8
        assert genome.retry_policy.max_retries == 3

    def test_genome_serialization_deserialization(self):
        """Test genome serialization and deserialization."""
        genome = PromptGenome(
            system_prompt="Test prompt",
            plan_strategy=PlanStrategy(max_depth=7),
        )

        data = genome.to_dict()
        restored = PromptGenome.from_dict(data)

        assert restored.system_prompt == genome.system_prompt
        assert restored.plan_strategy.max_depth == genome.plan_strategy.max_depth

    def test_mutation_record_tracking(self):
        """Test that mutations are tracked."""
        genome = PromptGenome()
        genome.mutation_log.append(MutationRecord(
            type="system_prompt_rewrite",
            target="system_prompt",
            description="Test mutation",
        ))

        assert len(genome.mutation_log) == 1
        assert genome.mutation_log[0].type == "system_prompt_rewrite"

    def test_lineage_tracking_parent_ids(self):
        """Test parent ID tracking."""
        parent1 = PromptGenome()
        parent2 = PromptGenome()
        child = PromptGenome(parent_ids=[parent1.id, parent2.id])

        assert len(child.parent_ids) == 2
        assert parent1.id in child.parent_ids

    def test_genome_clone(self):
        """Test genome cloning."""
        genome = PromptGenome(
            system_prompt="Original",
            plan_strategy=PlanStrategy(max_depth=5),
        )
        clone = genome.clone()

        assert clone.system_prompt == genome.system_prompt
        assert clone.id != genome.id
        assert genome.id in clone.parent_ids

    def test_genome_fitness_null_by_default(self):
        """Test that fitness is null by default."""
        genome = PromptGenome()
        assert genome.fitness is None
