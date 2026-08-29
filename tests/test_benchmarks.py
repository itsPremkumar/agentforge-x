"""Tests for benchmarks."""


from agentforge_x.evolution.benchmarks.builtin import (
    BenchmarkTask,
    create_budget_stress_benchmark,
    create_error_recovery_benchmark,
    create_plan_simple_benchmark,
    evaluate_benchmark,
    evaluate_task,
)
from agentforge_x.evolution.genome import PromptGenome


class TestBenchmarks:
    """Test benchmark framework."""

    def test_benchmark_loading_and_validation(self):
        """Test creating and validating a benchmark."""
        benchmark = create_plan_simple_benchmark()
        assert benchmark.id == "plan-simple"
        assert len(benchmark.tasks) > 0

    def test_task_evaluation_pass_fail(self):
        """Test task evaluation."""
        genome = PromptGenome(system_prompt="plan execute verify")
        task = BenchmarkTask(id="t1", name="Test", description="Test", expected_output="plan execute verify")
        result = evaluate_task(genome, task)
        assert result.passed is True
        assert result.score > 0.5

    def test_composite_score_calculation(self):
        """Test composite score calculation."""
        genome = PromptGenome(system_prompt="plan execute verify")
        benchmark = create_plan_simple_benchmark()
        result = evaluate_benchmark(genome, benchmark)
        assert result.composite_score >= 0.0
        assert result.composite_score <= 1.0

    def test_builtin_benchmarks(self):
        """Test that all builtin benchmarks can be created."""
        benchmarks = [
            create_plan_simple_benchmark(),
            create_error_recovery_benchmark(),
            create_budget_stress_benchmark(),
        ]
        for b in benchmarks:
            assert b.id is not None
            assert len(b.tasks) > 0

    def test_task_evaluation_failure(self):
        """Test task evaluation when it fails."""
        genome = PromptGenome(system_prompt="completely unrelated content")
        task = BenchmarkTask(id="t1", name="Test", description="Test", expected_output="plan execute verify")
        result = evaluate_task(genome, task)
        assert result.passed is False
        assert result.score < 0.5
