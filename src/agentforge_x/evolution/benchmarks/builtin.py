"""Benchmark framework for evaluating genomes."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentforge_x.evolution.genome import PromptGenome


class BenchmarkTask(BaseModel):
    """A single task in a benchmark."""

    id: str
    name: str
    description: str
    expected_output: str
    difficulty: str = "medium"  # easy | medium | hard | expert
    tags: list[str] = Field(default_factory=list)
    max_runtime: float = 60.0


class Benchmark(BaseModel):
    """A benchmark for evaluating genomes."""

    id: str
    name: str
    description: str
    tasks: list[BenchmarkTask] = Field(default_factory=list)
    timeout: float = 60.0


class TaskResult(BaseModel):
    """Result of evaluating a genome on a task."""

    task_id: str
    passed: bool
    score: float = 0.0
    output: str = ""
    expected: str = ""
    runtime: float = 0.0
    error: str | None = None


class BenchmarkResult(BaseModel):
    """Result of evaluating a genome on a benchmark."""

    genome_id: str
    benchmark_id: str
    task_results: list[TaskResult] = Field(default_factory=list)
    composite_score: float = 0.0
    passed: bool = False


def evaluate_task(genome: PromptGenome, task: BenchmarkTask) -> TaskResult:
    """Evaluate a genome on a single task.

    This is a simplified evaluation: check if the genome's system prompt
    contains keywords related to the task.
    """
    # Simple keyword matching evaluation
    prompt_lower = genome.system_prompt.lower()
    expected_lower = task.expected_output.lower()

    # Check for keyword overlap
    expected_words = set(expected_lower.split())
    prompt_words = set(prompt_lower.split())
    overlap = expected_words & prompt_words

    score = len(overlap) / max(len(expected_words), 1)
    passed = score >= 0.5

    return TaskResult(
        task_id=task.id,
        passed=passed,
        score=min(score, 1.0),
        output=genome.system_prompt[:200],
        expected=task.expected_output,
        runtime=0.1,
    )


def evaluate_benchmark(genome: PromptGenome, benchmark: Benchmark) -> BenchmarkResult:
    """Evaluate a genome on a full benchmark."""
    task_results = []
    for task in benchmark.tasks:
        result = evaluate_task(genome, task)
        task_results.append(result)

    composite_score = sum(r.score for r in task_results) / max(len(task_results), 1)
    passed = composite_score >= 0.6

    return BenchmarkResult(
        genome_id=genome.id,
        benchmark_id=benchmark.id,
        task_results=task_results,
        composite_score=composite_score,
        passed=passed,
    )


# Built-in benchmarks
def create_plan_simple_benchmark() -> Benchmark:
    """Create a simple planning benchmark."""
    return Benchmark(
        id="plan-simple",
        name="Simple Planning",
        description="Tests basic planning capabilities",
        tasks=[
            BenchmarkTask(
                id="task_1",
                name="Simple Plan",
                description="Create a simple plan",
                expected_output="plan execute verify",
                difficulty="easy",
            ),
            BenchmarkTask(
                id="task_2",
                name="Multi-step Plan",
                description="Create a multi-step plan",
                expected_output="research implement test deploy",
                difficulty="medium",
            ),
        ],
    )


def create_error_recovery_benchmark() -> Benchmark:
    """Create an error recovery benchmark."""
    return Benchmark(
        id="error-recovery",
        name="Error Recovery",
        description="Tests error recovery capabilities",
        tasks=[
            BenchmarkTask(
                id="task_1",
                name="Handle Failure",
                description="Recover from a tool failure",
                expected_output="retry fallback replan",
                difficulty="medium",
            ),
        ],
    )


def create_budget_stress_benchmark() -> Benchmark:
    """Create a budget stress benchmark."""
    return Benchmark(
        id="budget-stress",
        name="Budget Stress",
        description="Tests performance under tight budgets",
        tasks=[
            BenchmarkTask(
                id="task_1",
                name="Tight Budget",
                description="Complete task with minimal resources",
                expected_output="efficient optimize minimize",
                difficulty="hard",
            ),
        ],
    )
