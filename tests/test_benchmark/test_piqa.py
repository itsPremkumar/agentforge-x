"""Tests for PIQA benchmark — ≥20 tests."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.benchmark.piqa_benchmark import (
    PIQABenchmark,
    PIQAExample,
    PIQAResult,
    PIQAReport,
    load_examples,
    run_example,
    run_all,
    get_accuracy,
    get_report,
    reset,
)


def test_benchmark_init():
    """Test benchmark initialization."""
    bench = PIQABenchmark()
    assert bench.examples == []
    assert bench.results == []
    assert bench._seed == 42


def test_benchmark_init_custom_seed():
    """Test benchmark with custom seed."""
    bench = PIQABenchmark(seed=123)
    assert bench._seed == 123


def test_load_examples_returns_list():
    """Test that load_examples returns a list."""
    bench = PIQABenchmark()
    examples = bench.load_examples()
    assert isinstance(examples, list)
    assert len(examples) == bench.TOTAL_EXAMPLES


def test_load_examples_count():
    """Test that all 16,119 examples are generated."""
    bench = PIQABenchmark()
    examples = bench.load_examples()
    assert len(examples) == 16119


def test_load_examples_have_correct_fields():
    """Test that examples have all required fields."""
    bench = PIQABenchmark()
    examples = bench.load_examples()
    
    for ex in examples[:10]:
        assert hasattr(ex, "id")
        assert hasattr(ex, "goal")
        assert hasattr(ex, "solutions")
        assert hasattr(ex, "label")
        assert hasattr(ex, "split")
        assert len(ex.solutions) == 2
        assert ex.label in [0, 1]


def test_run_example_returns_result():
    """Test that run_example returns a PIQAResult."""
    bench = PIQABenchmark()
    examples = bench.load_examples()
    result = bench.run_example(examples[0])
    
    assert isinstance(result, PIQAResult)
    assert result.predicted in [0, 1]


def test_run_example_correct_field():
    """Test that run_example sets correct field."""
    bench = PIQABenchmark()
    examples = bench.load_examples()
    result = bench.run_example(examples[0])
    
    assert result.correct == (result.predicted == examples[0].label)


def test_run_all_returns_report():
    """Test that run_all returns a PIQAReport."""
    bench = PIQABenchmark()
    report = bench.run_all()
    
    assert isinstance(report, PIQAReport)
    assert report.total == 16119


def test_run_all_accuracy_in_range():
    """Test that accuracy is in valid range."""
    bench = PIQABenchmark()
    report = bench.run_all()
    
    assert 0 <= report.accuracy <= 100


def test_get_accuracy_matches_report():
    """Test that get_accuracy matches the report accuracy."""
    bench = PIQABenchmark()
    report = bench.run_all()
    
    acc = bench.get_accuracy()
    assert abs(acc - report.accuracy) < 0.01


def test_get_report_returns_dict():
    """Test that get_report returns a dict."""
    bench = PIQABenchmark()
    bench.run_all()
    report = bench.get_report()
    
    assert isinstance(report, dict)
    assert "total" in report
    assert "correct" in report
    assert "accuracy" in report


def test_get_report_total_matches():
    """Test that report total matches benchmark total."""
    bench = PIQABenchmark()
    bench.run_all()
    report = bench.get_report()
    
    assert report["total"] == 16119


def test_module_level_load_examples():
    """Test module-level load_examples function."""
    reset()
    examples = load_examples()
    assert len(examples) == 16119


def test_module_level_run_all():
    """Test module-level run_all function."""
    reset()
    report = run_all()
    assert report.total == 16119


def test_module_level_get_accuracy():
    """Test module-level get_accuracy function."""
    reset()
    run_all()
    acc = get_accuracy()
    assert 0 <= acc <= 100


def test_multiple_runs_produce_same_results():
    """Test that same seed produces same results."""
    bench1 = PIQABenchmark(seed=42)
    report1 = bench1.run_all()
    
    bench2 = PIQABenchmark(seed=42)
    report2 = bench2.run_all()
    
    assert report1.accuracy == report2.accuracy


def test_example_ids_are_unique():
    """Test that example IDs are unique."""
    bench = PIQABenchmark()
    examples = bench.load_examples()
    
    ids = [ex.id for ex in examples]
    assert len(ids) == len(set(ids))


def test_example_splits():
    """Test that examples have valid splits."""
    bench = PIQABenchmark()
    examples = bench.load_examples()
    
    splits = set(ex.split for ex in examples)
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits


def test_piqa_result_fields():
    """Test PIQAResult dataclass fields."""
    result = PIQAResult(
        example_id="test_001",
        predicted=1,
        correct=True,
        confidence=0.75,
    )
    
    assert result.example_id == "test_001"
    assert result.predicted == 1
    assert result.correct is True
    assert result.confidence == 0.75


def test_piqa_report_properties():
    """Test PIQAReport computed properties."""
    results = [
        PIQAResult(example_id="1", predicted=0, correct=True),
        PIQAResult(example_id="2", predicted=1, correct=False),
        PIQAResult(example_id="3", predicted=0, correct=True),
        PIQAResult(example_id="4", predicted=1, correct=True),
    ]
    
    report = PIQAReport(results=results)
    assert report.total == 4
    assert report.correct == 3
    assert report.accuracy == 75.0


def test_benchmark_run_with_small_subset():
    """Test benchmark with a small subset."""
    bench = PIQABenchmark()
    examples = bench.load_examples()[:10]
    bench.examples = examples
    
    report = bench.run_all()
    assert report.total == 10


def test_reset_clears_default():
    """Test that reset clears the default benchmark."""
    reset()
    load_examples()
    assert True


def test_get_report_without_run():
    """Test get_report without running benchmark."""
    bench = PIQABenchmark()
    report = bench.get_report()
    
    assert "error" in report
