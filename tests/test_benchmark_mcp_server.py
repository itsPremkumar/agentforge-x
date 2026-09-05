#!/usr/bin/env python3
"""Tests for benchmark_mcp_server."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from af_mcp.benchmark_mcp_server import (
    list_benchmarks,
    create_benchmark,
    add_task,
    run_benchmark,
    evaluate_task_tool,
    compare_results,
    get_benchmark_status,
    save_benchmark,
    load_benchmark,
    search_benchmarks,
    delete_benchmark,
    export_benchmark,
    get_benchmark,
    build_server,
    _BENCHMARKS,
    _RESULTS,
)


@pytest.fixture(autouse=True)
def clear_state():
    """Clear global state before each test."""
    _BENCHMARKS.clear()
    _RESULTS.clear()
    yield
    _BENCHMARKS.clear()
    _RESULTS.clear()


class TestListBenchmarks:
    def test_returns_builtins(self):
        result = list_benchmarks()
        assert result["total"] >= 3
        ids = [b["id"] for b in result["benchmarks"]]
        assert "plan-simple" in ids
        assert "error-recovery" in ids
        assert "budget-stress" in ids

    def test_includes_custom(self):
        create_benchmark("Custom Test", "A custom benchmark")
        result = list_benchmarks()
        ids = [b["id"] for b in result["benchmarks"]]
        assert any("Custom" in b["name"] for b in result["benchmarks"])


class TestCreateBenchmark:
    def test_creates_with_auto_id(self):
        result = create_benchmark("Test Bench", "Description")
        assert "id" in result
        assert result["name"] == "Test Bench"
        assert result["status"] == "created"

    def test_creates_with_custom_id(self):
        result = create_benchmark("Test", "Desc", benchmark_id="my-bench")
        assert result["id"] == "my-bench"

    def test_duplicate_id_returns_error(self):
        create_benchmark("Test", "Desc", benchmark_id="dup")
        result = create_benchmark("Test2", "Desc2", benchmark_id="dup")
        assert "isError" in result


class TestAddTask:
    def test_adds_to_existing(self):
        create_benchmark("Test", "", benchmark_id="b1")
        result = add_task("b1", "Task 1", "Do something", "expected output")
        assert result["task_id"] == "task-1"
        assert result["total_tasks"] == 1

    def test_adds_multiple(self):
        create_benchmark("Test", "", benchmark_id="b1")
        add_task("b1", "Task 1", "Desc 1", "output 1")
        result = add_task("b1", "Task 2", "Desc 2", "output 2")
        assert result["total_tasks"] == 2

    def test_missing_benchmark(self):
        result = add_task("nonexistent", "Task", "Desc", "output")
        assert "isError" in result


class TestRunBenchmark:
    def test_runs_with_tasks(self):
        create_benchmark("Test", "", benchmark_id="b1")
        add_task("b1", "Task 1", "Desc", "plan execute verify")
        result = run_benchmark("b1", "plan execute verify test")
        assert result["passed"] is True
        assert result["composite_score"] > 0.5

    def test_fails_with_low_score(self):
        create_benchmark("Test", "", benchmark_id="b1")
        add_task("b1", "Task 1", "Desc", "plan execute verify")
        result = run_benchmark("b1", "unrelated text here")
        assert result["passed"] is False

    def test_missing_benchmark(self):
        result = run_benchmark("nonexistent", "text")
        assert "isError" in result

    def test_no_tasks(self):
        create_benchmark("Test", "", benchmark_id="empty")
        result = run_benchmark("empty", "text")
        assert "isError" in result


class TestEvaluateTask:
    def test_evaluates_single(self):
        create_benchmark("Test", "", benchmark_id="b1")
        add_task("b1", "Task 1", "Desc", "plan execute")
        result = evaluate_task_tool("b1", "task-1", "plan execute verify")
        assert result["task"]["passed"] is True

    def test_missing_benchmark(self):
        result = evaluate_task_tool("nonexistent", "task-1", "text")
        assert "isError" in result

    def test_missing_task(self):
        create_benchmark("Test", "", benchmark_id="b1")
        result = evaluate_task_tool("b1", "nonexistent", "text")
        assert "isError" in result


class TestCompareResults:
    def test_compares_two_genomes(self):
        create_benchmark("Test", "", benchmark_id="b1")
        add_task("b1", "Task 1", "Desc", "plan execute")
        result = compare_results("b1", "plan execute verify", "unrelated text")
        assert "genome_1" in result
        assert "genome_2" in result
        assert result["winner"] == "genome_1"

    def test_missing_benchmark(self):
        result = compare_results("nonexistent", "text1", "text2")
        assert "isError" in result


class TestGetBenchmarkStatus:
    def test_not_run(self):
        create_benchmark("Test", "", benchmark_id="b1")
        result = get_benchmark_status("b1")
        assert result["status"] == "not_run"

    def test_completed(self):
        create_benchmark("Test", "", benchmark_id="b1")
        add_task("b1", "Task 1", "Desc", "plan execute")
        run_benchmark("b1", "plan execute verify")
        result = get_benchmark_status("b1")
        assert result["status"] in ("completed", "failed")

    def test_missing_benchmark(self):
        result = get_benchmark_status("nonexistent")
        assert "isError" in result


class TestSaveBenchmark:
    def test_saves_to_default_path(self):
        create_benchmark("Test", "", benchmark_id="b1")
        result = save_benchmark("b1")
        assert result["status"] == "saved"
        assert Path(result["path"]).exists()

    def test_saves_to_custom_path(self):
        create_benchmark("Test", "", benchmark_id="b1")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        result = save_benchmark("b1", filepath=path)
        assert result["status"] == "saved"
        assert Path(path).exists()
        os.unlink(path)

    def test_missing_benchmark(self):
        result = save_benchmark("nonexistent")
        assert "isError" in result


class TestLoadBenchmark:
    def test_loads_from_file(self):
        create_benchmark("Test", "Description", benchmark_id="b1")
        add_task("b1", "Task 1", "Desc", "output")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({
                "id": "loaded-bench",
                "name": "Loaded",
                "description": "Loaded benchmark",
                "timeout": 60.0,
                "tasks": [
                    {"id": "t1", "name": "Task 1", "description": "Desc", "expected_output": "output", "difficulty": "medium", "max_runtime": 60.0}
                ],
            }, f)
            path = f.name
        result = load_benchmark(path)
        assert result["id"] == "loaded-bench"
        assert result["task_count"] == 1
        os.unlink(path)

    def test_missing_file(self):
        result = load_benchmark("/nonexistent/path.json")
        assert "isError" in result


class TestSearchBenchmarks:
    def test_finds_by_name(self):
        create_benchmark("Planning Test", "", benchmark_id="b1")
        create_benchmark("Recovery Test", "", benchmark_id="b2")
        result = search_benchmarks("planning")
        assert result["count"] == 1

    def test_finds_by_description(self):
        create_benchmark("Test", "Error recovery benchmark", benchmark_id="b1")
        result = search_benchmarks("recovery")
        assert result["count"] == 1

    def test_no_results(self):
        create_benchmark("Test", "", benchmark_id="b1")
        result = search_benchmarks("nonexistent")
        assert result["count"] == 0


class TestDeleteBenchmark:
    def test_deletes_existing(self):
        create_benchmark("Test", "", benchmark_id="b1")
        result = delete_benchmark("b1")
        assert result["status"] == "deleted"
        assert "b1" not in _BENCHMARKS

    def test_missing_benchmark(self):
        result = delete_benchmark("nonexistent")
        assert "isError" in result


class TestExportBenchmark:
    def test_exports_as_json(self):
        create_benchmark("Test", "Description", benchmark_id="b1")
        result = export_benchmark("b1")
        assert "data" in result
        data = json.loads(result["data"])
        assert data["id"] == "b1"

    def test_missing_benchmark(self):
        result = export_benchmark("nonexistent")
        assert "isError" in result


class TestGetBenchmark:
    def test_gets_existing(self):
        create_benchmark("Test", "Description", benchmark_id="b1")
        result = get_benchmark("b1")
        assert result["id"] == "b1"
        assert result["name"] == "Test"

    def test_missing_benchmark(self):
        result = get_benchmark("nonexistent")
        assert "isError" in result


class TestMCPServer:
    def test_build_server(self):
        server = build_server()
        assert server is not None
        assert server.name == "benchmark-mcp-server"

    def test_list_tools(self):
        async def run():
            server = build_server()
            tools = await server.list_tools()
            tool_names = [t.name for t in tools]
            assert "list_benchmarks" in tool_names
            assert "create_benchmark" in tool_names
            assert "run_benchmark" in tool_names
        import asyncio
        asyncio.run(run())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
