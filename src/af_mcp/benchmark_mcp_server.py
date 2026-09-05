#!/usr/bin/env python3
"""benchmark_mcp_server.py — MCP server for benchmark operations.

Tools:
  - list_benchmarks: list available benchmark suites
  - create_benchmark: create a new benchmark suite
  - add_task: add a task to a benchmark
  - run_benchmark: execute a benchmark against a genome
  - evaluate_task: evaluate a single task
  - compare_results: compare two benchmark results
  - get_benchmark_status: check benchmark progress
  - save_benchmark: persist benchmark to disk
  - load_benchmark: load benchmark from disk
  - search_benchmarks: search benchmarks by tag/name
  - delete_benchmark: remove a benchmark
  - export_benchmark: export benchmark as JSON

Run: python -m af_mcp.benchmark_mcp_server
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from mcp.server import MCPServer

from .models import (
    Benchmark,
    BenchmarkTask,
    TaskResult,
    BenchmarkResult,
    EntityType,
)

log = logging.getLogger(__name__)

# In-memory store
_BENCHMARKS: dict[str, Benchmark] = {}
_RESULTS: dict[str, BenchmarkResult] = {}


def _get_storage_path() -> Path:
    env_path = os.environ.get("BENCHMARK_STORAGE_PATH")
    if env_path:
        return Path(env_path)
    return Path.home() / ".agentforge" / "benchmarks"


def list_benchmarks() -> dict:
    """List all available benchmark suites."""
    builtins = [
        {"id": "plan-simple", "name": "Simple Planning", "description": "Tests basic planning capabilities", "task_count": 2},
        {"id": "error-recovery", "name": "Error Recovery", "description": "Tests error recovery capabilities", "task_count": 1},
        {"id": "budget-stress", "name": "Budget Stress", "description": "Tests performance under tight budgets", "task_count": 1},
    ]
    custom = []
    for bid, bench in _BENCHMARKS.items():
        custom.append({
            "id": bench.id,
            "name": bench.name,
            "description": bench.description,
            "task_count": len(bench.tasks),
        })
    all_benchs = builtins + custom
    return {"benchmarks": all_benchs, "total": len(all_benchs)}


def create_benchmark(
    name: str,
    description: str = "",
    benchmark_id: Optional[str] = None,
) -> dict:
    """Create a new benchmark suite."""
    bid = benchmark_id or f"bench-{uuid.uuid4().hex[:8]}"
    if bid in _BENCHMARKS:
        return {"isError": True, "content": [{"type": "text", "text": f"Benchmark ID '{bid}' already exists"}]}
    bench = Benchmark(id=bid, name=name, description=description)
    _BENCHMARKS[bid] = bench
    return {"id": bid, "name": name, "status": "created"}


def add_task(
    benchmark_id: str,
    task_name: str,
    description: str,
    expected_output: str,
    difficulty: str = "medium",
    max_runtime: float = 60.0,
) -> dict:
    """Add a task to a benchmark."""
    bench = _BENCHMARKS.get(benchmark_id)
    if not bench:
        return {"isError": True, "content": [{"type": "text", "text": f"Benchmark '{benchmark_id}' not found"}]}
    task = BenchmarkTask(
        id=f"task-{len(bench.tasks) + 1}",
        name=task_name,
        description=description,
        expected_output=expected_output,
        difficulty=difficulty,
        max_runtime=max_runtime,
    )
    bench.tasks.append(task)
    return {"task_id": task.id, "benchmark_id": benchmark_id, "total_tasks": len(bench.tasks)}


def run_benchmark(benchmark_id: str, genome_text: str) -> dict:
    """Run a benchmark against a genome (text)."""
    bench = _BENCHMARKS.get(benchmark_id)
    if not bench:
        return {"isError": True, "content": [{"type": "text", "text": f"Benchmark '{benchmark_id}' not found"}]}
    if not bench.tasks:
        return {"isError": True, "content": [{"type": "text", "text": f"Benchmark '{benchmark_id}' has no tasks"}]}

    task_results = []
    for task in bench.tasks:
        result = _evaluate_task(genome_text, task)
        task_results.append(result)

    composite_score = sum(r["score"] for r in task_results) / max(len(task_results), 1)
    passed = composite_score >= 0.6

    result = {
        "genome_id": f"genome-{uuid.uuid4().hex[:8]}",
        "benchmark_id": benchmark_id,
        "composite_score": composite_score,
        "passed": passed,
        "task_count": len(task_results),
        "task_results": task_results,
    }
    _RESULTS[benchmark_id] = result
    return result


def _evaluate_task(genome_text: str, task: BenchmarkTask) -> dict:
    """Evaluate a genome text against a task."""
    prompt_lower = genome_text.lower()
    expected_lower = task.expected_output.lower()
    expected_words = set(expected_lower.split())
    prompt_words = set(prompt_lower.split())
    overlap = expected_words & prompt_words
    score = len(overlap) / max(len(expected_words), 1)
    passed = score >= 0.5
    return {
        "task_id": task.id,
        "task_name": task.name,
        "passed": passed,
        "score": min(score, 1.0),
        "expected": task.expected_output,
        "difficulty": task.difficulty,
    }


def evaluate_task_tool(benchmark_id: str, task_id: str, genome_text: str) -> dict:
    """Evaluate a single task from a benchmark."""
    bench = _BENCHMARKS.get(benchmark_id)
    if not bench:
        return {"isError": True, "content": [{"type": "text", "text": f"Benchmark '{benchmark_id}' not found"}]}
    task = next((t for t in bench.tasks if t.id == task_id), None)
    if not task:
        return {"isError": True, "content": [{"type": "text", "text": f"Task '{task_id}' not found"}]}
    result = _evaluate_task(genome_text, task)
    return {"benchmark_id": benchmark_id, "task": result}


def compare_results(benchmark_id: str, genome_text_1: str, genome_text_2: str) -> dict:
    """Compare two genomes on the same benchmark."""
    bench = _BENCHMARKS.get(benchmark_id)
    if not bench:
        return {"isError": True, "content": [{"type": "text", "text": f"Benchmark '{benchmark_id}' not found"}]}
    result_1 = run_benchmark(benchmark_id, genome_text_1)
    result_2 = run_benchmark(benchmark_id, genome_text_2)
    return {
        "benchmark_id": benchmark_id,
        "genome_1": {"score": result_1["composite_score"], "passed": result_1["passed"]},
        "genome_2": {"score": result_2["composite_score"], "passed": result_2["passed"]},
        "winner": "genome_1" if result_1["composite_score"] > result_2["composite_score"] else "genome_2",
    }


def get_benchmark_status(benchmark_id: str) -> dict:
    """Get status of a benchmark."""
    bench = _BENCHMARKS.get(benchmark_id)
    if not bench:
        return {"isError": True, "content": [{"type": "text", "text": f"Benchmark '{benchmark_id}' not found"}]}
    result = _RESULTS.get(benchmark_id)
    if not result:
        return {"benchmark_id": benchmark_id, "status": "not_run", "task_count": len(bench.tasks)}
    return {
        "benchmark_id": benchmark_id,
        "status": "completed" if result["passed"] else "failed",
        "composite_score": result["composite_score"],
        "tasks_completed": result["task_count"],
    }


def save_benchmark(benchmark_id: str, filepath: Optional[str] = None) -> dict:
    """Save a benchmark to disk."""
    bench = _BENCHMARKS.get(benchmark_id)
    if not bench:
        return {"isError": True, "content": [{"type": "text", "text": f"Benchmark '{benchmark_id}' not found"}]}
    storage = _get_storage_path()
    storage.mkdir(parents=True, exist_ok=True)
    path = storage / f"{benchmark_id}.json"
    if filepath:
        path = Path(filepath)
    data = {
        "id": bench.id,
        "name": bench.name,
        "description": bench.description,
        "timeout": bench.timeout,
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "expected_output": t.expected_output,
                "difficulty": t.difficulty,
                "max_runtime": t.max_runtime,
            }
            for t in bench.tasks
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return {"id": benchmark_id, "path": str(path), "status": "saved"}


def load_benchmark(filepath: str) -> dict:
    """Load a benchmark from disk."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        tasks = [
            BenchmarkTask(
                id=t["id"],
                name=t["name"],
                description=t.get("description", ""),
                expected_output=t.get("expected_output", ""),
                difficulty=t.get("difficulty", "medium"),
                max_runtime=t.get("max_runtime", 60.0),
            )
            for t in data.get("tasks", [])
        ]
        bench = Benchmark(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            timeout=data.get("timeout", 60.0),
            tasks=tasks,
        )
        _BENCHMARKS[bench.id] = bench
        return {"id": bench.id, "name": bench.name, "task_count": len(bench.tasks)}
    except Exception as exc:
        return {"isError": True, "content": [{"type": "text", "text": f"Failed to load: {exc}"}]}


def search_benchmarks(query: str) -> dict:
    """Search benchmarks by name or description."""
    query_lower = query.lower()
    results = []
    for bid, bench in _BENCHMARKS.items():
        if query_lower in bench.name.lower() or query_lower in bench.description.lower():
            results.append({"id": bid, "name": bench.name, "task_count": len(bench.tasks)})
    return {"results": results, "count": len(results)}


def delete_benchmark(benchmark_id: str) -> dict:
    """Delete a benchmark."""
    if benchmark_id not in _BENCHMARKS:
        return {"isError": True, "content": [{"type": "text", "text": f"Benchmark '{benchmark_id}' not found"}]}
    del _BENCHMARKS[benchmark_id]
    _RESULTS.pop(benchmark_id, None)
    return {"id": benchmark_id, "status": "deleted"}


def export_benchmark(benchmark_id: str) -> dict:
    """Export a benchmark as JSON."""
    bench = _BENCHMARKS.get(benchmark_id)
    if not bench:
        return {"isError": True, "content": [{"type": "text", "text": f"Benchmark '{benchmark_id}' not found"}]}
    return {
        "id": bench.id,
        "name": bench.name,
        "description": bench.description,
        "tasks": len(bench.tasks),
        "data": json.dumps({
            "id": bench.id,
            "name": bench.name,
            "description": bench.description,
            "tasks": [t.model_dump() for t in bench.tasks],
        }, indent=2),
    }


def get_benchmark(benchmark_id: str) -> dict:
    """Get a specific benchmark's details."""
    bench = _BENCHMARKS.get(benchmark_id)
    if not bench:
        return {"isError": True, "content": [{"type": "text", "text": f"Benchmark '{benchmark_id}' not found"}]}
    return {
        "id": bench.id,
        "name": bench.name,
        "description": bench.description,
        "timeout": bench.timeout,
        "tasks": [t.model_dump() for t in bench.tasks],
    }


# ─── Server assembly ────────────────────────────────────────────────

def build_server() -> MCPServer:
    server = MCPServer(
        name="benchmark-mcp-server",
        title="Benchmark MCP Server",
        description="MCP server for benchmark operations",
        version="0.1.0",
    )

    server.add_tool(list_benchmarks, name="list_benchmarks", description="List all available benchmark suites")
    server.add_tool(create_benchmark, name="create_benchmark", description="Create a new benchmark suite")
    server.add_tool(add_task, name="add_task", description="Add a task to a benchmark")
    server.add_tool(run_benchmark, name="run_benchmark", description="Run a benchmark against a genome")
    server.add_tool(evaluate_task_tool, name="evaluate_task", description="Evaluate a single task")
    server.add_tool(compare_results, name="compare_results", description="Compare two benchmark results")
    server.add_tool(get_benchmark_status, name="get_benchmark_status", description="Get status of a benchmark")
    server.add_tool(save_benchmark, name="save_benchmark", description="Save a benchmark to disk")
    server.add_tool(load_benchmark, name="load_benchmark", description="Load a benchmark from disk")
    server.add_tool(search_benchmarks, name="search_benchmarks", description="Search benchmarks by name/description")
    server.add_tool(delete_benchmark, name="delete_benchmark", description="Delete a benchmark")
    server.add_tool(export_benchmark, name="export_benchmark", description="Export a benchmark as JSON")
    server.add_tool(get_benchmark, name="get_benchmark", description="Get a specific benchmark's details")

    return server


def main():
    import sys
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
