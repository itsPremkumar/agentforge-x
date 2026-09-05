"""Models for benchmark MCP server."""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field
from typing import Any


class EntityType(str, Enum):
    ISSUE = "issue"
    PRODUCT = "product"
    SOLUTION = "solution"


class BenchmarkTask(BaseModel):
    id: str
    name: str
    description: str
    expected_output: str
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)
    max_runtime: float = 60.0


class Benchmark(BaseModel):
    id: str
    name: str
    description: str
    tasks: list[BenchmarkTask] = Field(default_factory=list)
    timeout: float = 60.0


class TaskResult(BaseModel):
    task_id: str
    passed: bool
    score: float = 0.0
    output: str = ""
    expected: str = ""
    runtime: float = 0.0
    error: str | None = None


class BenchmarkResult(BaseModel):
    genome_id: str
    benchmark_id: str
    task_results: list[TaskResult] = Field(default_factory=list)
    composite_score: float = 0.0
    passed: bool = False
