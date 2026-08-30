"""Executor interface and ToolExecutor implementation.

The Executor runs plan steps against available tools. The default
ToolExecutor uses a ToolRegistry. A MockExecutor is provided for testing.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from pydantic import BaseModel, Field

from agentforge_x.kernel.state import AgentState, ArtifactRef, PlanEntry


class ExecutionResult(BaseModel):
    """Result of executing a plan step."""

    success: bool
    output: str = ""
    duration_ms: float = 0.0
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    error: str | None = None
    retryable: bool = True


class ToolCall(BaseModel):
    """A tool call specification."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ToolRegistry:
    """Registry of callable tools."""

    def __init__(self):
        self._tools: dict[str, Any] = {}

    def register(self, name: str, func: Any) -> None:
        self._tools[name] = func

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Any:
        return self._tools[name]

    def list(self) -> list[str]:
        return list(self._tools.keys())


class Executor:
    """Interface for executing plan steps."""

    async def execute(self, state: AgentState, step: PlanEntry) -> ExecutionResult:
        raise NotImplementedError

    def can_execute(self, step: PlanEntry) -> bool:
        raise NotImplementedError


class ToolExecutor(Executor):
    """Executes plan steps against a ToolRegistry."""

    def __init__(self, tools: ToolRegistry | None = None):
        self.tools = tools or ToolRegistry()

    def can_execute(self, step: PlanEntry) -> bool:
        return self.tools.has(step.tool)

    async def execute(self, state: AgentState, step: PlanEntry) -> ExecutionResult:
        if not self.can_execute(step):
            return ExecutionResult(
                success=False,
                output="",
                error=f"Tool not found: {step.tool}",
                retryable=False,
            )

        func = self.tools.get(step.tool)
        str(uuid.uuid4())
        start = asyncio.get_event_loop().time()

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(step.args)
            else:
                result = func(step.args)

            duration = (asyncio.get_event_loop().time() - start) * 1000

            artifacts = []
            if isinstance(result, dict) and "artifact" in result:
                artifacts.append(ArtifactRef(
                    id=str(uuid.uuid4()),
                    type=result.get("artifact_type", "data"),
                    path=result.get("path", ""),
                ))

            return ExecutionResult(
                success=True,
                output=str(result),
                duration_ms=duration,
                artifacts=artifacts,
            )
        except Exception as e:
            duration = (asyncio.get_event_loop().time() - start) * 1000
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                duration_ms=duration,
                retryable=True,
            )


class MockExecutor(Executor):
    """Deterministic executor for testing."""

    def __init__(self, results: list[ExecutionResult] | None = None):
        self._results = results or []
        self._call_idx = 0

    def can_execute(self, step: PlanEntry) -> bool:
        return True

    async def execute(self, state: AgentState, step: PlanEntry) -> ExecutionResult:
        if self._call_idx < len(self._results):
            result = self._results[self._call_idx]
        else:
            result = ExecutionResult(success=True, output=f"Mock result for {step.id}")
        self._call_idx += 1
        return result
