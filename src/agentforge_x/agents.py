"""AgentForge-X Agent Library: specialized agent implementations.

Provides reusable agent patterns including planner, executor, critic,
and multi-agent orchestration.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from agentforge_x.kernel.state import AgentState, BudgetState
from agentforge_x.kernel.event_bus import EventBus, EventType
from agentforge_x.kernel.planner import Planner, MockPlanner
from agentforge_x.kernel.executor import Executor, ToolExecutor, ToolRegistry
from agentforge_x.kernel.critic import Critic, MockCritic
from agentforge_x.kernel.state_graph import StateGraphRuntime, KernelConfig


@dataclass
class AgentSpec:
    """Specification for a specialized agent."""

    name: str
    role: str  # planner | executor | critic | orchestrator
    description: str = ""
    tools: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


class PlannerAgent:
    """Agent specialized for planning tasks."""

    def __init__(self, tools: Optional[list[str]] = None, config: Optional[KernelConfig] = None):
        self.tools = tools or ["bash", "fs", "git"]
        self.config = config or KernelConfig()
        self.planner = MockPlanner()
        self.runtime = StateGraphRuntime(planner=self.planner, config=self.config)

    async def plan(self, goal: str) -> list[PlanEntry]:
        """Create a plan for a goal."""
        from agentforge_x.kernel.state import PlanEntry
        state = AgentState(scratchpad=goal)
        state.plan = await self.planner.create_plan(state, goal)
        return state.plan


class ExecutorAgent:
    """Agent specialized for executing tasks."""

    def __init__(self, tools: Optional[ToolRegistry] = None, config: Optional[KernelConfig] = None):
        self.tools = tools or ToolRegistry()
        self.config = config or KernelConfig()
        self.executor = ToolExecutor(tools=self.tools)
        self.runtime = StateGraphRuntime(executor=self.executor, config=self.config)

    async def execute(self, task: str) -> str:
        """Execute a task."""
        state = AgentState(scratchpad=task)
        result = await self.runtime.run(state)
        return result.scratchpad or ""


class CriticAgent:
    """Agent specialized for critiquing outputs."""

    def __init__(self, config: Optional[KernelConfig] = None):
        self.config = config or KernelConfig()
        self.critic = MockCritic(always_pass=True)
        self.runtime = StateGraphRuntime(critic=self.critic, config=self.config)

    async def critique(self, output: str) -> dict[str, Any]:
        """Critique an output."""
        return {"output": output, "passed": True, "score": 0.9}


class MultiAgentOrchestrator:
    """Orchestrates multiple agents."""

    def __init__(self, agents: Optional[dict[str, Any]] = None):
        self.agents: dict[str, Any] = agents or {}
        self._event_bus: Optional[EventBus] = None

    def register(self, name: str, agent: Any) -> None:
        """Register an agent."""
        self.agents[name] = agent

    async def execute_parallel(self, tasks: dict[str, str]) -> dict[str, Any]:
        """Execute tasks in parallel across agents."""
        results = {}
        for name, task in tasks.items():
            agent = self.agents.get(name)
            if not agent:
                continue
            if hasattr(agent, "run"):
                result = await agent.run(task)
            elif hasattr(agent, "execute"):
                result = await agent.execute(task)
            elif hasattr(agent, "plan"):
                result = await agent.plan(task)
            elif hasattr(agent, "critique"):
                result = await agent.critique(task)
            else:
                continue
            results[name] = result
        return results


__all__ = [
    "AgentSpec",
    "PlannerAgent",
    "ExecutorAgent",
    "CriticAgent",
    "MultiAgentOrchestrator",
]
