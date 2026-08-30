"""AgentForge-X Core: unified runtime orchestration."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from agentforge_x.kernel.state import AgentState, BudgetState, PlanEntry
from agentforge_x.kernel.event_bus import EventBus, EventType
from agentforge_x.kernel.state_graph import StateGraphRuntime, KernelConfig
from agentforge_x.evolution.genome import PromptGenome
from agentforge_x.evolution.loop import EvolutionConfig, EvolutionLoop


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""

    name: str
    genome: Optional[PromptGenome] = None
    budget: Optional[BudgetState] = None
    max_retries: int = 3
    timeout: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentInstance:
    """A running agent instance with lifecycle management."""

    def __init__(self, config: AgentConfig, runtime: Optional[StateGraphRuntime] = None):
        self.config = config
        self.runtime = runtime or StateGraphRuntime()
        self.state = AgentState(
            agent_id=f"agent-{uuid.uuid4().hex[:8]}",
            budget=config.budget or BudgetState(),
        )
        self.status = "idle"
        self.created_at = 0.0
        self.completed_at: Optional[float] = None

    async def run(self, task: str, events: Optional[EventBus] = None) -> AgentState:
        """Run the agent on a task."""
        self.status = "running"
        self.state.scratchpad = task
        result = await self.runtime.run(self.state, events=events)
        self.status = "completed"
        return result


class CoreKernel:
    """The main entry point for AgentForge-X.

    Orchestrates agents, tools, evolution, and deployment.
    """

    def __init__(self, kernel_config: Optional[KernelConfig] = None):
        self.kernel_config = kernel_config or KernelConfig()
        self.runtime = StateGraphRuntime(config=self.kernel_config)
        self._agents: dict[str, AgentInstance] = {}
        self._event_bus: Optional[EventBus] = None

    def create_agent(self, config: AgentConfig) -> AgentInstance:
        """Create a new agent instance."""
        agent = AgentInstance(config, self.runtime)
        self._agents[agent.state.agent_id] = agent
        return agent

    def get_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> list[str]:
        """List all agent IDs."""
        return list(self._agents.keys())

    async def run_agent(self, agent_id: str, task: str) -> Optional[AgentState]:
        """Run an agent on a task."""
        agent = self._agents.get(agent_id)
        if not agent:
            return None
        if not self._event_bus:
            self._event_bus = EventBus(run_id=agent.state.run_id, agent_id=agent_id)
        return await agent.run(task, events=self._event_bus)

    def start_evolution(self, config: Optional[EvolutionConfig] = None) -> EvolutionLoop:
        """Start the evolution engine."""
        return EvolutionLoop(config=config or EvolutionConfig())

    def get_event_bus(self) -> Optional[EventBus]:
        return self._event_bus


__all__ = [
    "AgentConfig",
    "AgentInstance",
    "CoreKernel",
]
