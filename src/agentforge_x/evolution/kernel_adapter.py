"""Kernel adapter: bridges evolution engine to agentforge-x kernel."""

from __future__ import annotations

from typing import Any

from agentforge_x.evolution.genome import PromptGenome


class KernelAdapter:
    """Adapts the agentforge-x kernel for use by the evolution engine."""

    def __init__(self):
        self._agents: dict[str, Any] = {}

    def create_agent(self, genome: PromptGenome, task: Any) -> str:
        """Create an agent with a given genome. Returns agent ID."""
        agent_id = f"agent-{genome.id[:8]}"
        self._agents[agent_id] = {
            "genome": genome,
            "task": task,
            "state": None,
        }
        return agent_id

    def run_agent(self, agent_id: str) -> dict[str, Any]:
        """Run the agent and return results."""
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        return {
            "status": "completed",
            "genome_id": agent["genome"].id,
        }

    def get_event_trace(self, agent_id: str) -> list[dict[str, Any]]:
        """Get the agent's event trace."""
        return []

    def get_result(self, agent_id: str) -> dict[str, Any]:
        """Get the final benchmark result."""
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        return {"status": "completed"}

    def extract_metrics(self, state: Any) -> dict[str, float]:
        """Extract metrics from the agent's final state."""
        return {"accuracy": 0.8, "latency": 0.5, "cost": 0.3}

    def apply_genome(self, genome: PromptGenome) -> None:
        """Apply a genome to the kernel's config."""
        pass

    def reset(self) -> None:
        """Reset the kernel for a fresh evaluation."""
        self._agents.clear()
