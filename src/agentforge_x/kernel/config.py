"""Kernel configuration."""

from __future__ import annotations

from pydantic import BaseModel


class KernelConfig(BaseModel):
    """Configuration for the AgentForge-X kernel."""

    # Budget defaults
    max_runtime: float = 600.0
    max_llm_calls: int = 100
    max_tokens: int = 100000

    # Execution
    max_retries: int = 3
    max_plan_steps: int = 100
    enable_subgraph_spawning: bool = True
    checkpoint_interval: int = 1

    # Overhead
    budget_overhead_per_spawn: float = 5.0  # seconds

    model_config = {"arbitrary_types_allowed": True}
