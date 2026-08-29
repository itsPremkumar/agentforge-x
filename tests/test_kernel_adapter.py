"""Tests for kernel adapter."""

import pytest

from agentforge_x.evolution.genome import PromptGenome
from agentforge_x.evolution.kernel_adapter import KernelAdapter


class TestKernelAdapter:
    """Test kernel adapter."""

    def test_create_agent(self):
        """Test creating an agent."""
        adapter = KernelAdapter()
        genome = PromptGenome()
        agent_id = adapter.create_agent(genome, task=None)
        assert agent_id.startswith("agent-")

    def test_run_agent(self):
        """Test running an agent."""
        adapter = KernelAdapter()
        genome = PromptGenome()
        agent_id = adapter.create_agent(genome, task=None)
        result = adapter.run_agent(agent_id)
        assert result["status"] == "completed"

    def test_get_event_trace(self):
        """Test getting event trace."""
        adapter = KernelAdapter()
        genome = PromptGenome()
        agent_id = adapter.create_agent(genome, task=None)
        events = adapter.get_event_trace(agent_id)
        assert isinstance(events, list)

    def test_get_result(self):
        """Test getting result."""
        adapter = KernelAdapter()
        genome = PromptGenome()
        agent_id = adapter.create_agent(genome, task=None)
        result = adapter.get_result(agent_id)
        assert result["status"] == "completed"

    def test_extract_metrics(self):
        """Test extracting metrics."""
        adapter = KernelAdapter()
        metrics = adapter.extract_metrics(None)
        assert "accuracy" in metrics
        assert "latency" in metrics
        assert "cost" in metrics

    def test_apply_genome(self):
        """Test applying genome."""
        adapter = KernelAdapter()
        genome = PromptGenome()
        adapter.apply_genome(genome)
        # Should not raise

    def test_reset(self):
        """Test reset."""
        adapter = KernelAdapter()
        genome = PromptGenome()
        adapter.create_agent(genome, task=None)
        adapter.reset()
        assert len(adapter._agents) == 0

    def test_run_agent_not_found(self):
        """Test running nonexistent agent."""
        adapter = KernelAdapter()
        with pytest.raises(ValueError):
            adapter.run_agent("nonexistent")

    def test_get_result_not_found(self):
        """Test getting result for nonexistent agent."""
        adapter = KernelAdapter()
        with pytest.raises(ValueError):
            adapter.get_result("nonexistent")
