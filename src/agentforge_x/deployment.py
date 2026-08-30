"""AgentForge-X Deployment: packaging, serving, and monitoring.

Provides deployment capabilities for agentforge-x agents including
containerization, HTTP serving, and health monitoring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""

    name: str
    version: str = "0.1.0"
    port: int = 8080
    host: str = "0.0.0.0"
    workers: int = 1
    timeout: float = 60.0
    health_check_path: str = "/health"
    metrics_path: str = "/metrics"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Health check response."""

    status: str  # healthy | unhealthy | degraded
    version: str
    uptime_seconds: float
    agents_count: int
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass
class MetricSample:
    """A single metric sample."""

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and exposes metrics."""

    def __init__(self):
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}

    def increment(self, name: str, value: float = 1.0, labels: Optional[dict[str, str]] = None) -> None:
        key = self._key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value

    def gauge(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        key = self._key(name, labels)
        self._gauges[key] = value

    def histogram(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        key = self._key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

    def get_counter(self, name: str, labels: Optional[dict[str, str]] = None) -> float:
        return self._counters.get(self._key(name, labels), 0.0)

    def get_gauge(self, name: str, labels: Optional[dict[str, str]] = None) -> float:
        return self._gauges.get(self._key(name, labels), 0.0)

    def get_histogram(self, name: str, labels: Optional[dict[str, str]] = None) -> list[float]:
        return self._histograms.get(self._key(name, labels), [])

    def _key(self, name: str, labels: Optional[dict[str, str]] = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        for key, value in self._counters.items():
            lines.append(f"{key} {value}")
        for key, value in self._gauges.items():
            lines.append(f"{key} {value}")
        return "\n".join(lines)


class DeploymentManager:
    """Manages agent deployment."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self._agents: dict[str, Any] = {}
        self._start_time = time.time()
        self._metrics = MetricsCollector()

    def register_agent(self, name: str, agent: Any) -> None:
        """Register an agent for deployment."""
        self._agents[name] = agent
        self.metrics.increment("agent_registered", labels={"name": name})

    def get_agent(self, name: str) -> Optional[Any]:
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

    def health_check(self) -> HealthStatus:
        """Run health checks."""
        checks = {
            "agents_loaded": len(self._agents) > 0,
            "uptime_ok": (time.time() - self._start_time) > 0,
        }
        status = "healthy" if all(checks.values()) else "degraded"
        return HealthStatus(
            status=status,
            version=self.config.version,
            uptime_seconds=time.time() - self._start_time,
            agents_count=len(self._agents),
            checks=checks,
        )

    def get_metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics


__all__ = [
    "DeploymentConfig",
    "HealthStatus",
    "MetricSample",
    "MetricsCollector",
    "DeploymentManager",
]
