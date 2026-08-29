"""JSONL event bus for the agentforge-x kernel.

All events are written as JSONL (one JSON object per line).
Each event: {ts, run_id, agent_id, type, payload}

Uses simulated timestamps (float seconds) for deterministic testing.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

from pydantic import BaseModel, Field


class BusEvent(BaseModel):
    """A single event in the JSONL event bus."""

    ts: float  # Simulated timestamp (seconds)
    run_id: str
    agent_id: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to a single JSONL line."""
        return json.dumps(self.model_dump(), default=str)

    @classmethod
    def from_json(cls, line: str) -> "BusEvent":
        """Parse a JSONL line into a BusEvent."""
        data = json.loads(line)
        return cls(**data)


class EventType:
    """Well-known event types emitted by the kernel (20 types)."""

    # Agent lifecycle
    AGENT_SPAWN = "agent_spawn"
    AGENT_SPAWN_END = "agent_spawn_end"
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_FAIL = "agent_fail"
    AGENT_HALT = "agent_halt"

    # Planner
    PLAN_CREATED = "plan_created"
    PLAN_STEP_START = "plan_step_start"
    PLAN_STEP_END = "plan_step_end"
    PLAN_STEP_FAIL = "plan_step_fail"
    PLAN_STEP_RETRY = "plan_step_retry"

    # Retry
    RETRY_SCHEDULED = "retry.scheduled"
    RETRY_EXECUTED = "retry.executed"
    RETRY_EXHAUSTED = "retry.exhausted"

    # Executor
    EXECUTOR_INVOKE = "executor_invoke"

    # Critic
    CRITIC_EVAL = "critic_eval"
    CRITIC_PASS = "critic_pass"
    CRITIC_FAIL = "critic_fail"

    # Subgraph
    SUBGRAPH_SPAWN = "subgraph_spawn"
    SUBGRAPH_COMPLETE = "subgraph_complete"

    # Budget
    BUDGET_CHECKPOINT = "budget_checkpoint"
    BUDGET_EXCEEDED = "budget_exceeded"

    # Artifact + message
    ARTIFACT_CREATE = "artifact_create"
    MESSAGE_EMIT = "message_emit"

    # Error
    ERROR = "error"


class EventBus:
    """JSONL event bus for kernel observability.

    Writes events to a JSONL file and/or in-memory buffer.
    Thread-safe via a lock. Uses simulated timestamps.
    """

    def __init__(
        self,
        run_id: str,
        agent_id: str = "primary",
        output_path: Optional[str] = None,
        buffer: bool = True,
        sim_clock: Optional[Any] = None,
    ):
        self.run_id = run_id
        self.agent_id = agent_id
        self.output_path = output_path
        self._buffer: list[BusEvent] = []
        self._buffer_enabled = buffer
        self._lock = threading.Lock()
        self._file_handle: Optional[Any] = None
        self._sim_clock = sim_clock

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            self._file_handle = open(output_path, "a", encoding="utf-8")

    def now(self) -> float:
        """Get current simulated time."""
        if self._sim_clock is not None:
            return self._sim_clock.now()
        import time
        return time.time()

    def emit(
        self,
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> BusEvent:
        """Emit an event to the bus."""
        event = BusEvent(
            ts=ts if ts is not None else self.now(),
            run_id=self.run_id,
            agent_id=agent_id or self.agent_id,
            type=event_type,
            payload=payload or {},
        )

        with self._lock:
            if self._buffer_enabled:
                self._buffer.append(event)
            if self._file_handle:
                self._file_handle.write(event.to_json() + "\n")
                self._file_handle.flush()

        return event

    def get_events(self, event_type: Optional[str] = None) -> list[BusEvent]:
        """Return buffered events, optionally filtered by type."""
        with self._lock:
            if event_type:
                return [e for e in self._buffer if e.type == event_type]
            return list(self._buffer)

    def clear(self) -> None:
        """Clear the in-memory buffer."""
        with self._lock:
            self._buffer.clear()

    def close(self) -> None:
        """Close the file handle if open."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self) -> "EventBus":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def read_jsonl(path: str) -> list[BusEvent]:
    """Read all events from a JSONL file."""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(BusEvent.from_json(line))
    return events
