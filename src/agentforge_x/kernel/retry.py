"""Retry scheduler: manages retry attempts for failed steps."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agentforge_x.kernel.budget import BudgetEnforcer
from agentforge_x.kernel.event_bus import EventBus, EventType
from agentforge_x.kernel.state import PlanEntry


@dataclass
class RetryDecision:
    """The decision made by the retry scheduler."""

    should_retry: bool
    backoff_seconds: float
    reason: str


class RetryScheduler:
    """Decides whether and how to retry a failed step."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def compute_backoff(self, retry_count: int) -> float:
        delay = self.base_delay * (self.backoff_factor ** retry_count)
        delay = min(delay, self.max_delay)
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        return round(delay, 3)

    def decide(
        self,
        step: PlanEntry,
        budget: BudgetEnforcer,
        events: EventBus | None = None,
        agent_id: str = "primary",
    ) -> RetryDecision:
        if step.retry_count >= budget.budget.max_retries:
            if events:
                events.emit(
                    EventType.RETRY_EXHAUSTED,
                    {"step_id": step.id, "retry_count": step.retry_count, "error": step.error},
                    agent_id=agent_id,
                )
            return RetryDecision(should_retry=False, backoff_seconds=0.0, reason="Max retries exceeded")

        backoff = self.compute_backoff(step.retry_count)

        if events:
            events.emit(
                EventType.RETRY_SCHEDULED,
                {"step_id": step.id, "retry_count": step.retry_count, "backoff_seconds": backoff, "error": step.error},
                agent_id=agent_id,
            )

        return RetryDecision(
            should_retry=True,
            backoff_seconds=backoff,
            reason=f"Retry {step.retry_count + 1}",
        )

    async def wait_and_retry(
        self,
        step: PlanEntry,
        budget: BudgetEnforcer,
        events: EventBus | None = None,
        agent_id: str = "primary",
    ) -> bool:
        decision = self.decide(step, budget, events, agent_id)
        if not decision.should_retry:
            return False

        if events:
            events.emit(
                EventType.RETRY_EXECUTED,
                {"step_id": step.id, "backoff_seconds": decision.backoff_seconds, "retry_number": step.retry_count + 1},
                agent_id=agent_id,
            )

        await asyncio.sleep(decision.backoff_seconds)
        step.retry_count += 1
        return True

    def record_retry(self, step: PlanEntry) -> int:
        step.retry_count += 1
        return step.retry_count
