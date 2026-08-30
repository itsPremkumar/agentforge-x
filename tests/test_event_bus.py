"""Tests for the JSONL event bus."""


from agentforge_x.kernel.event_bus import EventBus, EventType


class TestEventBus:
    """Test EventBus emission and format."""

    def test_event_emission_and_format(self):
        """Test basic event emission and JSON format."""
        bus = EventBus(run_id="run_1", agent_id="agent_1")
        event = bus.emit(EventType.AGENT_START, {"status": "planning"})

        assert event.run_id == "run_1"
        assert event.agent_id == "agent_1"
        assert event.type == "agent_start"
        assert event.payload == {"status": "planning"}
        assert isinstance(event.ts, float)

    def test_run_id_grouping(self):
        """Test that events are grouped by run_id."""
        bus = EventBus(run_id="run_1", agent_id="agent_1")
        bus.emit(EventType.PLAN_CREATED, {"steps": ["step_0"]})
        bus.emit(EventType.PLAN_STEP_START, {"step_id": "step_0"})

        events = bus.get_events()
        assert all(e.run_id == "run_1" for e in events)
        assert len(events) == 2

    def test_all_20_event_types(self):
        """Test that all 20 event types can be emitted without error."""
        bus = EventBus(run_id="run_1", agent_id="agent_1")

        all_types = [
            EventType.AGENT_SPAWN,
            EventType.AGENT_SPAWN_END,
            EventType.AGENT_START,
            EventType.AGENT_COMPLETE,
            EventType.AGENT_FAIL,
            EventType.AGENT_HALT,
            EventType.PLAN_CREATED,
            EventType.PLAN_STEP_START,
            EventType.PLAN_STEP_END,
            EventType.PLAN_STEP_FAIL,
            EventType.PLAN_STEP_RETRY,
            EventType.EXECUTOR_INVOKE,
            EventType.CRITIC_EVAL,
            EventType.CRITIC_PASS,
            EventType.CRITIC_FAIL,
            EventType.SUBGRAPH_SPAWN,
            EventType.SUBGRAPH_COMPLETE,
            EventType.BUDGET_CHECKPOINT,
            EventType.BUDGET_EXCEEDED,
            EventType.ARTIFACT_CREATE,
            EventType.MESSAGE_EMIT,
            EventType.ERROR,
        ]

        for event_type in all_types:
            bus.emit(event_type, {"test": True})

        events = bus.get_events()
        assert len(events) == len(all_types)

    def test_error_event_handling(self):
        """Test error event emission and retrieval."""
        bus = EventBus(run_id="run_1", agent_id="agent_1")
        bus.emit(
            EventType.ERROR,
            {"error_type": "timeout", "message": "LLM call timed out", "recoverable": True},
        )

        error_events = bus.get_events("error")
        assert len(error_events) == 1
        assert error_events[0].payload["error_type"] == "timeout"

    def test_concurrent_event_ordering(self):
        """Test that events maintain insertion order."""
        bus = EventBus(run_id="run_1", agent_id="agent_1")
        for i in range(100):
            bus.emit(EventType.PLAN_STEP_START, {"step_id": f"step_{i}"})

        events = bus.get_events()
        assert len(events) == 100
        for i, event in enumerate(events):
            assert event.payload["step_id"] == f"step_{i}"
