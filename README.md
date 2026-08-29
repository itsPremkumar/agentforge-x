# agentforge-x

Advanced OSS framework combining DeepAgents planning with LangGraph orchestration.

## Core Kernel

`src/agentforge_x/kernel/` — state management, execution loop, checkpoints, event bus.

## Status

Building the core kernel — StateGraph runtime, AgentState, SQLite checkpoints,
planner→executor→critic→retry loop, subgraph spawning, JSONL event bus.
