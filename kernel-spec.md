# agentforge-x Kernel — Architectural Specification

## Overview

The agentforge-x kernel is a **StateGraph runtime** over a structured
`AgentState` with a **planner→executor→critic→retry** control loop,
SQLite checkpoint persistence, subgraph spawning with budget caps,
and a JSONL event bus.

This spec defines:
1. AgentState schema
2. JSONL event bus message types
3. Budget cap model
4. Planner/executor/critic interfaces

---

## 1. AgentState Schema

`AgentState` is a TypedDict that serves as the single source of truth
for an agent's runtime state. It is checkpointed to SQLite at every
state transition.

```typescript
interface AgentState {
  // Identity & routing
  agent_id: string;           // UUID v7, unique per agent
  run_id: string;             // UUID v7, groups all agents in one simulation run
  parent_agent_id?: string;   // Set when spawned as a subgraph

  // Core working memory
  messages: AIMessage[];      // Conversation/messages history
  plan: PlanEntry[];          // Current plan (ordered steps)
  scratchpad: string;         // Free-form working memory / scratch space
  artifacts: ArtifactRef[];   // References to produced artifacts

  // Fleet-level events (audit trail)
  fleet_events: FleetEvent[]; // Events this agent emitted or received

  // Lifecycle
  status: AgentStatus;        // idle | planning | executing | critiquing | retrying | completed | failed | halted
  error?: string;             // Error message on failure

  // Budget tracking
  budget: BudgetState;

  // Metadata
  created_at: number;         // Simulated timestamp (seconds)
  updated_at: number;         // Simulated timestamp (seconds)
  completed_at?: number;      // When status became completed/failed/halted
}

type AgentStatus =
  | 'idle'
  | 'planning'
  | 'executing'
  | 'critiquing'
  | 'retrying'
  | 'completed'
  | 'failed'
  | 'halted';

interface AIMessage {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  ts?: number;                // Simulated timestamp
}

interface PlanEntry {
  id: string;                 // Plan step ID (e.g., "step_0")
  instruction: string;        // What to do
  tool: string;               // Tool to use
  args: Record<string, unknown>; // Tool arguments
  depends_on?: string[];      // IDs of prerequisite steps
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  result?: string;            // Output of completed step
  retry_count?: number;       // How many times this step was retried
}

interface ArtifactRef {
  id: string;
  type: string;               // e.g., "code", "data", "report"
  path: string;               // Location reference
  metadata?: Record<string, unknown>;
}

interface BudgetState {
  max_runtime: number;        // Max simulated seconds for this agent
  elapsed: number;            // Elapsed simulated seconds
  max_llm_calls: number;      // Max LLM calls in this agent's lifetime
  llm_calls: number;          // LLM calls made so far
  max_tokens: number;         // Max total tokens (in + out)
  tokens_used: number;        // Tokens consumed so far
  halted: boolean;            // True if any budget was exceeded
  halt_reason?: 'runtime' | 'llm_calls' | 'tokens' | 'subgraph_budget';
}
```

---

## 2. JSONL Event Bus

All events are written as JSONL (one JSON object per line) to
`events.jsonl` in the run directory.

### Event Schema

```typescript
interface BusEvent {
  ts: number;           // Simulated timestamp (seconds)
  run_id: string;       // UUID v7 — groups events across agents
  agent_id: string;     // UUID v7 — who emitted the event
  type: EventType;
  payload: Record<string, unknown>;
}

type EventType =
  | 'agent_spawn'
  | 'agent_spawn_end'
  | 'agent_start'
  | 'agent_complete'
  | 'agent_fail'
  | 'agent_halt'
  | 'plan_created'
  | 'plan_step_start'
  | 'plan_step_end'
  | 'plan_step_fail'
  | 'plan_step_retry'
  | 'executor_invoke'
  | 'critic_eval'
  | 'critic_pass'
  | 'critic_fail'
  | 'subgraph_spawn'
  | 'subgraph_complete'
  | 'budget_checkpoint'
  | 'budget_exceeded'
  | 'artifact_create'
  | 'message_emit'
  | 'error';
```

### Key Event Payloads

**agent_spawn**
```json
{
  "parent_agent_id": "uuid",
  "agent_type": "executor|critic|subgraph",
  "initial_state": { ...diff from parent... }
}
```

**agent_start**
```json
{ "status": "planning|executing|critiquing|retrying" }
```

**agent_complete**
```json
{ "status": "completed", "final_plan_status": "completed|failed" }
```

**agent_fail**
```json
{ "error": "string", "stack_trace": "string?" }
```

**agent_halt**
```json
{ "reason": "runtime|llm_calls|tokens|subgraph_budget", "budget": { ... } }
```

**plan_created**
```json
{ "steps": ["step_0", "step_1", ...], "total_steps": 42 }
```

**plan_step_start**
```json
{ "step_id": "step_0", "instruction": "...", "tool": "bash", "args": {...} }
```

**plan_step_end**
```json
{ "step_id": "step_0", "success": true, "result": "...", "duration_ms": 123 }
```

**plan_step_fail**
```json
{ "step_id": "step_0", "error": "string", "retry_count": 1 }
```

**executor_invoke**
```json
{ "step_id": "step_0", "tool": "bash", "args": {...}, "result_type": "string|file" }
```

**critic_eval**
```json
{ "step_id": "step_0", "criteria": ["correctness", "safety"], "threshold": 0.8 }
```

**critic_pass**
```json
{ "step_id": "step_0", "score": 0.92, "feedback": "..." }
```

**critic_fail**
```json
{ "step_id": "step_0", "score": 0.41, "feedback": "...", "reason": "below_threshold" }
```

**subgraph_spawn**
```json
{ "subgraph_id": "uuid", "entry_point": "string", "budget_allocation": {...} }
```

**subgraph_complete**
```json
{ "subgraph_id": "uuid", "status": "completed|failed|halted", "exit_state": {...} }
```

**budget_checkpoint**
```json
{ "elapsed": 42, "llm_calls": 17, "tokens_used": 12500, "remaining_runtime_pct": 0.78 }
```

**budget_exceeded**
```json
{ "field": "runtime|llm_calls|tokens", "limit": 600, "actual": 601 }
```

**artifact_create**
```json
{ "artifact_id": "string", "type": "code|data|report", "path": "string" }
```

**message_emit**
```json
{ "role": "assistant", "content": "...", "token_count": 42 }
```

**error**
```json
{ "error_type": "string", "message": "string", "recoverable": true, "retry_after_ms": 1000 }
```

---

## 3. Budget Cap Model

Budgets are hierarchical — parent agents allocate budgets to subgraphs,
and each agent enforces its own caps.

### 3.1 BudgetState (per-agent)

Already defined in the AgentState schema above. Each agent carries
its own `BudgetState` and checks it before every LLM call and at
every step boundary.

### 3.2 Budget Allocation for Subgraphs

When a parent spawns a subgraph, it must allocate budget:

```typescript
interface BudgetAllocation {
  max_runtime: number;     // Seconds — MUST be <= remaining parent runtime
  max_llm_calls: number;   // MUST be <= remaining parent llm_calls
  max_tokens: number;      // MUST be <= remaining parent tokens
  hard_cap: boolean;       // If true, subgraph failure halts parent; if false, it retries
}
```

### 3.3 Budget Enforcement

- Checked before every LLM call (`llm_calls` and `tokens`)
- Checked at every plan step boundary (`elapsed` runtime)
- If `hard_cap` and budget exceeded: agent → `halted`, parent receives
  `subgraph_complete` with `status: "halted"`
- If NOT `hard_cap` and budget exceeded: agent → retry with reduced scope

### 3.4 Budget Overhead

Subgraph spawning incurs a fixed overhead:
- `runtime_overhead = 5s` per spawn (checkpoint + initialization)
- `llm_overhead = 0` (no LLM calls for spawn itself)
- `token_overhead = 0` (metadata only)

This overhead is deducted from the parent's remaining budget BEFORE
allocating to the child.

---

## 4. Control Loop Interfaces

### 4.1 Planner Interface

```typescript
interface Planner {
  // Create a plan from the agent's current state + instructions
  createPlan(state: AgentState, instructions: string): Promise<PlanEntry[]>;

  // Re-plan when the current plan has failed or needs adjustment
  replan(state: AgentState, failedStepId?: string): Promise<PlanEntry[]>;

  // Default plan creation from a high-level goal
  decompose(goal: string, context: AgentState['messages']): Promise<PlanEntry[]>;

  // Estimate resource requirements for a plan
  estimate(plan: PlanEntry[]): {
    estimated_runtime: number;
    estimated_llm_calls: number;
    estimated_tokens: number;
  };
}

// Default implementation: LLMPlanner
interface LLMPlannerConfig {
  model: string;           // e.g., "poolside/laguna-s-2.1:free"
  max_steps: number;       // Hard cap on plan steps
  step_max_tokens: number; // Max tokens per step's tool output
}
```

### 4.2 Executor Interface

```typescript
interface Executor {
  // Execute a single plan step
  execute(state: AgentState, step: PlanEntry): Promise<ExecutionResult>;

  // Check if a tool is available for this step
  canExecute(step: PlanEntry): boolean;
}

interface ExecutionResult {
  success: boolean;
  output: string;            // Tool output (truncated to step_max_tokens)
  duration_ms: number;       // Simulated duration
  artifacts?: ArtifactRef[]; // Artifacts produced
  error?: string;            // On failure
  retryable: boolean;        // Whether the step can be retried
}
```

### 4.3 Critic Interface

```typescript
interface Critic {
  // Evaluate an execution result against criteria
  evaluate(state: AgentState, step: PlanEntry, result: ExecutionResult): Promise<Critique>;

  // Evaluate the entire plan's progress
  evaluateProgress(state: AgentState): Promise<Critique>;

  // Decide whether to retry a failed step
  shouldRetry(state: AgentState, step: PlanEntry, failure: string): boolean;
}

interface Critique {
  passed: boolean;
  score: number;             // 0.0 - 1.0
  feedback: string;          // Explanation
  suggested_fix?: string;    // How to fix the failure
  criteria: { name: string; score: number; weight: number }[];
}
```

### 4.4 StateGraph Runtime

The StateGraph orchestrates the control loop:

```
[agent_start] → [planner.createPlan] → [plan_created]
                                          ↓
         ┌────────────────────── [plan_step_start]
         │                           ↓
         │                   [executor.execute]
         │                           ↓
         │                   [plan_step_end|plan_step_fail]
         │                           ↓
         │                   [critic.evaluate] → [critic_pass|critic_fail]
         │                           ↓
         │              ┌──────────── retry? ────────────┐
         │              │                                │
         │           [plan_step_retry]               [next step]
         │              │                                │
         │              └────────────────────────────────┘
         │                            ↓
         │                     [agent_complete]
         │                            ↓
         └────────────────────→ [subgraph_complete]
```

### 4.5 Subgraph Spawning

Subgraphs are spawned when a plan step requires a nested agent:
- Parent creates a `BudgetAllocation` from its remaining budget
- Parent emits `subgraph_spawn` event
- Child agent inherits partial state (scratchpad, relevant artifacts)
- Child runs its own planner→executor→critic→retry loop
- When child completes, parent receives `subgraph_complete` and
  continues from its next plan step

---

## 5. SQLite Checkpoint Schema

```sql
CREATE TABLE checkpoints (
    run_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    state_json TEXT NOT NULL,  -- Full AgentState serialized
    PRIMARY KEY (run_id, agent_id, seq)
);

CREATE INDEX idx_checkpoint_run ON checkpoints(run_id);
CREATE INDEX idx_checkpoint_agent ON checkpoints(run_id, agent_id);
```

Checkpoints are written:
1. After agent_spawn (initial state)
2. After each state transition (planning→executing, etc.)
3. After each plan step completes/fails
4. On agent completion/failure/halt

---

## 6. Key Design Decisions

1. **Deterministic simulation**: The kernel uses a simulated clock.
   All timestamps in AgentState and BusEvent are simulated seconds,
   not wall-clock. This enables reproducible scenarios.

2. **Budget caps are immutable per-agent**: Once allocated, an agent
   cannot exceed its budget. Subgraph budget allocation is a deduction
   from the parent — the parent's budget is reduced at spawn time.

3. **Critic-driven retry**: The critic decides whether a step's failure
   is worth retrying (based on retryable flag + critique score).
   Retry count is capped at 3 per step by default.

4. **Event sourcing**: All state transitions are recorded as JSONL
   events. The SQLite checkpoint is a materialization of the current
   AgentState; the events are the source of truth for auditing/replay.

5. **Swarm-sim experience**: The failure injection patterns from
   swarm-sim (crash, stall, timeout, claim-loss) are mapped to
   kernel error types:
   - crash → recoverable=false, agent fails
   - stall → recoverable=true, retry after 5s
   - timeout → recoverable=true if under llm_calls budget
   - claim-loss → recoverable=true, replan that step

---

## 7. Test Plan (≥40 tests)

### 7.1 AgentState (4 tests)
- State creation with all fields
- State transition validation (idle→planning→executing→...)
- BudgetState enforcement
- Serialization/deserialization round-trip

### 7.2 Event Bus (8 tests)
- Event emission and format
- Run ID grouping
- All 20+ event types
- Error event handling
- Concurrent event ordering

### 7.3 Budget Caps (6 tests)
- Runtime cap enforcement
- LLM call cap enforcement
- Token cap enforcement
- Subgraph budget allocation from parent
- Budget overhead deduction
- Hard cap vs soft cap behavior

### 7.4 Planner (5 tests)
- Plan creation from instructions
- Plan decomposition (goal → steps)
- Replan after step failure
- Resource estimation accuracy
- Max steps enforcement

### 7.5 Executor (5 tests)
- Step execution success
- Step execution failure (non-retryable)
- Step execution failure (retryable)
- Artifact creation on execution
- Tool availability checking

### 7.6 Critic (5 tests)
- Crit pass when score ≥ threshold
- Crit fail when score < threshold
- Retry decision logic
- Progress evaluation
- Feedback quality

### 7.7 Control Loop (6 tests)
- Full planner→executor→critic→complete cycle
- Retry on crit fail (up to max retries)
- Halt on budget exceeded (hard cap)
- Subgraph spawn and completion
- StateGraph determinism across runs
- Error propagation to parent on subgraph halt

### 7.8 Scale Scenarios (6 tests)
- 10 concurrent agents with individual budgets
- 5-level deep subgraph nesting with cascading budget caps
- 100 plan steps in a single agent (max_steps=100)
- Budget exhaustion under load (10 agents, tight budgets)
- Event bus throughput (10K events)
- Checkpoint recovery after simulated crash

---

## 8. File Layout

```
src/agentforge_x/
  __init__.py
  kernel/
    __init__.py
    state.py              # AgentState TypedDict + serialization
    event_bus.py          # JSONL event bus
    budget.py             # BudgetState + allocation models
    planner.py            # Planner interface + LLMPlanner impl
    executor.py           # Executor interface + ToolExecutor impl
    critic.py             # Critic interface + LLMCritic impl
    state_graph.py        # StateGraph runtime / control loop
    checkpoint.py         # SQLite checkpoint manager
    subgraph.py           # Subgraph spawning/mgmt
    config.py             # KernelConfig TypedDict
  tests/
    test_state.py
    test_event_bus.py
    test_budget.py
    test_planner.py
    test_executor.py
    test_critic.py
    test_state_graph.py
    test_subgraph.py
    test_scale.py
```
