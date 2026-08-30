# Phase 8 Evolution Engine — Architectural Specification

## Overview

The Evolution Engine is a self-improvement layer that runs ON TOP of
the agentforge-x kernel (t_8040e0e7). It uses **GEPA (Generalized
Evolutionary Prompt Architecture)** to evolve agent prompts, strategies,
and tool configurations over multiple generations.

Core capabilities:
1. **GEPA prompt evolution** — mutate, crossover, and select prompts
2. **Benchmark framework** — evaluate evolved agents on standardized tasks
3. **Strategy search** — explore the space of planner/executor/critic configs
4. **Evolution loop** — iterate generations with safety gates and rollback

---

## 1. GEPA Prompt Evolution Interface

### 1.1 Prompt Representation

```typescript
interface PromptGenome {
  id: string;                   // UUID v7
  generation: number;           // Which generation this genome belongs to
  parent_ids: string[];         // Parent genome IDs (for lineage)
  fitness: number | null;       // Score from benchmark eval (null if unevaluated)

  // The evolvable components
  system_prompt: string;        // Base system prompt
  tool_descriptions: ToolPrompt[]; // Per-tool prompt enhancements
  plan_strategy: PlanStrategy;  // Planner behavior parameters
  critic_rubric: CriticRubric;  // Critic evaluation criteria
  retry_policy: RetryPolicy;    // When/how to retry failed steps

  // Metadata
  created_at: number;
  evaluated_at?: number;
  benchmark_results?: BenchmarkResult;
  mutation_log: MutationRecord[];
}

interface ToolPrompt {
  tool_name: string;
  description: string;          // Enhanced description
  examples: string[];           // Few-shot examples
  error_recovery: string;       // What to do on failure
}

interface PlanStrategy {
  decomposition_style: 'broad_first' | 'narrow_first' | 'dependency_first';
  max_depth: number;            // Max plan nesting depth
  parallelization: number;      // How many steps can run in parallel
  lookahead: number;            // How many steps to look ahead
}

interface CriticRubric {
  criteria: CriticCriterion[];
  aggregation: 'weighted_mean' | 'min' | 'majority';
  pass_threshold: number;       // 0.0 - 1.0
}

interface CriticCriterion {
  name: string;
  weight: number;
  evaluator: 'llm' | 'rule' | 'hybrid';
  prompt: string;               // For LLM evaluator
}

interface RetryPolicy {
  max_retries: number;
  backoff: 'linear' | 'exponential' | 'constant';
  initial_delay: number;        // Seconds
  max_delay: number;            // Seconds
  escalation: 'replan' | 'subgraph' | 'fail';
}

interface MutationRecord {
  type: MutationType;
  target: string;               // Which component was mutated
  description: string;
  timestamp: number;
}

type MutationType =
  | 'system_prompt_rewrite'
  | 'tool_description_enhance'
  | 'plan_strategy_adjust'
  | 'critic_rubric_tighten'
  | 'critic_rubric_loosen'
  | 'retry_policy_change'
  | 'crossover_merge';
```

### 1.2 Evolutionary Operators

```typescript
interface EvolutionOperators {
  // Mutation operators
  mutate(genome: PromptGenome, rng: () => number): PromptGenome;
  mutateSystemPrompt(genome: PromptGenome, instruction: string): PromptGenome;
  mutateToolPrompt(genome: PromptGenome, toolName: string): PromptGenome;
  mutateCriticRubric(genome: PromptGenome): PromptGenome;

  // Crossover operators
  crossover(parent1: PromptGenome, parent2: PromptGenome, rng: () => number): PromptGenome;
  uniformCrossover(parent1: PromptGenome, parent2: PromptGenome): PromptGenome;
  singlePointCrossover(parent1: PromptGenome, parent2: PromptGenome): PromptGenome;

  // Selection operators
  selectParent(pool: PromptGenome[], rng: () => number): PromptGenome;
  tournamentSelect(pool: PromptGenome[], tournamentSize: number, rng: () => number): PromptGenome;
  rouletteSelect(pool: PromptGenome[], rng: () => number): PromptGenome;

  // Diversity measurement
  diversity(pool: PromptGenome[]): number; // 0.0 = identical, 1.0 = maximally diverse
  similarity(a: PromptGenome, b: PromptGenome): number;
}
```

### 1.3 Mutation Strategies

```typescript
type MutationStrategy =
  | 'random_rewrite'           // Replace a random component
  | 'guided_rewrite'           // Use LLM to suggest improvements based on failures
  | 'targeted_rewrite'         // Mutate only the component that caused failure
  | 'minimal_change'           // Smallest possible change that alters behavior
  | 'maximal_exploration'      // Large random change to explore new areas
  | 'lineage_informed';        // Use mutation history to avoid repeating failures

interface MutationConfig {
  strategy: MutationStrategy;
  rate: number;               // 0.0 - 1.0, probability of mutating each component
  max_mutations_per_genome: number; // How many components to mutate at once
  guidance_prompt: string;    // For guided_rewrite strategy
  preserve_fitness: boolean;  // Elitism: keep top-N genomes unchanged
  elitism_count: number;      // How many top genomes to preserve
}
```

---

## 2. Benchmark Framework

### 2.1 Benchmark Definition

```typescript
interface Benchmark {
  id: string;
  name: string;
  version: string;
  description: string;
  tasks: BenchmarkTask[];
  timeout: number;             // Per-task timeout (simulated seconds)
  metrics: BenchmarkMetric[];
  environment: BenchmarkEnvironment;
}

interface BenchmarkTask {
  id: string;
  name: string;
  description: string;
  initial_state: Partial<AgentState>;  // What state the agent starts in
  expected_output: ExpectedOutput;     // What success looks like
  difficulty: 'easy' | 'medium' | 'hard' | 'expert';
  tags: string[];
  max_runtime: number;         // Task-level runtime cap
}

interface ExpectedOutput {
  type: 'exact_match' | 'contains' | 'regex' | 'llm_judge' | 'composite';
  value: string | string[] | RegExp | LLMJudgeConfig | ExpectedOutput[];
  threshold?: number;          // For composite: min passing fraction
}

interface LLMJudgeConfig {
  model: string;
  rubric: string;
  scale: { min: number; max: number; passing: number };
}

interface BenchmarkMetric {
  name: string;
  type: 'accuracy' | 'latency' | 'cost' | 'robustness' | 'efficiency';
  aggregation: 'mean' | 'median' | 'min' | 'max' | 'percentile_95';
  weight: number;              // For composite scoring
}

interface BenchmarkEnvironment {
  tools_available: string[];
  max_llm_calls: number;
  max_tokens: number;
  failure_rate: number;        // 0.0 - 1.0, simulates flaky environment
  noise_level: number;         // 0.0 - 1.0, how much noise in task descriptions
}
```

### 2.2 Evaluator

```typescript
interface Evaluator {
  // Run a single genome against a benchmark
  evaluate(genome: PromptGenome, benchmark: Benchmark): Promise<BenchmarkResult>;

  // Run a batch of genomes in parallel
  evaluateBatch(
    genomes: PromptGenome[],
    benchmark: Benchmark,
    max_parallel: number,
  ): Promise<BenchmarkResult[]>;

  // Compare two genomes head-to-head
  compare(
    genome_a: PromptGenome,
    genome_b: PromptGenome,
    benchmark: Benchmark,
  ): Promise<ComparisonResult>;
}

interface BenchmarkResult {
  genome_id: string;
  benchmark_id: string;
  task_results: TaskResult[];
  composite_score: number;      // Weighted average of task scores
  metrics: Record<string, number>;
  total_runtime: number;
  total_llm_calls: number;
  total_tokens: number;
  passed: boolean;              // Did it meet the pass threshold?
}

interface TaskResult {
  task_id: string;
  passed: boolean;
  score: number;                // 0.0 - 1.0
  output: string;
  expected: string;
  runtime: number;
  llm_calls: number;
  tokens: number;
  error?: string;
}

interface ComparisonResult {
  genome_a_id: string;
  genome_b_id: string;
  winner: 'a' | 'b' | 'tie';
  task_wins: { a: number; b: number; tie: number };
  score_diff: number;
  statistically_significant: boolean;  // p < 0.05
}
```

### 2.3 Built-in Benchmarks

```typescript
// Predefined benchmarks for common agent tasks
interface BuiltinBenchmarks {
  // Task planning & execution
  'plan-simple': Benchmark;        // 10 tasks, single-step plans
  'plan-complex': Benchmark;       // 50 tasks, multi-step plans with dependencies
  'plan-adversarial': Benchmark;   // Tasks designed to trick the planner

  // Error recovery
  'error-recovery': Benchmark;     // Tasks that always fail initially
  'flaky-tools': Benchmark;        // 30% tool failure rate

  // Budget efficiency
  'budget-stress': Benchmark;      // Tight budgets, measure completion rate
  'token-optimization': Benchmark; // Minimize tokens while maintaining accuracy

  // Robustness
  'ambiguous-tasks': Benchmark;    // Vague instructions
  'contradictory-goals': Benchmark;// Conflicting objectives

  // Reproduction of real incidents
  'starvation-repro': Benchmark;   // Reproduce 600s starvation scenario
  'pid-reaping': Benchmark;        // Reproduce pid-reaping patterns
}
```

---

## 3. Strategy Search

### 3.1 Search Space

```typescript
interface SearchSpace {
  // Dimensions of the search
  dimensions: SearchDimension[];

  // Constraints
  constraints: SearchConstraint[];

  // Prior knowledge (from previous runs)
  priors: SearchPrior[];
}

interface SearchDimension {
  name: string;
  type: 'continuous' | 'discrete' | 'categorical' | 'text';
  range?: { min: number; max: number };  // For continuous
  values?: string[];                      // For categorical/discrete
  length_range?: { min: number; max: number }; // For text (token count)
}

interface SearchConstraint {
  type: 'budget' | 'safety' | 'dependency' | 'exclusion';
  expression: string;  // e.g., "llm_calls < 100", "tool != 'bash'"
  penalty: number;     // How much to penalize violations
}

interface SearchPrior {
  dimension: string;
  distribution: 'normal' | 'uniform' | 'log_normal';
  mean?: number;
  std?: number;
  source: string;  // Where this prior came from
}
```

### 3.2 Search Algorithms

```typescript
type SearchAlgorithm =
  | 'genetic'           // Standard genetic algorithm
  | 'nsga2'             // Multi-objective: NSGA-II
  | 'cma_es'            // Covariance Matrix Adaptation
  | 'bayesian'          // Bayesian optimization
  | 'random'            // Random search (baseline)
  | 'grid'              // Grid search
  | 'mcts';             // Monte Carlo Tree Search

interface SearchConfig {
  algorithm: SearchAlgorithm;
  population_size: number;
  generations: number;
  elite_ratio: number;         // Top fraction to preserve
  crossover_rate: number;      // 0.0 - 1.0
  mutation_rate: number;       // 0.0 - 1.0
  early_stopping: {
    enabled: boolean;
    patience: number;          // Generations without improvement
    min_improvement: number;   // Min fitness gain to reset patience
  };
  multi_objective: {
    enabled: boolean;
    objectives: string[];      // e.g., ['accuracy', 'latency', 'cost']
  };
}

interface SearchProgress {
  generation: number;
  best_fitness: number;
  mean_fitness: number;
  diversity: number;
  genomes_evaluated: number;
  elapsed_time: number;
  estimated_remaining: number;
  pareto_front?: PromptGenome[];  // For multi-objective search
}
```

### 3.3 Strategy Encoding

The search space includes:

| Dimension | Type | Range | Description |
|-----------|------|-------|-------------|
| `decomposition_style` | categorical | broad_first, narrow_first, dependency_first | How plans are structured |
| `max_depth` | discrete | 1-10 | Plan nesting depth |
| `parallelization` | discrete | 1-8 | Concurrent step execution |
| `critic_threshold` | continuous | 0.5-0.99 | Critic pass threshold |
| `max_retries` | discrete | 0-5 | Retry attempts per step |
| `retry_backoff` | categorical | linear, exponential, constant | Backoff strategy |
| `system_prompt_length` | discrete | 50-2000 | System prompt token count |
| `tool_prompt_examples` | discrete | 0-5 | Few-shot examples per tool |
| `plan_lookahead` | discrete | 1-10 | Steps to look ahead |
| `mutation_rate` | continuous | 0.01-0.5 | Self-adaptive mutation rate |

---

## 4. Evolution Loop

### 4.1 Loop Structure

```typescript
interface EvolutionLoop {
  // Configuration
  config: EvolutionConfig;

  // State
  generation: number;
  population: PromptGenome[];
  best_genome: PromptGenome | null;
  history: GenerationHistory[];

  // Core operations
  initialize(): Promise<void>;
  step(): Promise<GenerationResult>;
  run(): Promise<EvolutionResult>;
  stop(): void;

  // Safety
  safety_gate: SafetyGate;
  rollback_manager: RollbackManager;
}

interface EvolutionConfig {
  // Population
  population_size: number;
  initial_population: PromptGenome[];  // Seed genomes

  // Search
  search: SearchConfig;
  mutation: MutationConfig;

  // Evaluation
  benchmark: Benchmark;
  evaluator: Evaluator;
  eval_batch_size: number;
  max_parallel_evals: number;

  // Safety
  safety: SafetyConfig;
  max_generations: number;
  max_total_runtime: number;   // Total simulated time budget
  max_total_llm_calls: number; // Total LLM call budget

  // Persistence
  checkpoint_interval: number;  // Save state every N generations
  checkpoint_path: string;
}

interface GenerationResult {
  generation: number;
  population: PromptGenome[];
  best: PromptGenome;
  mean_fitness: number;
  diversity: number;
  elapsed_time: number;
  safety_violations: SafetyViolation[];
  action: 'continue' | 'rollback' | 'stop' | 'pause';
}

interface EvolutionResult {
  success: boolean;
  best_genome: PromptGenome;
  total_generations: number;
  total_evaluations: number;
  total_runtime: number;
  total_llm_calls: number;
  fitness_curve: number[];
  final_diversity: number;
  safety_violations: SafetyViolation[];
  checkpoints: string[];
  rollback_count: number;
}
```

### 4.2 The Step Loop (per generation)

```
1. EVALUATE: Score all unevaluated genomes on benchmark
2. SELECT: Choose parents using tournament/roulette selection
3. REPRODUCE: Create offspring via crossover + mutation
4. SAFETY CHECK: Run safety gates on offspring
5. POPULATION UPDATE: Replace worst with best offspring (elitism)
6. HISTORY LOG: Record generation stats
7. CHECKPOINT: Save state every N generations
8. CONVERGENCE CHECK: Stop if patience exhausted or budget exceeded
```

### 4.3 Safety Gates

Safety gates prevent evolved genomes from producing harmful or
unintended behavior:

```typescript
interface SafetyGate {
  gates: SafetyRule[];
  evaluate(genome: PromptGenome, context: SafetyContext): SafetyViolation[];
}

interface SafetyRule {
  id: string;
  name: string;
  description: string;
  severity: 'warning' | 'error' | 'critical';
  category: SafetyCategory;
  evaluator: (genome: PromptGenome, context: SafetyContext) => SafetyCheckResult;
}

type SafetyCategory =
  | 'prompt_injection'      // Does the prompt contain injection attempts?
  | 'tool_misuse'           // Does it encourage dangerous tool use?
  | 'data_exfiltration'     // Does it leak sensitive information?
  | 'resource_exhaustion'   // Does it waste resources?
  | 'goal_divergence'       // Does it drift from the original goal?
  | 'output_quality'        // Does it produce low-quality output?
  | 'critic_gaming'         // Does it game the critic rather than solve?
  | 'stability';            // Does it produce inconsistent results?

interface SafetyCheckResult {
  passed: boolean;
  score: number;            // 0.0 - 1.0 (1.0 = safest)
  details: string;
  suggestion?: string;
}

interface SafetyViolation {
  rule_id: string;
  genome_id: string;
  severity: 'warning' | 'error' | 'critical';
  message: string;
  genome_snapshot: PromptGenome;
  action_taken: 'logged' | 'repaired' | 'rejected' | 'rolled_back';
}

interface SafetyConfig {
  enabled_gates: string[];
  max_violations_per_generation: number;
  auto_repair: boolean;
  repair_prompt: string;
  reject_on_critical: boolean;
  rollback_on_regression: boolean;
  regression_threshold: number;  // Fitness drop that triggers rollback
}
```

### 4.4 Rollback Manager

```typescript
interface RollbackManager {
  checkpoints: Checkpoint[];

  save(generation: number, population: PromptGenome[], metadata: Record<string, unknown>): Checkpoint;
  rollback(target_generation: number): PromptGenome[];
  rollback_to_last_safe(): PromptGenome[];

  // Smart rollback: find the last generation without safety violations
  find_safe_generation(): number;

  // Progressive rollback: gradually revert changes
  partial_rollback(target: PromptGenome, reference: PromptGenome): PromptGenome;
}

interface Checkpoint {
  id: string;
  generation: number;
  timestamp: number;
  population: PromptGenome[];
  best_fitness: number;
  metadata: Record<string, unknown>;
  safety_status: 'clean' | 'warning' | 'violation';
}
```

---

## 5. Integration with Kernel

### 5.1 How the Evolution Engine uses the kernel

```
┌─────────────────────────────────────────────────────────┐
│                  Evolution Engine                        │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Search     │  │  Evolution   │  │   Safety      │  │
│  │   Strategy   │──│  Loop        │──│   Gates       │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────────┘  │
│         │                │                               │
│  ┌──────▼────────────────▼──────────────────────────┐   │
│  │              Genome Evaluator                     │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │           agentforge-x Kernel (t_8040e0e7)        │   │
│  │  ┌────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │Planner │  │Executor  │  │Critic + Retry    │  │   │
│  │  └────────┘  └──────────┘  └──────────────────┘  │   │
│  │  ┌──────────────────────────────────────────┐    │   │
│  │  │  StateGraph + Budget + Event Bus          │    │   │
│  │  └──────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Kernel-Evolution Contracts

The evolution engine reads/writes the kernel via these interfaces:

```typescript
interface KernelAdapter {
  // Create an agent with a given genome
  createAgent(genome: PromptGenome, task: BenchmarkTask): Promise<AgentState>;

  // Run the agent and get results
  runAgent(agent_id: string): Promise<AgentState>;

  // Get the agent's event trace
  getEventTrace(agent_id: string): Promise<BusEvent[]>;

  // Get the final benchmark result
  getResult(agent_id: string): Promise<BenchmarkResult>;

  // Extract metrics from the agent's final state
  extractMetrics(state: AgentState): Record<string, number>;

  // Apply a genome to the kernel's config
  applyGenome(genome: PromptGenome): void;

  // Reset the kernel for a fresh evaluation
  reset(): void;
}
```

---

## 6. Persistence and Observability

### 6.1 Evolution Database

```sql
CREATE TABLE genomes (
    id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL,
    fitness REAL,
    genome_json TEXT NOT NULL,
    parent_ids TEXT,
    created_at REAL NOT NULL,
    evaluated_at REAL,
    benchmark_id TEXT,
    composite_score REAL
);

CREATE TABLE evaluations (
    id TEXT PRIMARY KEY,
    genome_id TEXT NOT NULL,
    benchmark_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    score REAL NOT NULL,
    runtime REAL NOT NULL,
    llm_calls INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    output TEXT,
    error TEXT,
    FOREIGN KEY (genome_id) REFERENCES genomes(id)
);

CREATE TABLE checkpoints (
    id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    population_ids TEXT NOT NULL,
    best_fitness REAL NOT NULL,
    metadata_json TEXT,
    safety_status TEXT NOT NULL
);

CREATE TABLE safety_violations (
    id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL,
    genome_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    timestamp REAL NOT NULL
);

CREATE INDEX idx_genomes_generation ON genomes(generation);
CREATE INDEX idx_evaluations_genome ON evaluations(genome_id);
CREATE INDEX idx_safety_generation ON safety_violations(generation);
```

### 6.2 Evolution Events

The evolution engine emits its own JSONL events alongside the
kernel's events:

```typescript
type EvolutionEventType =
  | 'evolution_start'
  | 'evolution_complete'
  | 'generation_start'
  | 'generation_complete'
  | 'genome_evaluated'
  | 'genome_selected'
  | 'genome_mutated'
  | 'genome_crossover'
  | 'safety_violation'
  | 'rollback_triggered'
  | 'rollback_complete'
  | 'checkpoint_saved'
  | 'early_stopping'
  | 'diversity_collapse'
  | 'pareto_update';
```

---

## 7. Test Plan (≥50 tests)

### 7.1 GEPA Genome (6 tests)
- Genome creation with all components
- Genome serialization/deserialization
- Mutation record tracking
- Lineage tracking (parent_ids)
- Genome similarity measurement
- Diversity calculation

### 7.2 Evolutionary Operators (8 tests)
- Single-point crossover
- Uniform crossover
- Random mutation
- Guided mutation
- Targeted mutation
- Tournament selection
- Roulette selection
- Elitism preservation

### 7.3 Safety Gates (8 tests)
- Prompt injection detection
- Tool misuse detection
- Resource exhaustion detection
- Goal divergence detection
- Critic gaming detection
- Auto-repair of fixable violations
- Rejection of critical violations
- Rollback on regression

### 7.4 Benchmark Framework (6 tests)
- Benchmark loading and validation
- Task evaluation (pass/fail)
- Composite score calculation
- LLM judge evaluation
- Head-to-head comparison
- Timeout handling

### 7.5 Strategy Search (6 tests)
- Search space definition
- Genetic algorithm convergence
- NSGA-II multi-objective optimization
- Bayesian optimization
- Early stopping
- Self-adaptive mutation rate

### 7.6 Evolution Loop (8 tests)
- Full evolution loop (5 generations)
- Population initialization
- Fitness convergence
- Budget-aware evolution
- Checkpoint save/rollback
- Safety violation handling
- Diversity maintenance
- Parallel evaluation

### 7.7 Kernel Integration (4 tests)
- Genome → AgentState mapping
- Event trace extraction
- Metric extraction from agent state
- Kernel reset between evaluations

### 7.8 Scale Scenarios (4 tests)
- 100-genome population for 20 generations
- 5-benchmark evaluation suite
- Concurrent evolution of 4 populations
- Evolution with 50% failure rate environment

---

## 8. File Layout

```
src/agentforge_x/
  evolution/
    __init__.py
    genome.py             # PromptGenome dataclass + serialization
    operators.py          # Mutation/crossover/selection operators
    search.py             # Search algorithms + search space
    loop.py               # EvolutionLoop + step logic
    safety.py             # SafetyGate + SafetyRule + violation handling
    rollback.py           # RollbackManager + checkpoint persistence
    evaluator.py          # Evaluator + benchmark runner
    kernel_adapter.py     # KernelAdapter (evolution ↔ kernel bridge)
    config.py             # EvolutionConfig TypedDict
    events.py             # Evolution event types + emission
    benchmarks/
      __init__.py
      loader.py           # Benchmark loading/validation
      builtin.py          # BuiltinBenchmarks definitions
      judge.py            # LLM judge evaluation
  tests/
    test_genome.py
    test_operators.py
    test_safety.py
    test_benchmarks.py
    test_search.py
    test_loop.py
    test_kernel_adapter.py
    test_scale.py
```

---

## 9. Key Design Decisions

1. **Kernel-agnostic evaluation**: The evaluator uses the kernel
   as a black box — it creates agents, runs them, and reads results
   via the KernelAdapter interface. This keeps evolution decoupled
   from kernel internals.

2. **Safety-first evolution**: Every genome passes through safety
   gates before entering the population. Violations can trigger
   auto-repair, rejection, or full rollback depending on severity.

3. **Multi-objective by default**: The evolution optimizes for
   accuracy, latency, cost, AND robustness simultaneously using
   NSGA-II. Single-objective is a special case.

4. **Self-adaptive mutation**: The mutation rate itself evolves —
   genomes that improve slowly get higher mutation rates, while
   stable genomes preserve their configuration.

5. **Reproducibility**: All randomness is seeded. The same seed +
   same benchmark + same initial population = same evolution trace.

6. **Graceful degradation**: If the kernel's budget is exhausted
   mid-evaluation, the evolution pauses, checkpoints, and resumes
   when budget is replenished.

7. **Pareto preservation**: For multi-objective search, the Pareto
   front of best genomes is always preserved, even if they don't
   score highest on any single metric.

---

## 10. Cross-Reference: Swarm-Sim Failure Mappings

The swarm-sim failure injection patterns (from the CEO's t_6e86ec56)
map directly to evolution safety categories:

| Swarm-Sim Failure | Evolution Safety Category | Detection Method |
|-------------------|---------------------------|------------------|
| crash | stability | Agent state → failed |
| stall | resource_exhaustion | Runtime exceeds budget |
| timeout | resource_exhaustion | Step timeout > threshold |
| claim-loss | goal_divergence | Task output ≠ expected |

This mapping ensures the evolution engine can detect and penalize
genomes that are prone to the same failure patterns observed in
production (the 600s starvation event).
