"""Safety gates for evolved genomes."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from agentforge_x.evolution.genome import PromptGenome


class SafetyCheckResult(BaseModel):
    """Result of a safety check."""

    passed: bool
    score: float = 1.0
    details: str = ""
    suggestion: str | None = None


class SafetyViolation(BaseModel):
    """A safety violation found in a genome."""

    rule_id: str
    genome_id: str
    severity: str  # warning | error | critical
    message: str
    action_taken: str = "logged"  # logged | repaired | rejected | rolled_back


class SafetyRule(BaseModel):
    """A safety rule to check genomes against."""

    id: str
    name: str
    description: str
    severity: str = "warning"
    category: str = "stability"
    evaluator: Callable[[PromptGenome], SafetyCheckResult] | None = None


class SafetyGate:
    """Safety gates prevent harmful genomes from entering the population."""

    def __init__(self):
        self.rules: list[SafetyRule] = []
        self._register_default_rules()

    def _register_default_rules(self):
        """Register the 8 default safety rules."""
        self.rules = [
            SafetyRule(
                id="prompt_injection",
                name="Prompt Injection Detection",
                description="Detects prompt injection attempts",
                severity="critical",
                category="prompt_injection",
                evaluator=self._check_prompt_injection,
            ),
            SafetyRule(
                id="tool_misuse",
                name="Tool Misuse Detection",
                description="Detects dangerous tool usage patterns",
                severity="error",
                category="tool_misuse",
                evaluator=self._check_tool_misuse,
            ),
            SafetyRule(
                id="resource_exhaustion",
                name="Resource Exhaustion Detection",
                description="Detects resource-wasting patterns",
                severity="warning",
                category="resource_exhaustion",
                evaluator=self._check_resource_exhaustion,
            ),
            SafetyRule(
                id="goal_divergence",
                name="Goal Divergence Detection",
                description="Detects drift from original goal",
                severity="warning",
                category="goal_divergence",
                evaluator=self._check_goal_divergence,
            ),
            SafetyRule(
                id="critic_gaming",
                name="Critic Gaming Detection",
                description="Detects attempts to game the critic",
                severity="error",
                category="critic_gaming",
                evaluator=self._check_critic_gaming,
            ),
            SafetyRule(
                id="stability",
                name="Stability Check",
                description="Checks for inconsistent behavior patterns",
                severity="warning",
                category="stability",
                evaluator=self._check_stability,
            ),
        ]

    def evaluate(self, genome: PromptGenome) -> list[SafetyViolation]:
        """Evaluate a genome against all safety rules."""
        violations = []
        for rule in self.rules:
            if rule.evaluator:
                result = rule.evaluator(genome)
                if not result.passed:
                    violations.append(SafetyViolation(
                        rule_id=rule.id,
                        genome_id=genome.id,
                        severity=rule.severity,
                        message=result.details,
                    ))
        return violations

    def _check_prompt_injection(self, genome: PromptGenome) -> SafetyCheckResult:
        """Check for prompt injection patterns."""
        suspicious = ["ignore previous", "disregard", "override", "jailbreak", "DAN"]
        for pattern in suspicious:
            if pattern.lower() in genome.system_prompt.lower():
                return SafetyCheckResult(
                    passed=False,
                    score=0.0,
                    details=f"Suspicious pattern detected: {pattern}",
                    suggestion="Remove the suspicious pattern from the system prompt",
                )
        return SafetyCheckResult(passed=True, score=1.0, details="No injection patterns detected")

    def _check_tool_misuse(self, genome: PromptGenome) -> SafetyCheckResult:
        """Check for dangerous tool usage patterns."""
        dangerous = ["rm -rf", "sudo", "chmod 777", "curl | bash"]
        for tool_prompt in genome.tool_descriptions:
            for pattern in dangerous:
                if pattern.lower() in tool_prompt.description.lower():
                    return SafetyCheckResult(
                        passed=False,
                        score=0.2,
                        details=f"Dangerous tool pattern: {pattern}",
                        suggestion="Remove dangerous tool usage patterns",
                    )
        return SafetyCheckResult(passed=True, score=1.0, details="No dangerous tool patterns")

    def _check_resource_exhaustion(self, genome: PromptGenome) -> SafetyCheckResult:
        """Check for resource-wasting patterns."""
        if genome.retry_policy.max_retries > 5:
            return SafetyCheckResult(
                passed=False,
                score=0.5,
                details=f"Excessive retries: {genome.retry_policy.max_retries}",
                suggestion="Reduce max_retries to 5 or less",
            )
        return SafetyCheckResult(passed=True, score=1.0, details="Resource usage acceptable")

    def _check_goal_divergence(self, genome: PromptGenome) -> SafetyCheckResult:
        """Check for goal divergence."""
        if len(genome.system_prompt) > 5000:
            return SafetyCheckResult(
                passed=False,
                score=0.6,
                details="System prompt is excessively long, may indicate goal drift",
                suggestion="Shorten the system prompt",
            )
        return SafetyCheckResult(passed=True, score=1.0, details="Goal alignment acceptable")

    def _check_critic_gaming(self, genome: PromptGenome) -> SafetyCheckResult:
        """Check for critic gaming patterns."""
        if genome.critic_rubric.pass_threshold < 0.5:
            return SafetyCheckResult(
                passed=False,
                score=0.3,
                details=f"Critic threshold too low: {genome.critic_rubric.pass_threshold}",
                suggestion="Increase pass_threshold to at least 0.5",
            )
        return SafetyCheckResult(passed=True, score=1.0, details="Critic configuration acceptable")

    def _check_stability(self, genome: PromptGenome) -> SafetyCheckResult:
        """Check for stability issues."""
        if len(genome.mutation_log) > 20:
            return SafetyCheckResult(
                passed=False,
                score=0.7,
                details=f"Excessive mutations: {len(genome.mutation_log)}",
                suggestion="Consider resetting to a stable baseline",
            )
        return SafetyCheckResult(passed=True, score=1.0, details="Mutation history acceptable")
