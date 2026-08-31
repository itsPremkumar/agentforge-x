"""Safety plugins — 6 runtime safety guardrails for AI agents.

Plugins:
    1. PromptInjectionGuard — Detects prompt injection attempts
    2. OutputLengthLimiter — Enforces max output length
    3. PIIRedactor — Redacts personally identifiable information
    4. ToxicityFilter — Filters toxic/harmful content
    5. RateLimiter — Enforces rate limits on tool calls
    6. CircuitBreaker — Stops agent after N consecutive failures
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SafetyAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    REDACT = "redact"


@dataclass
class SafetyResult:
    """Result of a safety check."""
    action: SafetyAction
    reason: str
    modified_output: str | None = None


class PromptInjectionGuard:
    """Detects prompt injection attempts in user input."""
    
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"new\s+persona",
        r"system\s+override",
        r"jailbreak",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(a|an)\s+",
        r"disregard\s+(all\s+)?instructions",
    ]
    
    def __init__(self, block_threshold: int = 1):
        self.block_threshold = block_threshold
    
    def check(self, text: str) -> SafetyResult:
        """Check text for prompt injection attempts."""
        matches = 0
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        if matches >= self.block_threshold:
            return SafetyResult(
                action=SafetyAction.BLOCK,
                reason=f"Prompt injection detected ({matches} patterns matched)",
            )
        
        return SafetyResult(action=SafetyAction.ALLOW, reason="No injection detected")


class OutputLengthLimiter:
    """Enforces maximum output length."""
    
    def __init__(self, max_chars: int = 10000, max_lines: int = 500):
        self.max_chars = max_chars
        self.max_lines = max_lines
    
    def check(self, text: str) -> SafetyResult:
        """Check if output exceeds length limits."""
        if len(text) > self.max_chars:
            return SafetyResult(
                action=SafetyAction.BLOCK,
                reason=f"Output exceeds max length ({len(text)} > {self.max_chars} chars)",
            )
        
        lines = text.count('\n') + 1
        if lines > self.max_lines:
            return SafetyResult(
                action=SafetyAction.BLOCK,
                reason=f"Output exceeds max lines ({lines} > {self.max_lines})",
            )
        
        return SafetyResult(action=SafetyAction.ALLOW, reason="Within limits")


class PIIRedactor:
    """Redacts personally identifiable information."""
    
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }
    
    def check(self, text: str) -> SafetyResult:
        """Check and redact PII from text."""
        modified = text
        found_pii = []
        
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, modified)
            if matches:
                found_pii.append(pii_type)
                modified = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", modified)
        
        if found_pii:
            return SafetyResult(
                action=SafetyAction.REDACT,
                reason=f"PII detected and redacted: {', '.join(found_pii)}",
                modified_output=modified,
            )
        
        return SafetyResult(action=SafetyAction.ALLOW, reason="No PII detected")


class ToxicityFilter:
    """Filters toxic/harmful content."""
    
    TOXIC_WORDS = {
        "hate", "kill", "harm", "attack", "destroy", "hurt", "hateful",
        "violent", "abuse", "threat", "murder", "assault",
    }
    
    def __init__(self, block_threshold: int = 1):
        self.block_threshold = block_threshold
    
    def check(self, text: str) -> SafetyResult:
        """Check text for toxic content."""
        words = set(text.lower().split())
        toxic_matches = words & self.TOXIC_WORDS
        
        if len(toxic_matches) >= self.block_threshold:
            return SafetyResult(
                action=SafetyAction.BLOCK,
                reason=f"Toxic content detected: {toxic_matches}",
            )
        
        return SafetyResult(action=SafetyAction.ALLOW, reason="No toxic content")


class RateLimiter:
    """Enforces rate limits on tool calls."""
    
    def __init__(self, max_calls: int = 10, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: list[float] = []
    
    def check(self, text: str = "") -> SafetyResult:
        """Check if rate limit is exceeded."""
        now = time.time()
        
        # Remove old calls outside the window
        self._calls = [t for t in self._calls if now - t < self.window_seconds]
        
        if len(self._calls) >= self.max_calls:
            return SafetyResult(
                action=SafetyAction.BLOCK,
                reason=f"Rate limit exceeded ({self.max_calls} calls per {self.window_seconds}s)",
            )
        
        self._calls.append(now)
        return SafetyResult(action=SafetyAction.ALLOW, reason="Within rate limit")
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        self._calls = []


class CircuitBreaker:
    """Stops agent after N consecutive failures."""
    
    def __init__(self, max_failures: int = 3):
        self.max_failures = max_failures
        self._failures = 0
        self._open = False
    
    def check(self, text: str = "") -> SafetyResult:
        """Check if circuit breaker is open."""
        if self._open:
            return SafetyResult(
                action=SafetyAction.BLOCK,
                reason=f"Circuit breaker open ({self.max_failures} consecutive failures)",
            )
        
        return SafetyResult(action=SafetyAction.ALLOW, reason="Circuit closed")
    
    def record_failure(self) -> None:
        """Record a failure."""
        self._failures += 1
        if self._failures >= self.max_failures:
            self._open = True
    
    def record_success(self) -> None:
        """Record a success, reset failure count."""
        self._failures = 0
    
    def reset(self) -> None:
        """Reset the circuit breaker."""
        self._failures = 0
        self._open = False
    
    @property
    def is_open(self) -> bool:
        return self._open
    
    @property
    def failure_count(self) -> int:
        return self._failures
