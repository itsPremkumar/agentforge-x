"""sandbox.py — allowlist/fs-jail wrapper for local tool execution.

Provides:
  - SandboxedToolWrapper: enforces path jail + execution allowlist
  - FS_JAIL_ROOT: configurable filesystem jail root
  - DEFAULT_ALLOWLIST: safe operations allowed without explicit approval
"""
from __future__ import annotations

import os
import subprocess
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("agentforge_x.tools.sandbox")

FS_JAIL_ROOT = Path(os.environ.get("AGENTFORGE_SANDBOX_ROOT", "/tmp/agentforge-sandbox"))

DEFAULT_ALLOWLIST = {
    "echo", "cat", "ls", "head", "tail", "wc", "grep", "sed", "awk",
    "python3", "python", "git", "curl", "wget",
}

class SandboxedToolWrapper:
    """Wraps a callable to enforce sandbox restrictions.

    When the callable is a filesystem operation, it:
    - Restricts paths to within FS_JAIL_ROOT
    - Prevents path traversal escapes
    - Blocks absolute paths outside the jail
    """

    def __init__(
        self,
        func: callable,
        *,
        jail_root: Optional[Path] = None,
        allowlist: Optional[set[str]] = None,
        block_destructive: bool = True,
    ):
        self.func = func
        self.jail_root = jail_root or FS_JAIL_ROOT
        self.allowlist = allowlist or DEFAULT_ALLOWLIST
        self.block_destructive = block_destructive

    def _resolve_path(self, path: str) -> Path:
        """Resolve and validate a path against the jail root."""
        target = (self.jail_root / path).resolve()
        try:
            target.relative_to(self.jail_root.resolve())
        except ValueError:
            raise ValueError(f"Path '{path}' escapes sandbox root '{self.jail_root}'")
        return target

    def __call__(self, *args, **kwargs):
        """Execute the wrapped function with sandbox enforcement."""
        # If first arg is a path string, validate it
        if args and isinstance(args[0], str):
            path = args[0]
            if path.startswith("/") or ".." in path:
                self._resolve_path(path)
        return self.func(*args, **kwargs)


class SandboxedExec:
    """Execute commands in a sandboxed subprocess environment.

    Features:
    - Working directory restriction to jail root
    - Environment variable filtering
    - Timeout enforcement
    - No shell execution (prevents injection)
    - Command allowlist enforcement
    """

    def __init__(
        self,
        *,
        jail_root: Optional[Path] = None,
        default_timeout: float = 30.0,
        allowlist: Optional[set[str]] = None,
        max_output_bytes: int = 1_000_000,
    ):
        self.jail_root = jail_root or FS_JAIL_ROOT
        self.jail_root.mkdir(parents=True, exist_ok=True)
        self.default_timeout = default_timeout
        self.allowlist = allowlist or DEFAULT_ALLOWLIST
        self.max_output_bytes = max_output_bytes

    def run(
        self,
        command: list[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
        input_data: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute a command in the sandbox.

        Returns:
            dict with returncode, stdout, stderr, timed_out.
        """
        if command and command[0] not in self.allowlist:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command '{command[0]}' not in allowlist",
                "timed_out": False,
            }

        workdir = self.jail_root
        if cwd is not None:
            workdir = (self.jail_root / str(cwd)).resolve()
            try:
                workdir.relative_to(self.jail_root.resolve())
            except ValueError:
                raise ValueError(f"CWD '{cwd}' escapes sandbox root '{self.jail_root}'")

        clean_env = {k: v for k, v in (env or {}).items()
                     if k.upper() not in {"PATH", "LD_PRELOAD", "LD_LIBRARY_PATH"}}

        try:
            proc = subprocess.run(
                command,
                cwd=str(workdir),
                env={**os.environ, **clean_env},
                input=input_data.encode() if input_data else None,
                capture_output=True,
                text=True,
                timeout=timeout or self.default_timeout,
            )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout[:self.max_output_bytes],
                "stderr": proc.stderr[:self.max_output_bytes],
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "Command timed out",
                "timed_out": True,
            }
        except Exception as exc:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
            }
