"""Shared utilities for MCP servers in the mcp-toolbox collection.

Provides:
  - make_server(): builds an MCPServer with standard metadata and error-wrapping
  - tool_wrapper(): decorator that wraps any tool callable so failures return
    CallToolResult(isError=True) instead of crashing the transport
  - safe_path(): resolves and validates a path against a sandbox root,
    rejecting traversal escapes and absolute paths outside the jail
  - resolve_root(): reads the MCP_TOOLBOX_ROOT env var or defaults to cwd
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ────────────────────────── server factory ──────────────────────────

def make_server(name: str, description: str, version: str = "0.1.0",
                instructions: Optional[str] = None):
    """Create a configured MCPServer instance.

    Uses the high-level MCPServer class from the mcp SDK (v2.0+).
    """
    from mcp.server import MCPServer

    return MCPServer(
        name=name,
        title=name.replace("-", " ").title(),
        description=description,
        instructions=instructions,
        version=version,
    )


# ────────────────────────── tool decorator ──────────────────────────

def tool_wrapper(description: str = "", title: Optional[str] = None,
                 annotations: Optional[dict] = None):
    """Decorator that registers a function as an MCP tool with safe error handling.

    The wrapped callable is registered on the server via add_tool(). The SDK
    derives inputSchema from the function's signature + type annotations.
    On any exception, the wrapper returns a structured error dict that the
    MCP layer converts to CallToolResult(isError=True).

    Usage:
        @tool_wrapper("Returns the sum of two numbers")
        def add(a: int, b: int) -> dict:
            return {"result": a + b}
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:

        @functools.wraps(fn)
        def safe_call(**kwargs: Any) -> Any:
            try:
                return fn(**kwargs)
            except Exception as exc:
                logger.exception("tool %s raised", fn.__name__)
                return {
                    "isError": True,
                    "content": [{"type": "text",
                                 "text": f"Error in {fn.__name__}: {exc}"}],
                }

        # Attach metadata for the server registration step
        safe_call.__mcp_description__ = description
        safe_call.__mcp_title__ = title
        safe_call.__mcp_annotations__ = annotations or {}
        return safe_call

    return decorator


# ────────────────────────── path safety ──────────────────────────

def resolve_root() -> Path:
    """Resolve the sandbox root directory.

    Reads MCP_TOOLBOX_ROOT env var; defaults to the current working directory.
    """
    env_root = os.environ.get("MCP_TOOLBOX_ROOT")
    if env_root:
        root = Path(env_root).resolve()
    else:
        root = Path.cwd().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_path(unsafe_path: str, root: Optional[Path] = None,
              must_exist: bool = True) -> Path:
    """Resolve *unsafe_path* under *root*, rejecting traversal escapes.

    - Empty path defaults to root itself.
    - Absolute paths are interpreted relative to root (not the real FS root).
    - Any resolved path that escapes root raises ValueError.

    Args:
        unsafe_path: User-supplied path string (may contain ../).
        root:        Sandbox root. Defaults to resolve_root().
        must_exist:  If True, raises FileNotFoundError when the resolved path
                     does not exist on disk.

    Returns:
        A resolved, validated Path strictly inside root.
    """
    if root is None:
        root = resolve_root()

    raw = unsafe_path.strip() if unsafe_path else ""
    if raw == "":
        target = root
    else:
        # Treat the input as relative to root regardless of whether it
        # starts with / — this prevents /etc/passwd escape.
        candidate = Path(raw)
        if candidate.is_absolute():
            candidate = Path(*candidate.parts[1:])  # strip leading "/"
        target = (root / candidate).resolve()

    # Boundary check: the resolved path must be root or inside it.
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Path '{unsafe_path}' escapes the sandbox root '{root}'"
        )

    if must_exist and not target.exists():
        raise FileNotFoundError(f"Path '{unsafe_path}' does not exist")

    return target


# ────────────────────────── JSON helpers ──────────────────────────

def to_text(result: Any) -> str:
    """Convert a tool return value to a text block suitable for MCP."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2, default=str)
    return str(result)
