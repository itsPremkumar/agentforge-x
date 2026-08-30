#!/usr/bin/env python3
"""git-inspector MCP server — repository health and metadata tools.

Exposes tools to inspect git repositories: checking commit status,
branch info, recent commit log, diff stat, and detecting untracked files.
All operations are read-only (no git mutations).

Transport: stdio (default)
Run:       python -m mcp_toolbox.git_inspector
           mcp-toolbox-git-inspector
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from mcp_toolbox.protocol_utils import make_server, resolve_root, safe_path

# ─── Tool implementations ─────────────────────────────────────────

def _run_git(repo_path: Path, args: list[str], timeout: int = 10) -> str:
    """Run a git command in *repo_path* and return stdout (empty string on error)."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def git_status(repo_path: str = ".") -> dict:
    """Show the working-tree status (branch, staged/unstaged changes, untracked).

    Args:
        repo_path: Path to a git repository (relative to sandbox root).
    """
    target = safe_path(repo_path, root=resolve_root(), must_exist=True)
    if not (target / ".git").exists():
        return {"isError": True,
                "content": [{"type": "text",
                             "text": f"'{repo_path}' is not a git repository"}]}
    branch = _run_git(target, ["rev-parse", "--abbrev-ref", "HEAD"])
    staged = _run_git(target, ["diff", "--cached", "--short"])
    unstaged = _run_git(target, ["diff", "--short"])
    untracked = _run_git(target, ["ls-files", "--others", "--exclude-standard"])
    return {
        "repo": str(target.name),
        "branch": branch or "unknown",
        "staged_changes": staged.split("\n") if staged else [],
        "unstaged_changes": unstaged.split("\n") if unstaged else [],
        "untracked_files": untracked.split("\n") if untracked else [],
        "is_dirty": bool(staged or unstaged or untracked),
    }


def git_recent_commits(repo_path: str = ".", count: int = 10) -> dict:
    """List the most recent commits.

    Args:
        repo_path: Path to a git repository.
        count:     Maximum number of commits to return.
    """
    target = safe_path(repo_path, root=resolve_root(), must_exist=True)
    if not (target / ".git").exists():
        return {"isError": True,
                "content": [{"type": "text",
                             "text": f"'{repo_path}' is not a git repository"}]}
    log = _run_git(target, ["log", f"-{count}", "--oneline", "--pretty=format:%H|%an|%s"])
    commits = []
    if log:
        for line in log.split("\n"):
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append({"hash": parts[0], "author": parts[1], "message": parts[2]})
    return {"repo": str(target.name), "commits": commits}


def git_diff_stat(repo_path: str = "."):
    """Show diff stat for the working tree vs HEAD.

    Args:
        repo_path: Path to a git repository.
    """
    target = safe_path(repo_path, root=resolve_root(), must_exist=True)
    if not (target / ".git").exists():
        return {"isError": True,
                "content": [{"type": "text",
                             "text": f"'{repo_path}' is not a git repository"}]}
    diff = _run_git(target, ["diff", "--stat"])
    return {"repo": str(target.name), "diff_stat": diff or "(no changes)"}


def git_branches(repo_path: str = ".") -> dict:
    """List all branches (local and remote).

    Args:
        repo_path: Path to a git repository.
    """
    target = safe_path(repo_path, root=resolve_root(), must_exist=True)
    if not (target / ".git").exists():
        return {"isError": True,
                "content": [{"type": "text",
                             "text": f"'{repo_path}' is not a git repository"}]}
    branches = _run_git(target, ["branch", "-a"])
    remotes = _run_git(target, ["branch", "-r"])
    return {
        "repo": str(target.name),
        "local_branches": [b.strip("* ").strip() for b in branches.split("\n") if b.strip()] if branches else [],
        "remote_branches": remotes.split("\n") if remotes else [],
    }


# ─── Server assembly ────────────────────────────────────────────────

def build_server():
    server = make_server(
        name="git-inspector",
        description="Read-only git repository health and metadata tools.",
        instructions="All git operations are read-only — no commits, pushes, "
                     "or mutations. Repos are confined to the MCP_TOOLBOX_ROOT sandbox.",
    )

    server.add_tool(git_status,
                    name="git_status",
                    description="Show working-tree status (branch, staged/unstaged/untracked).")
    server.add_tool(git_recent_commits,
                    name="git_recent_commits",
                    description="List the most recent commits.")
    server.add_tool(git_diff_stat,
                    name="git_diff_stat",
                    description="Show diff stat for working tree vs HEAD.")
    server.add_tool(git_branches,
                    name="git_branches",
                    description="List all local and remote branches.")

    return server


def main():
    import logging
    import sys
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
