#!/usr/bin/env python3
"""filesystem-safe MCP server — sandboxed filesystem read operations.

Exposes safe read-only tools for listing directories, reading files,
searching for files by name, and reading file metadata — all confined to
the MCP_TOOLBOX_ROOT sandbox (defaults to the current working directory).

Transport: stdio (default)
Run:       python -m mcp_toolbox.filesystem_safe
           mcp-toolbox-filesystem-safe
"""

from __future__ import annotations

import fnmatch

from mcp_toolbox.protocol_utils import (
    make_server,
    resolve_root,
    safe_path,
)

# ─── Tool implementations ─────────────────────────────────────────

def list_directory(path: str = ".") -> dict:
    """List entries in a directory.

    Args:
        path: Directory path relative to the sandbox root.
    """
    root = resolve_root()
    target = safe_path(path, root=root, must_exist=True)
    entries = []
    for child in sorted(target.iterdir()):
        entries.append({
            "name": child.name,
            "type": "directory" if child.is_dir() else "file",
            "size": child.stat().st_size if child.is_file() else None,
        })
    return {"entries": entries, "directory": str(target)}


def read_file(path: str, limit: int = 2000) -> dict:
    """Read a text file (first *limit* characters) from the sandbox.

    Args:
        path:  File path relative to the sandbox root.
        limit: Maximum number of characters to return.
    """
    root = resolve_root()
    target = safe_path(path, root=root, must_exist=True)
    if not target.is_file():
        return {"isError": True,
                "content": [{"type": "text",
                             "text": f"'{path}' is not a file"}]}
    text = target.read_text(encoding="utf-8", errors="replace")
    content = text[:limit]
    truncated = len(text) > limit
    return {
        "content": content,
        "path": str(target.name),
        "truncated": truncated,
        "truncated_at": limit if truncated else None,
    }


def file_stat(path: str) -> dict:
    """Return filesystem metadata for a path.

    Args:
        path: Path relative to the sandbox root.
    """
    root = resolve_root()
    target = safe_path(path, root=root, must_exist=True)
    st = target.stat()
    return {
        "path": str(target.name),
        "size_bytes": st.st_size,
        "is_file": target.is_file(),
        "is_dir": target.is_dir(),
        "is_symlink": target.is_symlink(),
        "modified_at": st.st_mtime,
        "created_at": st.st_ctime,
    }


def find_files(name_pattern: str, path: str = ".", max_results: int = 50) -> dict:
    """Find files whose name matches a glob pattern within the sandbox.

    Args:
        name_pattern: Glob pattern (e.g. '*.py', 'test_*').
        path:         Search root directory (relative to sandbox).
        max_results:  Cap on the number of matches returned.
    """
    root = resolve_root()
    target = safe_path(path, root=root, must_exist=True)
    matches = []
    for item in target.rglob("*"):
        if len(matches) >= max_results:
            break
        if not item.is_file():
            continue
        if fnmatch.fnmatch(item.name, name_pattern):
            rel = item.relative_to(root)
            matches.append({"path": str(rel), "size": item.stat().st_size})
    return {"matches": matches, "pattern": name_pattern}


# ─── Server assembly ────────────────────────────────────────────────

def build_server():
    server = make_server(
        name="filesystem-safe",
        description="Sandboxed filesystem read operations "
                    "(list, read, stat, find).",
        instructions="All paths are confined to the MCP_TOOLBOX_ROOT sandbox. "
                     "Absolute paths outside the root are rejected.",
    )

    server.add_tool(list_directory,
                    name="list_directory",
                    description="List entries in a directory.")
    server.add_tool(read_file,
                    name="read_file",
                    description="Read a text file (first N characters).")
    server.add_tool(file_stat,
                    name="file_stat",
                    description="Return filesystem metadata for a path.")
    server.add_tool(find_files,
                    name="find_files",
                    description="Find files matching a glob pattern.")

    return server


def main():
    import logging
    import sys
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
