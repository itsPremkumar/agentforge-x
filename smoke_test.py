#!/usr/bin/env python3
"""Smoke test: verify all three MCP servers import and build correctly."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mcp_toolbox.filesystem_safe import build_server as build_fs
    fs_server = build_fs()
    print("filesystem-safe: OK, server name =", fs_server.name)
except Exception as e:
    print("filesystem-safe: FAILED:", e)
    sys.exit(1)

try:
    from mcp_toolbox.git_inspector import build_server as build_git
    git_server = build_git()
    print("git-inspector: OK, server name =", git_server.name)
except Exception as e:
    print("git-inspector: FAILED:", e)
    sys.exit(1)

try:
    from mcp_toolbox.board_reader import build_server as build_board
    board_server = build_board()
    print("board-reader: OK, server name =", board_server.name)
except Exception as e:
    print("board-reader: FAILED:", e)
    sys.exit(1)

print("\nAll servers built successfully.")
