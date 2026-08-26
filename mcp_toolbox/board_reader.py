#!/usr/bin/env python3
"""board-reader MCP server — kanban query tools matching the Hermes scan schema.

Provides read-only query tools for kanban boards stored in a simple
JSON-lines file format. Each line is a JSON object with fields:
  id, title, status, priority, assignee, created_at, updated_at, tags

The board file path is configurable via the MCP_TOOLBOX_BOARD env var
(defaults to <root>/board.jsonl).

Transport: stdio (default)
Run:       python -m mcp_toolbox.board_reader
           mcp-toolbox-board-reader
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp_toolbox.protocol_utils import make_server, resolve_root

# ─── Board data access ──────────────────────────────────────────────

def _board_path() -> Path:
    """Resolve the kanban board file path."""
    env_path = os.environ.get("MCP_TOOLBOX_BOARD")
    if env_path:
        return Path(env_path).resolve()
    return resolve_root() / "board.jsonl"


def _load_cards() -> list[dict]:
    """Load all cards from the board file. Returns empty list if file missing."""
    path = _board_path()
    if not path.exists():
        return []
    cards = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cards.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return cards


def _validate_card(card: dict) -> dict:
    """Ensure the card has the required schema fields, filling defaults."""
    return {
        "id": card.get("id", ""),
        "title": card.get("title", ""),
        "status": card.get("status", "todo"),
        "priority": card.get("priority", "normal"),
        "assignee": card.get("assignee", ""),
        "created_at": card.get("created_at", ""),
        "updated_at": card.get("updated_at", ""),
        "tags": card.get("tags", []),
        "description": card.get("description", ""),
    }


# ─── Tool implementations ─────────────────────────────────────────

def list_cards(status: str = None, assignee: str = None,
               tag: str = None, priority: str = None) -> dict:
    """Query cards on the kanban board with optional filters.

    Args:
        status:   Filter by status (e.g. 'todo', 'in_progress', 'done').
        assignee: Filter by assignee name.
        tag:      Filter by tag (cards whose tags list contains *tag*).
        priority: Filter by priority ('low', 'normal', 'high', 'urgent').
    """
    cards = [_validate_card(c) for c in _load_cards()]

    if status:
        cards = [c for c in cards if c["status"] == status]
    if assignee:
        cards = [c for c in cards if c["assignee"] == assignee]
    if tag:
        cards = [c for c in cards if tag in c["tags"]]
    if priority:
        cards = [c for c in cards if c["priority"] == priority]

    return {"cards": cards, "count": len(cards)}


def get_card(card_id: str) -> dict:
    """Retrieve a single card by its ID.

    Args:
        card_id: The unique identifier of the card.
    """
    cards = [_validate_card(c) for c in _load_cards()]
    for card in cards:
        if card["id"] == card_id:
            return {"card": card}
    return {"isError": True,
            "content": [{"type": "text",
                         "text": f"Card '{card_id}' not found"}]}


def board_summary() -> dict:
    """Return an aggregate summary of the board: counts per status and priority."""
    cards = [_validate_card(c) for c in _load_cards()]
    by_status = {}
    by_priority = {}
    by_assignee = {}
    for c in cards:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1
        by_priority[c["priority"]] = by_priority.get(c["priority"], 0) + 1
        if c["assignee"]:
            by_assignee[c["assignee"]] = by_assignee.get(c["assignee"], 0) + 1
    return {
        "total_cards": len(cards),
        "by_status": by_status,
        "by_priority": by_priority,
        "by_assignee": by_assignee,
        "board_file": str(_board_path()),
    }


def search_cards(query: str) -> dict:
    """Search card titles and descriptions for a keyword (case-insensitive).

    Args:
        query: Search term.
    """
    cards = [_validate_card(c) for c in _load_cards()]
    q = query.lower().strip()
    results = [
        c for c in cards
        if q in c["title"].lower() or q in c.get("description", "").lower()
    ]
    return {"cards": results, "count": len(results), "query": query}


# ─── Server assembly ────────────────────────────────────────────────

def build_server():
    server = make_server(
        name="board-reader",
        description="Read-only kanban board query tools "
                    "(list, get, summary, search).",
        instructions="Queries a JSONL kanban board file. The board file path "
                     "is set via MCP_TOOLBOX_BOARD or defaults to board.jsonl "
                     "in the sandbox root.",
    )

    server.add_tool(list_cards,
                    name="list_cards",
                    description="List cards with optional filters by status, assignee, tag, priority.")
    server.add_tool(get_card,
                    name="get_card",
                    description="Retrieve a single card by ID.")
    server.add_tool(board_summary,
                    name="board_summary",
                    description="Return aggregate counts by status, priority, and assignee.")
    server.add_tool(search_cards,
                    name="search_cards",
                    description="Search card titles/descriptions for a keyword.")

    return server


def main():
    import logging
    import sys
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
