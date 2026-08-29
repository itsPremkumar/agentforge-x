"""SQLite checkpoint manager for the agentforge-x kernel.

Stores checkpoints as JSON in a SQLite database, keyed by (run_id, agent_id, seq).
Supports save, load, and checkpoint listing for resumable agent runs.

Schema matches the architectural spec:
    CREATE TABLE checkpoints (
        run_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        ts REAL NOT NULL,
        state_json TEXT NOT NULL,
        PRIMARY KEY (run_id, agent_id, seq)
    );
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Optional

from agentforge_x.kernel.state import AgentState


class SQLiteCheckpointStore:
    """Persistent checkpoint storage backed by SQLite.

    Thread-safe. Each checkpoint is a row in the `checkpoints` table.
    Checkpoints are versioned by (run_id, agent_id, seq).
    """

    def __init__(self, db_path: str = ":memory:"):
        """Initialize the checkpoint store.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for ephemeral
                     in-memory storage (not shared across connections).
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        """Create the checkpoints table if it doesn't exist."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                ts REAL NOT NULL,
                state_json TEXT NOT NULL,
                PRIMARY KEY (run_id, agent_id, seq)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoint_run
            ON checkpoints (run_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoint_agent
            ON checkpoints (run_id, agent_id)
        """)
        conn.commit()

    def save(self, run_id: str, agent_id: str, seq: int, ts: float, state: AgentState) -> None:
        """Save a checkpoint for the given run/agent/seq.

        Overwrites any existing checkpoint at the same key.
        """
        conn = self._get_conn()
        state_json = json.dumps(state.to_dict(), default=str)
        conn.execute(
            """
            INSERT OR REPLACE INTO checkpoints (run_id, agent_id, seq, ts, state_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, agent_id, seq, ts, state_json),
        )
        conn.commit()

    def load(
        self,
        run_id: str,
        agent_id: str,
        seq: Optional[int] = None,
    ) -> Optional[tuple[float, AgentState]]:
        """Load a checkpoint.

        If seq is None, loads the latest checkpoint for the run/agent.
        Returns (ts, state) or None if no checkpoint exists.
        """
        conn = self._get_conn()
        if seq is not None:
            row = conn.execute(
                "SELECT ts, state_json FROM checkpoints WHERE run_id = ? AND agent_id = ? AND seq = ?",
                (run_id, agent_id, seq),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT ts, state_json FROM checkpoints
                WHERE run_id = ? AND agent_id = ?
                ORDER BY seq DESC LIMIT 1
                """,
                (run_id, agent_id),
            ).fetchone()

        if row is None:
            return None

        data = json.loads(row["state_json"])
        return (row["ts"], AgentState.from_dict(data))

    def list_checkpoints(self, run_id: str, agent_id: str) -> list[int]:
        """Return all seq numbers for a given run/agent, sorted ascending."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT seq FROM checkpoints WHERE run_id = ? AND agent_id = ? ORDER BY seq ASC",
            (run_id, agent_id),
        ).fetchall()
        return [row["seq"] for row in rows]

    def latest_seq(self, run_id: str, agent_id: str) -> Optional[int]:
        """Return the latest seq number, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT MAX(seq) as max_seq FROM checkpoints WHERE run_id = ? AND agent_id = ?",
            (run_id, agent_id),
        ).fetchone()
        return row["max_seq"] if row and row["max_seq"] is not None else None

    def delete_run(self, run_id: str) -> int:
        """Delete all checkpoints for a run. Returns count of deleted rows."""
        conn = self._get_conn()
        cur = conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
        conn.commit()
        return cur.rowcount

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

    def __enter__(self) -> "SQLiteCheckpointStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
