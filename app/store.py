import json
import sqlite3
from contextlib import closing
from pathlib import Path

from app.messages import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, created_at TEXT, finished_at TEXT, status TEXT,
  provider TEXT, model TEXT, request TEXT, result TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts TEXT, agent TEXT, kind TEXT,
  name TEXT, status TEXT, duration_ms INTEGER, payload TEXT
);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id);
"""


class Store:
    """Explicit task state: SQLite tables `runs` and `events` (every agent and tool call)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)  # autocommit, safe across threads
        conn.row_factory = sqlite3.Row
        return conn

    def _write(self, sql: str, params: tuple) -> None:
        with closing(self._connect()) as conn:
            conn.execute(sql, params)

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with closing(self._connect()) as conn:
            return [dict(row) for row in conn.execute(sql, params)]

    def create_run(self, run_id: str, request: dict, provider: str, model: str) -> None:
        self._write(
            "INSERT INTO runs(run_id, created_at, status, provider, model, request) VALUES (?,?,?,?,?,?)",
            (run_id, now_iso(), "queued", provider, model, json.dumps(request, ensure_ascii=False)),
        )

    def set_status(self, run_id: str, status: str) -> None:
        self._write("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))

    def finish_run(self, run_id: str, status: str, result: dict | None, error: str | None) -> None:
        self._write(
            "UPDATE runs SET status=?, finished_at=?, result=?, error=? WHERE run_id=?",
            (status, now_iso(), json.dumps(result, ensure_ascii=False) if result else None, error, run_id),
        )

    def get_run(self, run_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return self._parse_run(rows[0]) if rows else None

    def list_runs(self, limit: int = 20) -> list[dict]:
        rows = self._rows("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._parse_run(row) for row in rows]

    @staticmethod
    def _parse_run(row: dict) -> dict:
        row["request"] = json.loads(row["request"]) if row["request"] else None
        row["result"] = json.loads(row["result"]) if row["result"] else None
        return row

    def add_event(
        self, run_id: str, agent: str, kind: str, name: str, status: str, duration_ms: int | None, payload: dict
    ) -> None:
        self._write(
            "INSERT INTO events(run_id, ts, agent, kind, name, status, duration_ms, payload) VALUES (?,?,?,?,?,?,?,?)",
            (
                run_id,
                now_iso(),
                agent,
                kind,
                name,
                status,
                duration_ms,
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )

    def events(self, run_id: str) -> list[dict]:
        rows = self._rows("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))
        for row in rows:
            row["payload"] = json.loads(row["payload"])
        return rows

    def workload(self, run_id: str) -> dict:
        """Share of LLM + tool calls per agent (the <= 40 % rule is checked on this)."""
        rows = self._rows(
            "SELECT agent, COUNT(*) AS n FROM events "
            "WHERE run_id=? AND kind IN ('llm_call','tool_call') GROUP BY agent",
            (run_id,),
        )
        calls = {row["agent"]: row["n"] for row in rows}
        total = sum(calls.values())
        shares = {agent: n / total for agent, n in calls.items()} if total else {}
        return {
            "calls_by_agent": calls,
            "total_calls": total,
            "shares": shares,
            "max_share": max(shares.values(), default=0.0),
            "threshold": 0.4,
        }
