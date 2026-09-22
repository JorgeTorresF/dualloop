"""Persistence backends for decisions, outcomes and
calibration/bandit/threshold state.

`InMemoryStore` is the default (zero configuration). `SQLiteStore` uses
only the standard library's `sqlite3` — no external dependencies — so that
anyone can persist state across process restarts without standing up a
separate database. Both implement the same `Store` protocol, so you can
swap in your own backend (Postgres, Redis...) by implementing the same
methods.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from typing import Optional, Protocol

from .types import Decision, Outcome, Question, Vote


class Store(Protocol):
    def save_decision(self, decision: Decision) -> None: ...
    def get_decision(self, decision_id: str) -> Optional[Decision]: ...
    def save_outcome(self, outcome: Outcome) -> None: ...
    def get_outcomes(self, decision_id: str) -> list[Outcome]: ...
    def save_state_blob(self, key: str, blob: dict) -> None: ...
    def load_state_blob(self, key: str) -> Optional[dict]: ...


class InMemoryStore:
    """No persistence across processes -- ideal for tests and demos."""

    def __init__(self) -> None:
        self._decisions: dict[str, Decision] = {}
        self._outcomes: dict[str, list[Outcome]] = {}
        self._blobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def save_decision(self, decision: Decision) -> None:
        with self._lock:
            self._decisions[decision.id] = decision

    def get_decision(self, decision_id: str) -> Optional[Decision]:
        return self._decisions.get(decision_id)

    def save_outcome(self, outcome: Outcome) -> None:
        with self._lock:
            self._outcomes.setdefault(outcome.decision_id, []).append(outcome)

    def get_outcomes(self, decision_id: str) -> list[Outcome]:
        return list(self._outcomes.get(decision_id, []))

    def save_state_blob(self, key: str, blob: dict) -> None:
        with self._lock:
            self._blobs[key] = blob

    def load_state_blob(self, key: str) -> Optional[dict]:
        return self._blobs.get(key)


def _decision_to_jsonable(d: Decision) -> dict:
    return asdict(d)


def _decision_from_jsonable(data: dict) -> Decision:
    data = dict(data)
    data["question"] = Question(**data["question"])
    data["votes"] = [Vote(**v) for v in data["votes"]]
    return Decision(**data)


class SQLiteStore:
    """Persistence across restarts using sqlite3 (stdlib, no dependencies)."""

    def __init__(self, path: str = "dualloop.db") -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS decisions "
            "(id TEXT PRIMARY KEY, data TEXT NOT NULL, created_at REAL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS outcomes "
            "(decision_id TEXT NOT NULL, data TEXT NOT NULL, reported_at REAL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS state_blobs (key TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self._conn.commit()

    def save_decision(self, decision: Decision) -> None:
        data = json.dumps(_decision_to_jsonable(decision))
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)",
                (decision.id, data, decision.created_at),
            )
            self._conn.commit()

    def get_decision(self, decision_id: str) -> Optional[Decision]:
        row = self._conn.execute(
            "SELECT data FROM decisions WHERE id = ?", (decision_id,)
        ).fetchone()
        if row is None:
            return None
        return _decision_from_jsonable(json.loads(row[0]))

    def save_outcome(self, outcome: Outcome) -> None:
        data = json.dumps(asdict(outcome))
        with self._lock:
            self._conn.execute(
                "INSERT INTO outcomes VALUES (?, ?, ?)",
                (outcome.decision_id, data, outcome.reported_at),
            )
            self._conn.commit()

    def get_outcomes(self, decision_id: str) -> list[Outcome]:
        rows = self._conn.execute(
            "SELECT data FROM outcomes WHERE decision_id = ? ORDER BY reported_at",
            (decision_id,),
        ).fetchall()
        return [Outcome(**json.loads(r[0])) for r in rows]

    def save_state_blob(self, key: str, blob: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO state_blobs VALUES (?, ?)",
                (key, json.dumps(blob)),
            )
            self._conn.commit()

    def load_state_blob(self, key: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT data FROM state_blobs WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None
