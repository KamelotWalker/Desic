"""SQLite persistence for models, decisions (the feedback log) and datasets."""

from __future__ import annotations

import json
import pickle
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    name        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    schema      TEXT NOT NULL,
    rules       TEXT NOT NULL DEFAULT '[]',
    meta        TEXT NOT NULL DEFAULT '{}',
    state       BLOB,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id               TEXT PRIMARY KEY,
    model            TEXT NOT NULL,
    created_at       REAL NOT NULL,
    features         TEXT NOT NULL,
    prediction       TEXT,
    model_prediction TEXT,
    confidence       REAL NOT NULL DEFAULT 0,
    probabilities    TEXT NOT NULL DEFAULT '{}',
    source           TEXT NOT NULL,
    rule_id          TEXT,
    label            TEXT,
    correct          INTEGER,
    feedback_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_decisions_model ON decisions(model, created_at);
CREATE TABLE IF NOT EXISTS datasets (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    source     TEXT NOT NULL,
    columns    TEXT NOT NULL,
    meta       TEXT NOT NULL DEFAULT '{}',
    n_rows     INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_rows (
    dataset_id TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    data       TEXT NOT NULL,
    PRIMARY KEY (dataset_id, idx)
);
"""


class Storage:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA_SQL)
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    # ------------------------------------------------------------------ models
    def save_model(self, name: str, kind: str, description: str, schema: dict, rules: list, meta: dict, state: Any) -> None:
        now = time.time()
        blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        with self.lock:
            self.conn.execute(
                """INSERT INTO models(name, kind, description, schema, rules, meta, state, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET kind=excluded.kind, description=excluded.description,
                     schema=excluded.schema, rules=excluded.rules, meta=excluded.meta, state=excluded.state,
                     updated_at=excluded.updated_at""",
                (name, kind, description, json.dumps(schema), json.dumps(rules), json.dumps(meta), blob, now, now),
            )
            self.conn.commit()

    def load_models(self) -> list[dict]:
        # Model state is pickled by this process into its own data directory;
        # never point Desic at a database you did not create.
        with self.lock:
            rows = self.conn.execute("SELECT * FROM models ORDER BY created_at").fetchall()
        out = []
        for r in rows:
            out.append({
                "name": r["name"],
                "kind": r["kind"],
                "description": r["description"],
                "schema": json.loads(r["schema"]),
                "rules": json.loads(r["rules"]),
                "meta": json.loads(r["meta"]),
                "state": pickle.loads(r["state"]) if r["state"] else None,
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })
        return out

    def delete_model(self, name: str) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM models WHERE name=?", (name,))
            self.conn.execute("DELETE FROM decisions WHERE model=?", (name,))
            self.conn.commit()

    # --------------------------------------------------------------- decisions
    def insert_decision(self, d: dict) -> None:
        with self.lock:
            self.conn.execute(
                """INSERT INTO decisions(id, model, created_at, features, prediction, model_prediction, confidence,
                                         probabilities, source, rule_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (d["id"], d["model"], d["created_at"], json.dumps(d["features"]), d["prediction"], d["model_prediction"],
                 d["confidence"], json.dumps(d["probabilities"]), d["source"], d.get("rule_id")),
            )
            self.conn.commit()

    def set_feedback(self, decision_id: str, label: str, correct: bool) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE decisions SET label=?, correct=?, feedback_at=? WHERE id=?",
                (label, int(correct), time.time(), decision_id),
            )
            self.conn.commit()

    @staticmethod
    def _decision(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["features"] = json.loads(d["features"])
        d["probabilities"] = json.loads(d["probabilities"])
        d["correct"] = None if d["correct"] is None else bool(d["correct"])
        return d

    def get_decision(self, decision_id: str) -> dict | None:
        with self.lock:
            r = self.conn.execute("SELECT * FROM decisions WHERE id=?", (decision_id,)).fetchone()
        return self._decision(r) if r else None

    def list_decisions(self, model: str, limit: int = 50, pending: bool = False, uncertain_first: bool = False) -> list[dict]:
        sql = "SELECT * FROM decisions WHERE model=?"
        if pending:
            sql += " AND label IS NULL"
        sql += " ORDER BY confidence ASC, created_at DESC" if uncertain_first else " ORDER BY created_at DESC"
        sql += " LIMIT ?"
        with self.lock:
            rows = self.conn.execute(sql, (model, limit)).fetchall()
        return [self._decision(r) for r in rows]

    def count_pending(self, model: str) -> int:
        with self.lock:
            return self.conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE model=? AND label IS NULL", (model,)
            ).fetchone()[0]

    # ---------------------------------------------------------------- datasets
    def create_dataset(self, name: str, source: str, columns: list[str], rows: list[dict], meta: dict | None = None) -> str:
        ds_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.conn.execute(
                "INSERT INTO datasets(id, name, source, columns, meta, n_rows, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ds_id, name, source, json.dumps(columns), json.dumps(meta or {}), len(rows), time.time()),
            )
            self.conn.executemany(
                "INSERT INTO dataset_rows(dataset_id, idx, data) VALUES (?, ?, ?)",
                ((ds_id, i, json.dumps(r)) for i, r in enumerate(rows)),
            )
            self.conn.commit()
        return ds_id

    def list_datasets(self) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [self._dataset(r) for r in rows]

    @staticmethod
    def _dataset(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["columns"] = json.loads(d["columns"])
        d["meta"] = json.loads(d["meta"])
        return d

    def get_dataset(self, ds_id: str) -> dict | None:
        with self.lock:
            r = self.conn.execute("SELECT * FROM datasets WHERE id=?", (ds_id,)).fetchone()
        return self._dataset(r) if r else None

    def dataset_rows(self, ds_id: str, limit: int | None = None, offset: int = 0) -> list[dict]:
        sql = "SELECT data FROM dataset_rows WHERE dataset_id=? ORDER BY idx"
        params: tuple = (ds_id,)
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = (ds_id, limit, offset)
        with self.lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def iter_dataset_rows(self, ds_id: str, batch: int = 1000) -> Iterator[dict]:
        offset = 0
        while True:
            chunk = self.dataset_rows(ds_id, batch, offset)
            if not chunk:
                return
            yield from chunk
            offset += batch

    def delete_dataset(self, ds_id: str) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM datasets WHERE id=?", (ds_id,))
            self.conn.execute("DELETE FROM dataset_rows WHERE dataset_id=?", (ds_id,))
            self.conn.commit()
