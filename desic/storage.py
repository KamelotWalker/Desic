"""SQLite persistence.

* ``tasks``            – the live student of each question (pickled) + its spec
* ``snapshots``        – versioned copies of a student, for rollback
* ``decisions``        – every decision served (state, full answers)
* ``decision_answers`` – one row per (decision, question), for feeds and the review queue
* ``feedback_events``  – append-only log of every label learned (human, teacher, dataset);
                         a label can be retracted, never edited, so a student can be
                         rebuilt from the log at any time
* ``datasets``         – uploaded / generated data
"""

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
CREATE TABLE IF NOT EXISTS tasks (
    name       TEXT PRIMARY KEY,
    spec       TEXT NOT NULL,
    state      BLOB NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    task       TEXT NOT NULL,
    version    INTEGER NOT NULL,
    created_at REAL NOT NULL,
    labels     INTEGER NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    metrics    TEXT NOT NULL DEFAULT '{}',
    state      BLOB NOT NULL,
    PRIMARY KEY (task, version)
);
CREATE TABLE IF NOT EXISTS decisions (
    id         TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    state      TEXT NOT NULL,
    answers    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decision_answers (
    decision_id TEXT NOT NULL,
    task        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    answer      TEXT,
    confidence  REAL NOT NULL,
    abstain     INTEGER NOT NULL,
    source      TEXT NOT NULL,
    label       TEXT,
    PRIMARY KEY (decision_id, task)
);
CREATE INDEX IF NOT EXISTS idx_answers_task ON decision_answers(task, created_at);
CREATE TABLE IF NOT EXISTS feedback_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    task        TEXT NOT NULL,
    decision_id TEXT,
    state       TEXT NOT NULL,
    label       TEXT NOT NULL,
    source      TEXT NOT NULL,
    weight      REAL NOT NULL,
    retracted   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_task ON feedback_events(task, id);
CREATE TABLE IF NOT EXISTS neural_checkpoints (
    id         TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    status     TEXT NOT NULL,
    backbone   TEXT NOT NULL,
    path       TEXT NOT NULL,
    report     TEXT NOT NULL DEFAULT '{}',
    shadow     TEXT NOT NULL DEFAULT '{}',
    note       TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
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


def _dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


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

    def _all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(sql, params).fetchone()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    # ------------------------------------------------------------------- tasks
    # Student state is pickled by this process into its own data directory;
    # never point Desic at a database you did not create.
    def save_task(self, name: str, spec: dict, task: Any) -> None:
        now = time.time()
        self._exec(
            """INSERT INTO tasks(name, spec, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET spec=excluded.spec, state=excluded.state, updated_at=excluded.updated_at""",
            (name, _dumps(spec), pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL), now, now),
        )

    def load_tasks(self) -> list[Any]:
        return [pickle.loads(r["state"]) for r in self._all("SELECT state FROM tasks ORDER BY created_at")]

    def delete_task(self, name: str) -> None:
        with self.lock:
            for table in ("tasks", "snapshots", "decision_answers", "feedback_events"):
                col = "name" if table == "tasks" else "task"
                self.conn.execute(f"DELETE FROM {table} WHERE {col}=?", (name,))
            self.conn.commit()

    # --------------------------------------------------------------- snapshots
    KEEP_AUTOMATIC = 10  # automatic snapshots kept per task (a large student pickles to tens of MB)

    def save_snapshot(self, name: str, version: int, labels: int, metrics: dict, task: Any, note: str = "") -> None:
        self._exec(
            "INSERT OR REPLACE INTO snapshots(task, version, created_at, labels, note, metrics, state) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, version, time.time(), labels, note, _dumps(metrics), pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL)),
        )
        if note == "automatic":  # manual and "before rebuild" snapshots are never pruned
            self._exec(
                """DELETE FROM snapshots WHERE task=? AND note='automatic' AND version NOT IN (
                   SELECT version FROM snapshots WHERE task=? AND note='automatic' ORDER BY version DESC LIMIT ?)""",
                (name, name, self.KEEP_AUTOMATIC))

    def list_snapshots(self, name: str) -> list[dict]:
        rows = self._all("SELECT task, version, created_at, labels, note, metrics FROM snapshots WHERE task=? ORDER BY version DESC", (name,))
        return [{**dict(r), "metrics": json.loads(r["metrics"])} for r in rows]

    def load_snapshot(self, name: str, version: int) -> Any | None:
        r = self._one("SELECT state FROM snapshots WHERE task=? AND version=?", (name, version))
        return pickle.loads(r["state"]) if r else None

    # --------------------------------------------------------------- decisions
    def insert_decision(self, decision_id: str, created_at: float, state: Any, answers: dict, rows: list[tuple]) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO decisions(id, created_at, state, answers) VALUES (?, ?, ?, ?)",
                              (decision_id, created_at, _dumps(state), _dumps(answers)))
            self.conn.executemany(
                """INSERT INTO decision_answers(decision_id, task, created_at, answer, confidence, abstain, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(decision_id, task, created_at, answer, conf, int(abstain), source) for task, answer, conf, abstain, source in rows],
            )
            self.conn.commit()

    def get_decision(self, decision_id: str) -> dict | None:
        r = self._one("SELECT * FROM decisions WHERE id=?", (decision_id,))
        if r is None:
            return None
        labels = {x["task"]: x["label"] for x in self._all("SELECT task, label FROM decision_answers WHERE decision_id=?", (decision_id,))}
        return {"id": r["id"], "created_at": r["created_at"], "state": json.loads(r["state"]),
                "answers": json.loads(r["answers"]), "labels": labels}

    def set_answer_label(self, decision_id: str, task: str, label: str) -> None:
        self._exec("UPDATE decision_answers SET label=? WHERE decision_id=? AND task=?", (label, decision_id, task))

    def list_answers(self, task: str, limit: int = 50, pending: bool = False, uncertain_first: bool = False,
                     labelled: bool = False) -> list[dict]:
        sql = """SELECT a.*, d.state, d.answers FROM decision_answers a JOIN decisions d ON d.id = a.decision_id
                 WHERE a.task=?"""
        if pending:
            sql += " AND a.label IS NULL"
        if labelled:
            sql += " AND a.label IS NOT NULL"
        sql += " ORDER BY a.confidence ASC, a.created_at DESC" if uncertain_first else " ORDER BY a.created_at DESC"
        sql += " LIMIT ?"
        out = []
        for r in self._all(sql, (task, limit)):
            answers = json.loads(r["answers"])
            out.append({
                "decision_id": r["decision_id"], "task": r["task"], "created_at": r["created_at"],
                "answer": r["answer"], "confidence": r["confidence"], "abstain": bool(r["abstain"]),
                "source": r["source"], "label": r["label"], "state": json.loads(r["state"]),
                "result": answers.get(task, {}),
            })
        return out

    def count_pending(self, task: str) -> int:
        r = self._one("SELECT COUNT(*) FROM decision_answers WHERE task=? AND label IS NULL", (task,))
        return r[0] if r else 0

    # ----------------------------------------------------------- feedback log
    def add_event(self, task: str, state: Any, label: Any, source: str, weight: float, decision_id: str | None = None) -> int:
        cur = self._exec(
            "INSERT INTO feedback_events(created_at, task, decision_id, state, label, source, weight) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), task, decision_id, _dumps(state), _dumps(label), source, weight),
        )
        return int(cur.lastrowid)

    def add_events(self, rows: list[tuple]) -> list[int]:
        now = time.time()
        ids = []
        with self.lock:
            for task, did, state, label, source, weight in rows:
                cur = self.conn.execute(
                    "INSERT INTO feedback_events(created_at, task, decision_id, state, label, source, weight) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (now, task, did, _dumps(state), _dumps(label), source, weight))
                ids.append(int(cur.lastrowid))
            self.conn.commit()
        return ids

    @staticmethod
    def _event(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["state"] = json.loads(d["state"])
        d["label"] = json.loads(d["label"])
        d["retracted"] = bool(d["retracted"])
        return d

    def list_events(self, task: str, limit: int = 100, offset: int = 0) -> list[dict]:
        rows = self._all("SELECT * FROM feedback_events WHERE task=? ORDER BY id DESC LIMIT ? OFFSET ?", (task, limit, offset))
        return [self._event(r) for r in rows]

    def iter_events(self, task: str, include_retracted: bool = False, batch: int = 1000) -> Iterator[dict]:
        last = 0
        while True:
            sql = "SELECT * FROM feedback_events WHERE task=? AND id>?"
            if not include_retracted:
                sql += " AND retracted=0"
            rows = self._all(sql + " ORDER BY id LIMIT ?", (task, last, batch))
            if not rows:
                return
            for r in rows:
                yield self._event(r)
            last = rows[-1]["id"]

    def get_event(self, event_id: int) -> dict | None:
        r = self._one("SELECT * FROM feedback_events WHERE id=?", (event_id,))
        return self._event(r) if r else None

    def set_retracted(self, event_id: int, retracted: bool) -> None:
        self._exec("UPDATE feedback_events SET retracted=? WHERE id=?", (int(retracted), event_id))

    def event_counts(self, task: str) -> dict:
        rows = self._all("SELECT source, retracted, COUNT(*) AS n FROM feedback_events WHERE task=? GROUP BY source, retracted", (task,))
        out: dict[str, int] = {}
        for r in rows:
            key = "retracted" if r["retracted"] else r["source"]
            out[key] = out.get(key, 0) + r["n"]
        return out

    # ---------------------------------------------------------------- settings
    def get_setting(self, key: str, default: Any = None) -> Any:
        r = self._one("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(r["value"]) if r else default

    def set_setting(self, key: str, value: Any) -> None:
        self._exec("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, _dumps(value)))

    # ------------------------------------------------------- neural checkpoints
    def add_checkpoint(self, ck_id: str, status: str, backbone: str, path: str, report: dict, note: str = "") -> None:
        self._exec("INSERT INTO neural_checkpoints(id, created_at, status, backbone, path, report, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (ck_id, time.time(), status, backbone, path, _dumps(report), note))

    def update_checkpoint(self, ck_id: str, **fields: Any) -> None:
        cols = []
        vals: list[Any] = []
        for k, v in fields.items():
            if k not in ("status", "shadow", "note", "report"):
                raise ValueError(k)
            cols.append(f"{k}=?")
            vals.append(_dumps(v) if k in ("shadow", "report") else v)
        self._exec(f"UPDATE neural_checkpoints SET {', '.join(cols)} WHERE id=?", (*vals, ck_id))

    @staticmethod
    def _checkpoint(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["report"] = json.loads(d["report"])
        d["shadow"] = json.loads(d["shadow"])
        return d

    def list_checkpoints(self) -> list[dict]:
        return [self._checkpoint(r) for r in self._all("SELECT * FROM neural_checkpoints ORDER BY created_at DESC")]

    def get_checkpoint(self, ck_id: str) -> dict | None:
        r = self._one("SELECT * FROM neural_checkpoints WHERE id=?", (ck_id,))
        return self._checkpoint(r) if r else None

    def checkpoint_with_status(self, status: str) -> dict | None:
        r = self._one("SELECT * FROM neural_checkpoints WHERE status=? ORDER BY created_at DESC LIMIT 1", (status,))
        return self._checkpoint(r) if r else None

    # ---------------------------------------------------------------- datasets
    def create_dataset(self, name: str, source: str, columns: list[str], rows: list[dict], meta: dict | None = None) -> str:
        ds_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.conn.execute(
                "INSERT INTO datasets(id, name, source, columns, meta, n_rows, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ds_id, name, source, _dumps(columns), _dumps(meta or {}), len(rows), time.time()),
            )
            self.conn.executemany(
                "INSERT INTO dataset_rows(dataset_id, idx, data) VALUES (?, ?, ?)",
                ((ds_id, i, _dumps(r)) for i, r in enumerate(rows)),
            )
            self.conn.commit()
        return ds_id

    @staticmethod
    def _dataset(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["columns"] = json.loads(d["columns"])
        d["meta"] = json.loads(d["meta"])
        return d

    def list_datasets(self) -> list[dict]:
        return [self._dataset(r) for r in self._all("SELECT * FROM datasets ORDER BY created_at DESC")]

    def get_dataset(self, ds_id: str) -> dict | None:
        r = self._one("SELECT * FROM datasets WHERE id=?", (ds_id,))
        return self._dataset(r) if r else None

    def dataset_rows(self, ds_id: str, limit: int | None = None, offset: int = 0) -> list[dict]:
        sql = "SELECT data FROM dataset_rows WHERE dataset_id=? ORDER BY idx"
        params: tuple = (ds_id,)
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = (ds_id, limit, offset)
        return [json.loads(r["data"]) for r in self._all(sql, params)]

    def delete_dataset(self, ds_id: str) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM datasets WHERE id=?", (ds_id,))
            self.conn.execute("DELETE FROM dataset_rows WHERE dataset_id=?", (ds_id,))
            self.conn.commit()
