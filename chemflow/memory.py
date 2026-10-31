"""
Persistent research memory.

Everything the agent does is stored in a SQLite database inside the session
directory: hypotheses, experiments (including failed and blocked ones),
findings, acquired data, approval requests and an append-only event log.
Because the memory lives on disk, research can stop and resume in a later
process without losing progress.
"""
import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started REAL NOT NULL,
    ended REAL,
    cycles INTEGER DEFAULT 0,
    stop_reason TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    cycle INTEGER,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    statement TEXT NOT NULL,
    params TEXT NOT NULL,
    origin TEXT NOT NULL,
    status TEXT NOT NULL,
    prior REAL NOT NULL,
    created_cycle INTEGER NOT NULL,
    updated_cycle INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT,
    kind TEXT NOT NULL,
    spec TEXT NOT NULL,
    status TEXT NOT NULL,
    seed INTEGER,
    cycle INTEGER,
    expected TEXT,
    result TEXT,
    error TEXT,
    started REAL,
    finished REAL
);
CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    claim TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL,
    evidence TEXT NOT NULL,
    created_cycle INTEGER NOT NULL,
    updated_cycle INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS datapoints (
    smiles TEXT PRIMARY KEY,
    value REAL NOT NULL,
    source TEXT NOT NULL,
    experiment_id TEXT,
    cycle INTEGER
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    description TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    requested REAL NOT NULL,
    decided REAL,
    decided_by TEXT,
    note TEXT
);
"""

_JSON_COLUMNS = {"params", "spec", "expected", "result", "evidence", "payload"}


def new_id(prefix: str) -> str:
    return "%s-%s" % (prefix, uuid.uuid4().hex[:8])


class _Rows(list):

    def fetchone(self) -> Optional[sqlite3.Row]:
        return self[0] if self else None


class _LockedConnection:
    """SQLite connection shared safely between the agent and the worker
    thread the sandbox runs experiments in."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

    def execute(self, sql: str, args: Any = ()) -> _Rows:
        with self._lock:
            return _Rows(self._conn.execute(sql, args).fetchall())

    def executemany(self, sql: str, rows: Any) -> None:
        with self._lock:
            self._conn.executemany(sql, rows)

    def executescript(self, sql: str) -> None:
        with self._lock:
            self._conn.executescript(sql)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class ResearchMemory:
    """SQLite-backed store of everything a research session has done."""

    def __init__(self, path: str):
        self.path = path
        self.conn = _LockedConnection(path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        d = dict(row)
        for key in _JSON_COLUMNS & d.keys():
            if d[key] is not None:
                d[key] = json.loads(d[key])
        return d

    def _rows(self, sql: str, args: tuple = ()) -> List[Dict[str, Any]]:
        return [self._row(r) for r in self.conn.execute(sql, args)]

    # -- key/value state -------------------------------------------------
    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM state WHERE key=?",
                                (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def set_state(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))
        self.conn.commit()

    @property
    def cycle(self) -> int:
        return self.get_state("cycle", 0)

    # -- runs and events -------------------------------------------------
    def start_run(self) -> str:
        run_id = new_id("run")
        self.conn.execute("INSERT INTO runs(id, started) VALUES(?, ?)",
                          (run_id, time.time()))
        self.conn.commit()
        return run_id

    def end_run(self, run_id: str, cycles: int, stop_reason: str) -> None:
        self.conn.execute(
            "UPDATE runs SET ended=?, cycles=?, stop_reason=? WHERE id=?",
            (time.time(), cycles, stop_reason, run_id))
        self.conn.commit()

    def runs(self) -> List[Dict[str, Any]]:
        return self._rows("SELECT * FROM runs ORDER BY started")

    def log(self, event: str, **payload: Any) -> None:
        self.conn.execute(
            "INSERT INTO events(ts, cycle, kind, payload) VALUES(?, ?, ?, ?)",
            (time.time(), self.cycle, event, json.dumps(payload, default=str)))
        self.conn.commit()

    def events(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        if kind is None:
            return self._rows("SELECT * FROM events ORDER BY id")
        return self._rows("SELECT * FROM events WHERE kind=? ORDER BY id",
                          (kind,))

    # -- hypotheses ------------------------------------------------------
    def add_hypothesis(self, kind: str, statement: str, params: Dict[str, Any],
                       origin: str, prior: float) -> str:
        hid = new_id("H")
        self.conn.execute(
            "INSERT INTO hypotheses VALUES(?, ?, ?, ?, ?, 'proposed', ?, ?, ?)",
            (hid, kind, statement, json.dumps(params), origin, prior,
             self.cycle, self.cycle))
        self.conn.commit()
        self.log("hypothesis_generated",
                 id=hid,
                 kind=kind,
                 statement=statement,
                 origin=origin)
        return hid

    def set_hypothesis_status(self, hid: str, status: str) -> None:
        self.conn.execute(
            "UPDATE hypotheses SET status=?, updated_cycle=? WHERE id=?",
            (status, self.cycle, hid))
        self.conn.commit()

    def hypothesis(self, hid: str) -> Optional[Dict[str, Any]]:
        return self._row(
            self.conn.execute("SELECT * FROM hypotheses WHERE id=?",
                              (hid,)).fetchone())

    def hypotheses(self,
                   status: Optional[str] = None,
                   kind: Optional[str] = None) -> List[Dict[str, Any]]:
        sql, args = "SELECT * FROM hypotheses WHERE 1=1", []
        if status is not None:
            sql += " AND status=?"
            args.append(status)
        if kind is not None:
            sql += " AND kind=?"
            args.append(kind)
        return self._rows(sql + " ORDER BY rowid", tuple(args))

    def has_hypothesis(self, kind: str, params: Dict[str, Any]) -> bool:
        key = json.dumps(params)
        row = self.conn.execute(
            "SELECT 1 FROM hypotheses WHERE kind=? AND params=?",
            (kind, key)).fetchone()
        return row is not None

    # -- experiments -----------------------------------------------------
    def add_experiment(self, hypothesis_id: Optional[str], kind: str,
                       spec: Dict[str, Any], seed: int,
                       expected: Optional[Dict[str, Any]] = None) -> str:
        eid = new_id("E")
        self.conn.execute(
            "INSERT INTO experiments(id, hypothesis_id, kind, spec, status, "
            "seed, cycle, expected) VALUES(?, ?, ?, ?, 'planned', ?, ?, ?)",
            (eid, hypothesis_id, kind, json.dumps(spec), seed, self.cycle,
             json.dumps(expected)))
        self.conn.commit()
        return eid

    def update_experiment(self, eid: str, **fields: Any) -> None:
        cols, args = [], []
        for key, value in fields.items():
            cols.append("%s=?" % key)
            args.append(
                json.dumps(value, default=float)
                if key in _JSON_COLUMNS else value)
        args.append(eid)
        self.conn.execute(
            "UPDATE experiments SET %s WHERE id=?" % ", ".join(cols), args)
        self.conn.commit()

    def experiment(self, eid: str) -> Optional[Dict[str, Any]]:
        return self._row(
            self.conn.execute("SELECT * FROM experiments WHERE id=?",
                              (eid,)).fetchone())

    def experiments(self,
                    hypothesis_id: Optional[str] = None,
                    kind: Optional[str] = None,
                    status: Optional[str] = None) -> List[Dict[str, Any]]:
        sql, args = "SELECT * FROM experiments WHERE 1=1", []
        for col, val in (("hypothesis_id", hypothesis_id), ("kind", kind),
                         ("status", status)):
            if val is not None:
                sql += " AND %s=?" % col
                args.append(val)
        return self._rows(sql + " ORDER BY rowid", tuple(args))

    # -- findings --------------------------------------------------------
    def upsert_finding(self, hypothesis_id: str, claim: str, status: str,
                       confidence: Optional[float],
                       evidence: Dict[str, Any]) -> str:
        row = self.conn.execute(
            "SELECT id FROM findings WHERE hypothesis_id=?",
            (hypothesis_id,)).fetchone()
        if row is None:
            fid = new_id("F")
            self.conn.execute(
                "INSERT INTO findings VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (fid, hypothesis_id, claim, status, confidence,
                 json.dumps(evidence, default=float), self.cycle, self.cycle))
        else:
            fid = row["id"]
            self.conn.execute(
                "UPDATE findings SET claim=?, status=?, confidence=?, "
                "evidence=?, updated_cycle=? WHERE id=?",
                (claim, status, confidence, json.dumps(evidence,
                                                       default=float),
                 self.cycle, fid))
        self.conn.commit()
        self.log("finding_updated",
                 id=fid,
                 hypothesis_id=hypothesis_id,
                 status=status,
                 claim=claim)
        return fid

    def findings(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status is None:
            return self._rows("SELECT * FROM findings ORDER BY rowid")
        return self._rows(
            "SELECT * FROM findings WHERE status=? ORDER BY rowid", (status,))

    # -- acquired data ---------------------------------------------------
    def add_datapoints(self, smiles: List[str], values: List[float],
                       source: str, experiment_id: Optional[str]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO datapoints VALUES(?, ?, ?, ?, ?)",
            [(s, float(v), source, experiment_id, self.cycle)
             for s, v in zip(smiles, values)])
        self.conn.commit()

    def datapoints(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        if source is None:
            return self._rows("SELECT * FROM datapoints ORDER BY rowid")
        return self._rows(
            "SELECT * FROM datapoints WHERE source=? ORDER BY rowid",
            (source,))

    # -- human approvals -------------------------------------------------
    def request_approval(self, kind: str, description: str,
                         payload: Dict[str, Any]) -> str:
        aid = new_id("A")
        self.conn.execute(
            "INSERT INTO approvals(id, kind, description, payload, status, "
            "requested) VALUES(?, ?, ?, ?, 'pending', ?)",
            (aid, kind, description, json.dumps(payload,
                                                default=float), time.time()))
        self.conn.commit()
        self.log("approval_requested", id=aid, kind=kind)
        return aid

    def decide_approval(self, aid: str, approved: bool, decided_by: str,
                        note: str = "") -> None:
        row = self.approval(aid)
        if row is None:
            raise KeyError("No approval request %s" % aid)
        if row["status"] != "pending":
            raise ValueError("Approval %s is already %s" % (aid,
                                                             row["status"]))
        status = "approved" if approved else "rejected"
        self.conn.execute(
            "UPDATE approvals SET status=?, decided=?, decided_by=?, note=? "
            "WHERE id=?", (status, time.time(), decided_by, note, aid))
        self.conn.commit()
        self.log("approval_decided", id=aid, status=status, by=decided_by)

    def complete_approval(self, aid: str) -> None:
        self.conn.execute("UPDATE approvals SET status='completed' WHERE id=?",
                          (aid,))
        self.conn.commit()

    def approval(self, aid: str) -> Optional[Dict[str, Any]]:
        return self._row(
            self.conn.execute("SELECT * FROM approvals WHERE id=?",
                              (aid,)).fetchone())

    def approvals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status is None:
            return self._rows("SELECT * FROM approvals ORDER BY requested")
        return self._rows(
            "SELECT * FROM approvals WHERE status=? ORDER BY requested",
            (status,))
