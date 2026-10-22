"""Private platform persistence. User identity is server-issued, not a Reviewer ID."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import secrets
import sqlite3
import time
from pathlib import Path


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def new_id() -> str:
    return secrets.token_hex(16)


class PlatformStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "platform.sqlite3"
        with self.db() as con:
            con.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users (
                  id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL,
                  code_hash TEXT UNIQUE NOT NULL, disabled INTEGER NOT NULL DEFAULT 0,
                  created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                  token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                  csrf TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS projects (
                  id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id),
                  name TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS shares (
                  project_id TEXT NOT NULL REFERENCES projects(id),
                  expert_id TEXT NOT NULL REFERENCES users(id), active INTEGER NOT NULL,
                  updated REAL NOT NULL, PRIMARY KEY(project_id, expert_id));
                CREATE TABLE IF NOT EXISTS samples (
                  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                  name TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                  format TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
                  created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY, sample_id TEXT NOT NULL REFERENCES samples(id),
                  task TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
                  started REAL, finished REAL, error TEXT, cancel_requested INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS feedback (
                  id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                  user_id TEXT NOT NULL REFERENCES users(id), revision INTEGER NOT NULL,
                  action TEXT NOT NULL, note TEXT NOT NULL, geometry TEXT NOT NULL,
                  prediction_seen INTEGER NOT NULL, request_id TEXT NOT NULL,
                  created REAL NOT NULL, UNIQUE(job_id,user_id,request_id),
                  UNIQUE(job_id,user_id,revision));
                CREATE TABLE IF NOT EXISTS research_requests (
                  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                  user_id TEXT NOT NULL REFERENCES users(id), reason TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'requested', created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS project_owner ON projects(owner_id);
                CREATE INDEX IF NOT EXISTS sample_project ON samples(project_id);
                CREATE INDEX IF NOT EXISTS job_sample ON jobs(sample_id);
                CREATE INDEX IF NOT EXISTS feedback_job ON feedback(job_id,user_id,revision);
            """)
        self.path.chmod(0o600)

    @contextmanager
    def db(self):
        con = sqlite3.connect(self.path, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            with con:
                yield con
        finally:
            con.close()

    def all(self, sql, args=()):
        with self.db() as con:
            return [dict(r) for r in con.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        rows = self.all(sql, args)
        return rows[0] if rows else None

    def execute(self, sql, args=()):
        with self.db() as con:
            con.execute(sql, args)

    def create_user(self, name: str, role="researcher"):
        if role not in {"researcher", "expert", "admin"} or not name.strip():
            raise ValueError("Invalid user")
        uid = new_id()
        code = f"PEM.{uid}.{secrets.token_urlsafe(32)}"
        self.execute("INSERT INTO users(id,name,role,code_hash,created) VALUES(?,?,?,?,?)",
                     (uid, name.strip(), role, digest(code), time.time()))
        return {"id": uid, "name": name.strip(), "role": role, "activation_code": code}

    def authenticate(self, code: str):
        return self.one("SELECT id,name,role FROM users WHERE code_hash=? AND disabled=0",
                        (digest(code),))

    def renew_code(self, uid: str):
        """Rotate credentials without replacing the identity or its projects/reviews."""
        with self.db() as con:
            user = con.execute("SELECT id,name,role FROM users WHERE id=?", (uid,)).fetchone()
            if not user:
                raise ValueError("Unknown user")
            code = f"PEM.{uid}.{secrets.token_urlsafe(32)}"
            con.execute("UPDATE users SET code_hash=?,disabled=0 WHERE id=?", (digest(code), uid))
            con.execute("UPDATE sessions SET expires=0 WHERE user_id=?", (uid,))
        return {**dict(user), "activation_code": code}

    def session(self, token: str):
        return self.one("""SELECT u.id,u.name,u.role,s.csrf FROM sessions s JOIN users u
          ON u.id=s.user_id WHERE s.token_hash=? AND s.expires>? AND u.disabled=0""",
                        (digest(token), time.time()))

    def project(self, pid: str, uid: str, *, owner=False):
        if owner:
            return self.one("SELECT * FROM projects WHERE id=? AND owner_id=?", (pid, uid))
        return self.one("""SELECT p.* FROM projects p WHERE p.id=? AND
           (p.owner_id=? OR EXISTS (SELECT 1 FROM shares s WHERE s.project_id=p.id
           AND s.expert_id=? AND s.active=1))""", (pid, uid, uid))

    def projects(self, uid: str):
        return self.all("""SELECT p.*,u.name AS owner_name FROM projects p JOIN users u
          ON p.owner_id=u.id WHERE p.owner_id=? OR EXISTS (SELECT 1 FROM shares s
          WHERE s.project_id=p.id AND s.expert_id=? AND s.active=1) ORDER BY p.created DESC""",
                        (uid, uid))

    def sample_path(self, sid: str) -> Path:
        return self._path("samples", sid)

    def job_path(self, jid: str) -> Path:
        return self._path("jobs", jid)

    def _path(self, kind: str, key: str):
        if len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid identifier")
        path = self.root / kind / key
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Path outside platform storage")
        return path

    def claim_job(self):
        with self.db() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM jobs WHERE status='queued' AND cancel_requested=0 ORDER BY created LIMIT 1").fetchone()
            if not row:
                return None
            con.execute("UPDATE jobs SET status='running',started=? WHERE id=?", (time.time(), row["id"]))
            return dict(row)
