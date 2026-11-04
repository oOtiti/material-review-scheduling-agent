import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, path):
        self.path = Path(path)
        with self.transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS drafts (request_id TEXT PRIMARY KEY, body TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS preferences (profile TEXT PRIMARY KEY, latest_first INTEGER NOT NULL)"
            )

    def remember_preference(self, profile, *, latest_first):
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO preferences VALUES (?, ?) "
                "ON CONFLICT(profile) DO UPDATE SET latest_first = excluded.latest_first",
                (profile, int(latest_first)),
            )

    def preference(self, profile):
        with self.transaction() as conn:
            row = conn.execute("SELECT latest_first FROM preferences WHERE profile = ?", (profile,)).fetchone()
        return {"latest_first": bool(row[0])} if row is not None else {}

    @contextmanager
    def transaction(self):
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def session(self, conn, request_id):
        row = conn.execute("SELECT body FROM sessions WHERE id = ?", (request_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, conn, session):
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET body = excluded.body",
            (session["request"]["request_id"], json.dumps(session, ensure_ascii=False)),
        )

    def draft(self, conn, request_id):
        row = conn.execute("SELECT body FROM drafts WHERE request_id = ?", (request_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def insert_draft(self, conn, request_id, draft):
        conn.execute("INSERT INTO drafts VALUES (?, ?)", (request_id, json.dumps(draft, ensure_ascii=False)))

    def count_drafts(self):
        with self.transaction() as conn:
            return conn.execute("SELECT count(*) FROM drafts").fetchone()[0]
