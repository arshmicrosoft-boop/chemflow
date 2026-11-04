import sqlite3
import os
DB_PATH = os.getenv("CHEMFLOW_DB", "chemflow.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        objective TEXT,
        dataset TEXT,
        session_dir TEXT,
        created_at REAL DEFAULT CURRENT_TIMESTAMP,
        active INTEGER DEFAULT 1
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS topic_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT,
        report TEXT,
        sources TEXT,
        searches INTEGER,
        search_errors TEXT,
        model TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")
    conn.commit()
    conn.close()
