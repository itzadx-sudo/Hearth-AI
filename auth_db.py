import sqlite3
import os
import hashlib
import sys
from paths import _data_path

DB_PATH = _data_path('user.db')

def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

# sets up user + guardian tables, runs on every startup
def init_db():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'guardian'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS guardian_patients (
            guardian_username TEXT NOT NULL,
            patient_id INTEGER NOT NULL,
            PRIMARY KEY (guardian_username, patient_id),
            FOREIGN KEY (guardian_username) REFERENCES users(username) ON DELETE CASCADE
        )
    """)
    conn.commit()
    # migration: add display_name if it doesn't exist yet
    try:
        cur.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

# sha256 is fine for a local demo app, don't use this in prod
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def create_user(username: str, password: str, role: str = 'guardian') -> bool:
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, _hash_password(password), role)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        print(f"[ERROR] create_user: {e}", file=sys.stderr)
        return False

def verify_user(username: str, password: str) -> dict:
    try:
        conn = _connect()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()
        
        if row and row['password_hash'] == _hash_password(password):
            disp = row['display_name'] if 'display_name' in row.keys() else None
            return {"username": row['username'], "role": row['role'], "display_name": disp}
    except Exception as e:
        print(f"[ERROR] verify_user: {e}", file=sys.stderr)
    return None

def assign_patient(guardian_username: str, patient_id: int) -> bool:
    try:
        conn = _connect()
        cur = conn.cursor()
        # Verify guardian exists and is a guardian
        cur.execute("SELECT role FROM users WHERE username = ?", (guardian_username,))
        row = cur.fetchone()
        if not row or row[0] != 'guardian':
            conn.close()
            return False
            
        cur.execute(
            "INSERT OR IGNORE INTO guardian_patients (guardian_username, patient_id) VALUES (?, ?)",
            (guardian_username, patient_id)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] assign_patient: {e}", file=sys.stderr)
        return False

def get_guardian_patients(guardian_username: str) -> list:
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("SELECT patient_id FROM guardian_patients WHERE guardian_username = ?", (guardian_username,))
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception as e:
        print(f"[ERROR] get_guardian_patients: {e}", file=sys.stderr)
        return []

def get_all_guardians() -> list:
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE role = 'guardian'")
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception as e:
        print(f"[ERROR] get_all_guardians: {e}", file=sys.stderr)
        return []

def update_display_name(username: str, display_name: str) -> bool:
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("UPDATE users SET display_name = ? WHERE username = ?", (display_name, username))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] update_display_name: {e}", file=sys.stderr)
        return False

