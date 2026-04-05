
import json
import os
import queue
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Optional
import sys
from paths import _data_path
from config import (HR_CHANGE_THRESHOLD, SPO2_CHANGE_THRESHOLD,
                    TEMP_CHANGE_THRESHOLD, SYS_BP_CHANGE_THRESHOLD)

SENSOR_DB_PATH = os.environ.get('HEARTH_SENSOR_DB', _data_path('hearth_sensor.db'))
RESULTS_DB_PATH = _data_path('hearth_results.db')

_sensor_lock    = threading.Lock()
_results_lock   = threading.Lock()
_tables_ensured = False

# all DB writes go through this queue so we don't block the hot path
_write_queue: queue.Queue = queue.Queue(maxsize=10_000)


def _results_writer():
    while True:
        fn = _write_queue.get()
        if fn is None:
            _write_queue.task_done()
            break

        batch = [fn]
        try:
            while True:
                item = _write_queue.get_nowait()
                batch.append(item)
        except queue.Empty:
            pass

        sentinel_found = False
        for item in batch:
            if item is None:
                sentinel_found = True
                _write_queue.task_done()
                continue
            try:
                item()
            except Exception as e:
                print(f"[DATA LAYER] Write error: {e}")
            finally:
                _write_queue.task_done()

        if sentinel_found:
            break


_writer_thread = threading.Thread(
    target=_results_writer, daemon=True, name='dl-writer')
_writer_thread.start()


@contextmanager
def _db_conn(path, row_factory=None):
    conn = sqlite3.connect(path)
    if row_factory:
        conn.row_factory = row_factory
    try:
        yield conn
    finally:
        conn.close()

def ensure_sensor_db(path: str):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   INTEGER NOT NULL,
                timestamp    TEXT    NOT NULL,
                sim_date     TEXT    NOT NULL,
                heart_rate   REAL,
                systolic_bp  REAL,
                diastolic_bp REAL,
                body_temp    REAL,
                spo2         REAL,
                activity     INTEGER
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensor_sim_date "
            "ON sensor_data(sim_date)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensor_patient_id "
            "ON sensor_data(patient_id)")
        conn.commit()
    finally:
        conn.close()


def _ensure_tables():
    global _tables_ensured
    if _tables_ensured:
        return
    with _results_lock:
        if _tables_ensured:
            return
        _do_ensure_tables()
        _tables_ensured = True


def _do_ensure_tables():
    sc = sqlite3.connect(SENSOR_DB_PATH)
    try:
        sc.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   INTEGER NOT NULL,
                timestamp    TEXT    NOT NULL,
                sim_date     TEXT    NOT NULL,
                heart_rate   REAL,
                systolic_bp  REAL,
                diastolic_bp REAL,
                body_temp    REAL,
                spo2         REAL,
                activity     INTEGER
            )
        """)
        sc.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensor_sim_date ON sensor_data(sim_date)")
        sc.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensor_patient_id ON sensor_data(patient_id)")
        sc.commit()
    finally:
        sc.close()

    rc = sqlite3.connect(RESULTS_DB_PATH)
    try:
        rc.execute("PRAGMA journal_mode=WAL")

        rc.execute("""
            CREATE TABLE IF NOT EXISTS daily_summaries (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id        INTEGER NOT NULL,
                sim_date          TEXT    NOT NULL,
                avg_heart_rate    REAL,
                avg_systolic      REAL,
                avg_diastolic     REAL,
                avg_temp          REAL,
                avg_spo2          REAL,
                dominant_activity INTEGER,
                status            TEXT,
                confidence        REAL,
                max_heart_rate    REAL,
                max_systolic      REAL,
                max_temp          REAL,
                min_spo2          REAL,
                activity_ratio    REAL,
                worst_status      TEXT,
                critical_count    INTEGER DEFAULT 0,
                total_readings    INTEGER DEFAULT 0,
                avg_confidence    REAL,
                UNIQUE(patient_id, sim_date)
            )
        """)
        rc.execute(
            "CREATE INDEX IF NOT EXISTS idx_summary_date_patient "
            "ON daily_summaries(sim_date, patient_id)")

        rc.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                patient_id  INTEGER NOT NULL,
                sim_date    TEXT,
                risk_label  TEXT    NOT NULL,
                risk_score  REAL    NOT NULL,
                top_factors TEXT,
                window_days INTEGER DEFAULT 7
            )
        """)
        rc.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_patient ON predictions(patient_id)")
        rc.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_patient_ts ON predictions(patient_id, timestamp DESC)")

        try:
            rc.execute(
                "ALTER TABLE daily_summaries ADD COLUMN low_confidence_count INTEGER DEFAULT 0"
            )
        except Exception:
            pass

        rc.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                patient_id  INTEGER NOT NULL,
                alert_type  TEXT    NOT NULL,
                severity    TEXT,
                details     TEXT
            )
        """)
        rc.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_patient "
            "ON alerts(patient_id, timestamp DESC)")

        rc.commit()
    finally:
        rc.close()


def insert_sensor_batch(rows, conn: Optional[sqlite3.Connection] = None):
    _ensure_tables()
    _own_conn = conn is None

    _INSERT_SQL = (
        "INSERT INTO sensor_data "
        "(patient_id, timestamp, sim_date, heart_rate, systolic_bp, "
        "diastolic_bp, body_temp, spo2, activity) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    _rows = [
        (
            r['patient_id'], r['timestamp'], r['timestamp'][:10],
            r.get('heart_rate'), r.get('systolic_bp'), r.get('diastolic_bp'),
            r.get('body_temp'), r.get('spo2'), r.get('activity'),
        )
        for r in rows
    ]

    if _own_conn:
        with _sensor_lock:
            conn = sqlite3.connect(SENSOR_DB_PATH)
            try:
                conn.executemany(_INSERT_SQL, _rows)
                conn.commit()
            finally:
                conn.close()
    else:
        conn.executemany(_INSERT_SQL, _rows)


def get_dates_available():
    _ensure_tables()
    with _db_conn(SENSOR_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT sim_date FROM sensor_data ORDER BY sim_date ASC"
        ).fetchall()
    return [r[0] for r in rows]


def get_readings_for_date(sim_date, chunk_size=1000):
    _ensure_tables()
    conn = sqlite3.connect(SENSOR_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT patient_id, timestamp, heart_rate, systolic_bp, diastolic_bp, "
            "body_temp, spo2, activity "
            "FROM sensor_data WHERE sim_date = ?", (sim_date,)
        )
        results = []
        while True:
            batch = cursor.fetchmany(chunk_size)
            if not batch:
                break
            results.extend(dict(r) for r in batch)
        return results
    finally:
        conn.close()


def get_last_reading_per_patient_for_date(sim_date):
    _ensure_tables()
    conn = sqlite3.connect(SENSOR_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT patient_id, timestamp, heart_rate, systolic_bp, diastolic_bp, "
            "body_temp, spo2, activity "
            "FROM sensor_data "
            "WHERE sim_date = ? "
            "  AND rowid IN ("
            "      SELECT MAX(rowid) FROM sensor_data "
            "      WHERE sim_date = ? GROUP BY patient_id"
            ")",
            (sim_date, sim_date),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def store_daily_summary(patient_id, sim_date, aggregates, status, confidence):
    _ensure_tables()
    params = (
        patient_id, sim_date,
        aggregates.get('heart_rate'), aggregates.get('systolic_bp'),
        aggregates.get('diastolic_bp'), aggregates.get('body_temp'),
        aggregates.get('spo2'), aggregates.get('activity'),
        status, confidence,
        aggregates.get('max_heart_rate'), aggregates.get('max_systolic'),
        aggregates.get('max_temp'), aggregates.get('min_spo2'),
        aggregates.get('activity_ratio'), aggregates.get('worst_status'),
        aggregates.get('critical_count', 0), aggregates.get('total_readings', 0),
        aggregates.get('avg_confidence'), aggregates.get('low_confidence_count', 0),
    )
    def _write():
        with _db_conn(RESULTS_DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_summaries
                    (patient_id, sim_date,
                     avg_heart_rate, avg_systolic, avg_diastolic,
                     avg_temp, avg_spo2, dominant_activity,
                     status, confidence,
                     max_heart_rate, max_systolic, max_temp, min_spo2,
                     activity_ratio, worst_status,
                     critical_count, total_readings, avg_confidence,
                     low_confidence_count)
                VALUES (?, ?,  ?, ?, ?,  ?, ?, ?,  ?, ?,  ?, ?, ?, ?,  ?, ?,  ?, ?, ?, ?)
            """, params)
            conn.commit()
    _write_queue.put(_write)


def get_rolling_window(patient_id, days=7, before_date=None):
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        if before_date is not None:
            rows = conn.execute("""
                SELECT * FROM daily_summaries
                WHERE patient_id = ? AND sim_date < ?
                ORDER BY sim_date DESC LIMIT ?
            """, (patient_id, before_date, days)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM daily_summaries
                WHERE patient_id = ?
                ORDER BY sim_date DESC LIMIT ?
            """, (patient_id, days)).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_all_patient_ids_in_db2():
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT patient_id FROM daily_summaries").fetchall()
    return [r[0] for r in rows]


def get_daily_summary(patient_id, sim_date):
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        row = conn.execute("""
            SELECT * FROM daily_summaries
            WHERE patient_id = ? AND sim_date = ?
        """, (patient_id, sim_date)).fetchone()
    return dict(row) if row else None


def get_patient_history(patient_id, days=30):
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        rows = conn.execute("""
            SELECT * FROM daily_summaries
            WHERE patient_id = ?
            ORDER BY sim_date DESC LIMIT ?
        """, (patient_id, days)).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_all_patients_latest():
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        rows = conn.execute("""
            SELECT ds.* FROM daily_summaries ds
            INNER JOIN (
                SELECT patient_id, MAX(sim_date) AS max_date
                FROM daily_summaries GROUP BY patient_id
            ) latest ON ds.patient_id = latest.patient_id
                     AND ds.sim_date = latest.max_date
            ORDER BY ds.patient_id
        """).fetchall()
    return [dict(r) for r in rows]


def get_day_overview(sim_date):
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH) as conn:
        row = conn.execute("""
            SELECT
                COUNT(DISTINCT patient_id) AS total_patients,
                SUM(CASE WHEN worst_status = 'Healthy' THEN 1 ELSE 0 END) AS healthy_count,
                SUM(CASE WHEN worst_status = 'Unhealthy' THEN 1 ELSE 0 END) AS unhealthy_count,
                SUM(CASE WHEN worst_status = 'Critical' THEN 1 ELSE 0 END) AS critical_count
            FROM daily_summaries WHERE sim_date = ?
        """, (sim_date,)).fetchone()

    if not row or row[0] == 0:
        return None
    return {
        'sim_date': sim_date,
        'total_patients': row[0],
        'healthy_count': row[1] or 0,
        'unhealthy_count': row[2] or 0,
        'critical_count': row[3] or 0,
    }


def get_low_confidence_patients(sim_date):
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        rows = conn.execute("""
            SELECT * FROM daily_summaries
            WHERE sim_date = ? AND low_confidence_count > 0
            ORDER BY low_confidence_count DESC
        """, (sim_date,)).fetchall()
    return [dict(r) for r in rows]


def get_critical_timeline(patient_id, days=30):
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        rows = conn.execute("""
            SELECT sim_date, critical_count, total_readings
            FROM daily_summaries
            WHERE patient_id = ?
            ORDER BY sim_date DESC LIMIT ?
        """, (patient_id, days)).fetchall()
    return [dict(r) for r in reversed(rows)]


def store_prediction(patient_id, sim_date, risk_label, risk_score, top_factors=None):
    _ensure_tables()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    factors_str = json.dumps(top_factors) if top_factors else None

    params = (ts, patient_id, sim_date, risk_label, risk_score, factors_str)
    def _write():
        with _db_conn(RESULTS_DB_PATH) as conn:
            conn.execute("""
                INSERT INTO predictions
                    (timestamp, patient_id, sim_date, risk_label, risk_score, top_factors)
                VALUES (?, ?, ?, ?, ?, ?)
            """, params)
            conn.commit()
    _write_queue.put(_write)


def get_predictions_for_patient(patient_id, limit=10):
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE patient_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (patient_id, limit)).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        if d.get('top_factors'):
            try:
                d['top_factors'] = json.loads(d['top_factors'])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


def get_high_risk_patients():
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        rows = conn.execute("""
            SELECT p.* FROM predictions p
            INNER JOIN (
                SELECT patient_id, MAX(timestamp) AS max_ts
                FROM predictions GROUP BY patient_id
            ) latest ON p.patient_id = latest.patient_id
                     AND p.timestamp = latest.max_ts
            WHERE p.risk_label = 'HIGH RISK'
            ORDER BY p.risk_score DESC
        """).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        if d.get('top_factors'):
            try:
                d['top_factors'] = json.loads(d['top_factors'])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


def get_all_latest_predictions(limit=None):
    _ensure_tables()
    query = """
        SELECT p.* FROM predictions p
        INNER JOIN (
            SELECT patient_id, MAX(timestamp) AS max_ts
            FROM predictions GROUP BY patient_id
        ) latest ON p.patient_id = latest.patient_id
                 AND p.timestamp = latest.max_ts
        ORDER BY p.risk_score DESC
    """
    params = []
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        rows = conn.execute(query, params).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        if d.get('top_factors'):
            try:
                d['top_factors'] = json.loads(d['top_factors'])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


def get_sudden_changes(sim_date):
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        rows = conn.execute("""
            SELECT
                t.patient_id,
                t.avg_heart_rate  AS cur_hr,  y.avg_heart_rate  AS prev_hr,
                t.avg_spo2        AS cur_spo2, y.avg_spo2        AS prev_spo2,
                t.avg_temp        AS cur_temp, y.avg_temp        AS prev_temp,
                t.avg_systolic    AS cur_sys,  y.avg_systolic    AS prev_sys
            FROM daily_summaries t
            JOIN daily_summaries y ON t.patient_id = y.patient_id
            WHERE t.sim_date = ?
              AND y.sim_date = (
                  SELECT MAX(sim_date) FROM daily_summaries WHERE sim_date < ?
              )
              AND (
                  (t.avg_heart_rate IS NOT NULL AND y.avg_heart_rate IS NOT NULL
                   AND ABS(t.avg_heart_rate - y.avg_heart_rate) >= ?)
               OR (t.avg_spo2 IS NOT NULL AND y.avg_spo2 IS NOT NULL
                   AND (t.avg_spo2 - y.avg_spo2) <= -?)
               OR (t.avg_temp IS NOT NULL AND y.avg_temp IS NOT NULL
                   AND ABS(t.avg_temp - y.avg_temp) >= ?)
               OR (t.avg_systolic IS NOT NULL AND y.avg_systolic IS NOT NULL
                   AND ABS(t.avg_systolic - y.avg_systolic) >= ?)
              )
        """, (sim_date, sim_date,
              HR_CHANGE_THRESHOLD, SPO2_CHANGE_THRESHOLD,
              TEMP_CHANGE_THRESHOLD, SYS_BP_CHANGE_THRESHOLD)).fetchall()

    vital_checks = [
        ('heart_rate', 'cur_hr',   'prev_hr',   HR_CHANGE_THRESHOLD,   'both'),
        ('spo2',       'cur_spo2', 'prev_spo2', SPO2_CHANGE_THRESHOLD,  'decrease'),
        ('temp',       'cur_temp', 'prev_temp', TEMP_CHANGE_THRESHOLD,   'both'),
        ('systolic',   'cur_sys',  'prev_sys',  SYS_BP_CHANGE_THRESHOLD, 'both'),
    ]

    results = []
    for row in rows:
        changes = []
        for vital_name, cur_col, prev_col, threshold, direction in vital_checks:
            curr_val = row[cur_col]
            prev_val = row[prev_col]
            if curr_val is None or prev_val is None:
                continue
            delta = curr_val - prev_val
            if direction == 'both':
                exceeded = abs(delta) >= threshold
            elif direction == 'decrease':
                exceeded = delta <= -threshold
            else:
                exceeded = delta >= threshold
            if exceeded:
                changes.append({
                    'vital': vital_name,
                    'previous': round(prev_val, 2),
                    'current': round(curr_val, 2),
                    'delta': round(delta, 2),
                })
        if changes:
            results.append({'patient_id': row['patient_id'], 'changes': changes})

    results.sort(key=lambda x: len(x['changes']), reverse=True)
    return results


def get_bulk_rolling_windows(patient_ids, days=7, before_date=None):
    _ensure_tables()
    if not patient_ids:
        return {}
    placeholders = ','.join('?' * len(patient_ids))
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        if before_date is not None:
            rows = conn.execute(f"""
                SELECT * FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY patient_id ORDER BY sim_date DESC
                           ) AS _rn
                    FROM daily_summaries
                    WHERE patient_id IN ({placeholders})
                      AND sim_date < ?
                ) sub
                WHERE _rn <= ?
                ORDER BY patient_id ASC, sim_date ASC
            """, (*patient_ids, before_date, days)).fetchall()
        else:
            rows = conn.execute(f"""
                SELECT * FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY patient_id ORDER BY sim_date DESC
                           ) AS _rn
                    FROM daily_summaries
                    WHERE patient_id IN ({placeholders})
                ) sub
                WHERE _rn <= ?
                ORDER BY patient_id ASC, sim_date ASC
            """, (*patient_ids, days)).fetchall()

    result = {}
    for row in rows:
        d = dict(row)
        d.pop('_rn', None)
        pid = d['patient_id']
        if pid not in result:
            result[pid] = []
        result[pid].append(d)
    return result

def store_alert(patient_id, alert_type, severity=None, details=None):
    _ensure_tables()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    details_str = json.dumps(details) if details and not isinstance(details, str) else details

    params = (ts, patient_id, alert_type, severity, details_str)
    def _write():
        with _db_conn(RESULTS_DB_PATH) as conn:
            conn.execute("""
                INSERT INTO alerts (timestamp, patient_id, alert_type, severity, details)
                VALUES (?, ?, ?, ?, ?)
            """, params)
            conn.commit()
    _write_queue.put(_write)


def get_alerts_from_db(limit=100, alert_type=None):
    _ensure_tables()
    with _db_conn(RESULTS_DB_PATH, sqlite3.Row) as conn:
        if alert_type:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE alert_type = ? ORDER BY id DESC LIMIT ?",
                (alert_type, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get('details'):
            try:
                d['details'] = json.loads(d['details'])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


# separate db for live sessions so we don't pollute batch data
LIVE_DB_PATH = _data_path("hearth_live.db")
_live_lock = threading.Lock()
_live_initted = False


def _live_conn():
    conn = sqlite3.connect(LIVE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_live_db():
    global _live_initted
    with _live_lock:
        if _live_initted:
            return
        conn = _live_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS live_tick_results (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT    NOT NULL,
                    tick         INTEGER NOT NULL,
                    tick_time    TEXT    NOT NULL,
                    patient_id   INTEGER NOT NULL,
                    status       TEXT,
                    confidence   REAL,
                    heart_rate   REAL,
                    systolic_bp  REAL,
                    diastolic_bp REAL,
                    body_temp    REAL,
                    spo2         REAL,
                    activity     INTEGER,
                    attention_weights TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_ltr_sess_pid_tick
                    ON live_tick_results(session_id, patient_id, tick);

                CREATE TABLE IF NOT EXISTS live_predictions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id       TEXT    NOT NULL,
                    computed_at_tick INTEGER NOT NULL,
                    computed_at_time TEXT    NOT NULL,
                    patient_id       INTEGER NOT NULL,
                    risk_label       TEXT,
                    risk_score       REAL,
                    top_factors      TEXT,
                    window_ticks     INTEGER DEFAULT 7,
                    UNIQUE(session_id, computed_at_tick, patient_id)
                        ON CONFLICT REPLACE
                );
                CREATE INDEX IF NOT EXISTS idx_lp_sess_pid
                    ON live_predictions(session_id, patient_id);
            """)
            conn.commit()
        finally:
            conn.close()
        _live_initted = True


def store_tick_results(session_id: str, tick: int, tick_time: str,
                       patient_results: list):
    # write all patient results for this tick in one transaction
    rows = [
        (
            session_id, tick, tick_time,
            r["patient_id"], r.get("status"), r.get("confidence"),
            r.get("heart_rate"), r.get("systolic_bp"), r.get("diastolic_bp"),
            r.get("body_temp"), r.get("spo2"), r.get("activity"),
            json.dumps(r.get("attention")) if r.get("attention") else None,
        )
        for r in patient_results
    ]
    conn = _live_conn()
    try:
        conn.executemany(
            "INSERT INTO live_tick_results "
            "(session_id, tick, tick_time, patient_id, status, confidence, "
            "heart_rate, systolic_bp, diastolic_bp, body_temp, spo2, activity, attention_weights) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def get_patient_window(session_id: str, patient_id: int, limit: int = 7) -> list:
    # fetch last N tick results for a patient
    conn = _live_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM live_tick_results "
            "WHERE session_id=? AND patient_id=? "
            "ORDER BY tick DESC LIMIT ?",
            (session_id, patient_id, limit),
        ).fetchall()
    finally:
        conn.close()
    # reverse so it's oldest-first, map status -> worst_status for predict_risk()
    result = []
    for r in reversed(rows):
        d = dict(r)
        d["worst_status"] = d.get("status")
        result.append(d)
    return result


def store_prediction(session_id: str, tick: int, tick_time: str,
                     patient_id: int, result: dict):
    # queue a 7-day risk prediction write
    conn = _live_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO live_predictions "
            "(session_id, computed_at_tick, computed_at_time, patient_id, "
            "risk_label, risk_score, top_factors, window_ticks) "
            "VALUES (?,?,?,?,?,?,?,7)",
            (
                session_id, tick, tick_time, patient_id,
                result.get("risk_label"), result.get("risk_score"),
                json.dumps(result.get("top_factors", [])),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_high_risk_patients(session_id: str, at_tick: int,
                           threshold: float = 0.5) -> list:
    # patients flagged HIGH RISK at a given tick
    conn = _live_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM live_predictions "
            "WHERE session_id=? AND computed_at_tick=? AND risk_score>=? "
            "ORDER BY risk_score DESC",
            (session_id, at_tick, threshold),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_latest_predictions(session_id: str, limit: Optional[int] = None) -> list:
    # most recent risk prediction per patient
    conn = _live_conn()
    try:
        rows = conn.execute(
            """
            SELECT lp.*
            FROM live_predictions lp
            INNER JOIN (
                SELECT patient_id, MAX(computed_at_tick) AS max_tick
                FROM live_predictions WHERE session_id = ?
                GROUP BY patient_id
            ) m ON  lp.patient_id        = m.patient_id
                AND lp.computed_at_tick  = m.max_tick
                AND lp.session_id        = ?
            ORDER BY lp.risk_score DESC
            """,
            (session_id, session_id),
        ).fetchall()
    finally:
        conn.close()
    results = [dict(r) for r in rows]
    return results[:limit] if limit is not None else results


def get_latest_patient_states(session_id: str) -> list:
    # most recent tick result per patient
    conn = _live_conn()
    try:
        rows = conn.execute(
            """
            SELECT t.*
            FROM live_tick_results t
            INNER JOIN (
                SELECT patient_id, MAX(tick) AS max_tick
                FROM live_tick_results WHERE session_id = ?
                GROUP BY patient_id
            ) m ON t.patient_id = m.patient_id
               AND t.tick       = m.max_tick
               AND t.session_id = ?
            ORDER BY t.patient_id
            """,
            (session_id, session_id),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_latest_tick_stats(session_id: str) -> dict:
    # summary counts for the most recent tick
    conn = _live_conn()
    try:
        latest_tick = conn.execute(
            "SELECT COALESCE(MAX(tick),0) FROM live_tick_results WHERE session_id=?",
            (session_id,),
        ).fetchone()[0]
        if not latest_tick:
            return {}
        rows = conn.execute(
            "SELECT * FROM live_tick_results WHERE session_id=? AND tick=?",
            (session_id, latest_tick),
        ).fetchall()
    finally:
        conn.close()

    patients = [dict(r) for r in rows]
    if not patients:
        return {}

    def _avg(key):
        vals = [p[key] for p in patients if p.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    sc = {}
    for p in patients:
        st = p.get("status") or "Unknown"
        sc[st] = sc.get(st, 0) + 1

    return {
        "tick": latest_tick,
        "patient_count": len(patients),
        "status_counts": sc,
        "avg_heart_rate": _avg("heart_rate"),
        "avg_systolic": _avg("systolic_bp"),
        "avg_diastolic": _avg("diastolic_bp"),
        "avg_temp": _avg("body_temp"),
        "avg_spo2": _avg("spo2"),
    }


def get_tick_series(session_id: str, n: int = 40) -> list:
    # status breakdown across all ticks
    conn = _live_conn()
    try:
        rows = conn.execute(
            """
            SELECT tick, tick_time, status, COUNT(*) AS cnt
            FROM live_tick_results
            WHERE session_id = ?
            GROUP BY tick, status
            ORDER BY tick DESC
            LIMIT ?
            """,
            (session_id, n * 6),
        ).fetchall()
    finally:
        conn.close()

    ticks = {}
    for r in rows:
        t = r["tick"]
        if t not in ticks:
            ticks[t] = {"tick": t, "tick_time": r["tick_time"],
                        "Healthy": 0, "Unhealthy": 0, "Critical": 0}
        if r["status"] in ticks[t]:
            ticks[t][r["status"]] = r["cnt"]
    return sorted(ticks.values(), key=lambda x: x["tick"])[-n:]


def get_latest_session() -> Optional[str]:
    # most recent session ID from live DB
    conn = _live_conn()
    try:
        row = conn.execute(
            "SELECT session_id FROM live_tick_results ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def get_session_summary(session_id: str) -> dict:
    # aggregate stats for the current session
    conn = _live_conn()
    try:
        total_ticks = conn.execute(
            "SELECT COALESCE(MAX(tick),0) FROM live_tick_results WHERE session_id=?",
            (session_id,),
        ).fetchone()[0]

        total_patients = conn.execute(
            "SELECT COUNT(DISTINCT patient_id) FROM live_tick_results WHERE session_id=?",
            (session_id,),
        ).fetchone()[0]

        status_counts = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT status, COUNT(*) FROM live_tick_results "
                "WHERE session_id=? GROUP BY status",
                (session_id,),
            ).fetchall()
        }

        high_risk = conn.execute(
            """
            SELECT COUNT(DISTINCT lp.patient_id)
            FROM live_predictions lp
            INNER JOIN (
                SELECT patient_id, MAX(computed_at_tick) AS max_tick
                FROM live_predictions WHERE session_id = ?
                GROUP BY patient_id
            ) m ON lp.patient_id       = m.patient_id
               AND lp.computed_at_tick = m.max_tick
            WHERE lp.session_id = ? AND lp.risk_score >= 0.5
            """,
            (session_id, session_id),
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "session_id": session_id,
        "total_ticks": total_ticks,
        "total_patients": total_patients,
        "status_counts": status_counts,
        "high_risk_patients": high_risk,
    }


def search_live_patient(session_id: str, patient_id: int) -> Optional[dict]:
    conn = _live_conn()
    try:
        latest = conn.execute(
            """
            SELECT * FROM live_tick_results
            WHERE session_id=? AND patient_id=?
            ORDER BY tick DESC LIMIT 1
            """,
            (session_id, patient_id),
        ).fetchone()
        if not latest:
            return None

        history = conn.execute(
            """
            SELECT tick, tick_time, status, confidence, heart_rate, systolic_bp,
                   diastolic_bp, body_temp, spo2, activity
            FROM live_tick_results
            WHERE session_id=? AND patient_id=?
            ORDER BY tick ASC
            """,
            (session_id, patient_id),
        ).fetchall()

        prediction = conn.execute(
            """
            SELECT * FROM live_predictions
            WHERE session_id=? AND patient_id=?
            ORDER BY computed_at_tick DESC LIMIT 1
            """,
            (session_id, patient_id),
        ).fetchone()
    finally:
        conn.close()

    result = dict(latest)
    if result.get("attention_weights"):
        try:
            result["attention"] = json.loads(result["attention_weights"])
        except (json.JSONDecodeError, TypeError):
            result["attention"] = None
    else:
        result["attention"] = None
    result["history"] = [dict(r) for r in history]
    if prediction:
        pred = dict(prediction)
        try:
            pred["top_factors"] = json.loads(pred.get("top_factors", "[]"))
        except (json.JSONDecodeError, TypeError):
            pred["top_factors"] = []
        result["prediction"] = pred
    else:
        result["prediction"] = None
    return result
