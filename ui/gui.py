import argparse
import json
import os
import sqlite3
import sys
import threading
import time as _time
import webbrowser
from paths import _path, _data_path, BASE_DIR

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from flask import Flask, jsonify, render_template_string, request, session

except ImportError:
    print("[ERROR] Flask not installed.  Run:  pip install flask")
    sys.exit(1)

from server import alert_engine
from server import api
from data import logger as live_db
from model.engine import get_engine as _get_engine

from auth import db as auth_db
import secrets

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))
app.json.sort_keys = False
app.secret_key = os.environ.get('HEARTH_SESSION_KEY', secrets.token_hex(16))
auth_db.init_db()
auth_db.create_user('admin', 'adminpassword', 'admin')

_PUBLIC_API_ENDPOINTS = {'api_auth_status', 'api_login', 'api_logout', 'api_signup'}

@app.before_request
def _require_api_login():
    if not request.path.startswith('/api/') or request.endpoint in _PUBLIC_API_ENDPOINTS:
        return None
    if not session.get('username') or session.get('role') not in {'admin', 'guardian'}:
        return jsonify({'error': 'Login required'}), 401


# background thread refreshes predictions every 8s so the dashboard doesn't lag
_pred_cache: dict = {}
_pred_lock = threading.Lock()

def _refresh_pred_cache():
    engine = _safe(_get_engine)
    if not engine or not getattr(engine, 'is_ready', False):
        return
    try:
        pids = _safe(live_db.get_all_patient_ids_in_db2, default=[])
        if not pids:
            return
        bulk_windows = _safe(live_db.get_bulk_rolling_windows, pids, 7, default={})
        new_cache = {}
        for pid, window in bulk_windows.items():
            if len(window) < 7:
                continue
            try:
                result = engine.predict_risk(str(pid), window)
                if result and 'error' not in result:
                    new_cache[int(pid)] = {
                        'patient_id': int(pid),
                        'risk_score': float(result.get('risk_score') or 0),
                        'risk_label': result.get('risk_label', '—'),
                        'top_factors': result.get('top_factors') or [],
                        'model':       result.get('model', 'Hearth Model'),
                        'sim_date':    (window[-1].get('sim_date', '') if window else ''),
                        'computed_at': _time.strftime('%Y-%m-%d %H:%M:%S'),
                    }
            except Exception as _cache_exc:
                print(f'[WARN] pred cache: patient {pid} skipped — {_cache_exc}')
        new_cache['_refreshed_at'] = _time.time()
        with _pred_lock:
            # replace dict contents atomically — new_cache was built outside the lock
            _pred_cache.clear()
            _pred_cache.update(new_cache)
    except Exception as exc:
        print(f'[WARN] pred cache: {exc}')

def _pred_cache_worker():
    _time.sleep(4)
    while True:
        start_t = _time.time()
        _refresh_pred_cache()
        elapsed = _time.time() - start_t
        sleep_time = max(0.0, 8.0 - elapsed)
        _time.sleep(sleep_time)

threading.Thread(target=_pred_cache_worker, daemon=True).start()




def _patient_cache_snap() -> dict:
    with _pred_lock:
        return {k: v for k, v in _pred_cache.items() if isinstance(k, int)}


def _safe(fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"[WARN] GUI: {exc}")
        return default


def _live_session(patient_ids=None):
    live_db.init_live_db()
    sid = live_db.get_latest_session(patient_ids=patient_ids)
    return (sid, live_db.get_session_summary(sid, patient_ids=patient_ids)) if sid else (None, {})


def _get_guardian_patients():
    if session.get('role') == 'admin':
        return None
    if session.get('role') == 'guardian' and session.get('username'):
        return auth_db.get_guardian_patients(session['username'])
    return []

# None means admin (see all), list means guardian (restricted)
def _filter_patients(patients_list, allowed_pids):
    if allowed_pids is None:
        return patients_list
    return [p for p in patients_list if p.get('patient_id') in allowed_pids]


def _scoped_health(patient_ids, sid=None, summary=None):
    if patient_ids is None:
        health = api.get_system_health_sync()
        health['live_size_mb'] = round(os.path.getsize(live_db.LIVE_DB_PATH) / (1024 * 1024), 2) if os.path.exists(live_db.LIVE_DB_PATH) else 0
    else:
        health = live_db.get_history_counts(patient_ids)
        model_status = 'loaded' if os.path.exists(api.CHECKPOINT_PATH) else 'missing'
        health.update(model_status=model_status, predictor_status=model_status)
        if sid:
            health['total_patients'] = (summary or {}).get('total_patients', 0)
    health['live_ticks'] = (summary or {}).get('total_ticks', 0)
    health['live_session'] = sid or '—'
    return health


def _scoped_patients(sid, patient_ids):
    if sid:
        patients = live_db.get_latest_patient_states(sid, patient_ids=patient_ids)
        predictions = {row['patient_id']: row for row in live_db.get_latest_predictions(sid, patient_ids=patient_ids)}
        for row in patients:
            prediction = predictions.get(row['patient_id'], {})
            row.update(risk_score=prediction.get('risk_score', 0.0), risk_label=prediction.get('risk_label', '—'),
                       worst_status=row.get('status', 'Unknown'), sim_date=row.get('tick_time', ''),
                       avg_heart_rate=row.get('heart_rate'), avg_systolic=row.get('systolic_bp'),
                       avg_diastolic=row.get('diastolic_bp'), avg_temp=row.get('body_temp'), avg_spo2=row.get('spo2'))
    else:
        patients = _filter_patients(live_db.get_all_patients_latest(), patient_ids)
        cache = _patient_cache_snap()
        for row in patients:
            prediction = cache.get(int(row['patient_id']))
            if prediction:
                row.update(risk_score=prediction['risk_score'], risk_label=prediction.get('risk_label', '—'))
    return patients


def _scoped_alerts(sid, patient_ids, limit=200):
    rows = (live_db.get_live_alerts(sid, limit=limit, patient_ids=patient_ids) if sid else
            live_db.get_alerts_from_db(limit=limit, patient_ids=patient_ids))
    alerts = []
    for row in rows:
        details = row.get('details') if isinstance(row.get('details'), dict) else {}
        predictive = row['alert_type'] == 'predictive'
        confidence = row.get('confidence')
        if confidence is None:
            confidence = details.get('risk_score') if predictive else details.get('confidence')
        alerts.append({
            'id': row.get('id'), 'patient_id': row['patient_id'],
            'alert_type': row['alert_type'], 'type': row['alert_type'],
            'timestamp': row.get('timestamp', ''), 'sim_date': row.get('timestamp', ''),
            'confidence': confidence, 'risk_label': row.get('severity') if predictive else None,
            'top_factors': details.get('top_factors') or [],
            'session_id': row.get('session_id'), 'tick': row.get('tick'),
        })
    return alerts


@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.json or {}
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'guardian')
    if role != 'guardian':
        return jsonify({'error': 'Signup creates guardian accounts only'}), 403
    if not username or not password:
        return jsonify({"error": "Missing credentials"}), 400
    success = auth_db.create_user(username, password, role)
    if success:
        return jsonify({"message": "User created"})
    return jsonify({"error": "Username already exists or error occurred"}), 400

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json or {}
    username = data.get('username')
    password = data.get('password')
    user = auth_db.verify_user(username, password)
    if user:
        session['username'] = user['username']
        session['role'] = user['role']
        session['display_name'] = user.get('display_name')
        return jsonify({"message": "Logged in", "user": user})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/api/logout", methods=["POST", "GET"])
def api_logout():
    session.clear()
    return jsonify({"message": "Logged out"})

@app.route("/api/auth_status")
def api_auth_status():
    if 'username' in session:
        return jsonify({
            "logged_in": True,
            "username": session['username'],
            "display_name": session.get('display_name'),
            "role": session.get('role'),
            "patients": auth_db.get_guardian_patients(session['username']) if session.get('role') == 'guardian' else []
        })
    return jsonify({"logged_in": False})

@app.route("/api/update_name", methods=["POST"])
def api_update_name():
    data = request.json or {}
    new_name = data.get('display_name')
    if not new_name or 'username' not in session:
        return jsonify({"error": "Invalid request"}), 400
    if auth_db.update_display_name(session['username'], new_name):
        session['display_name'] = new_name
        return jsonify({"message": "Name updated successfully"})
    return jsonify({"error": "Failed to update name"}), 500

@app.route("/api/assign_patient", methods=["POST"])
def api_assign_patient():
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    data = request.json or {}
    g_user = data.get('guardian_username')
    pid = data.get('patient_id')
    try:
        pid = int(pid)
    except:
        return jsonify({"error": "Invalid patient ID"}), 400
    if auth_db.assign_patient(g_user, pid):
        return jsonify({"message": f"Patient {pid} assigned to {g_user}"})
    return jsonify({"error": "Failed to assign. Check if guardian exists."}), 400

@app.route("/api/guardians")
def api_guardians():
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(auth_db.get_all_guardians())



def _load_landing_html():
    p = _path('hearth_landing.html')
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return '<h2 style="font-family:sans-serif;padding:40px">hearth_landing.html not found next to hearth_gui.py</h2>'


@app.route("/landing")
def landing():
    return _load_landing_html(), 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route("/")
def index():
    return _load_html(), 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route("/api/overview")
def api_overview():
    allowed_pids = _get_guardian_patients()
    sid, summary = _live_session(allowed_pids)
    patients = _scoped_patients(sid, allowed_pids)
    if sid:
        day_summary = live_db.get_latest_tick_stats(sid, patient_ids=allowed_pids)
        leaderboard = live_db.get_latest_predictions(sid, limit=10, patient_ids=allowed_pids)
    else:
        latest_date = max((row['sim_date'] for row in patients), default=None)
        day_summary = live_db.get_day_overview(latest_date, patient_ids=allowed_pids) if latest_date else None
        leaderboard = _filter_patients(list(_patient_cache_snap().values()), allowed_pids)
        if not leaderboard:
            leaderboard = _filter_patients(live_db.get_all_latest_predictions(), allowed_pids)
        leaderboard = sorted(leaderboard, key=lambda row: row.get('risk_score') or 0, reverse=True)[:10]
    return jsonify({
        'live_mode': bool(sid), 'session_id': sid, 'patients': patients,
        'tick_stats': day_summary if sid else {}, 'day_summary': day_summary,
        'leaderboard': leaderboard, 'recent_alerts': _scoped_alerts(sid, allowed_pids, limit=12),
        'system_health': _scoped_health(allowed_pids, sid, summary), 'summary': summary,
    })


@app.route("/api/patients")
def api_patients():
    allowed_pids = _get_guardian_patients()
    sid, _ = _live_session(allowed_pids)
    patients = _scoped_patients(sid, allowed_pids)
    return jsonify(sorted(patients, key=lambda row: row.get('risk_score') or 0, reverse=True))

@app.route("/api/patients/<int:pid>")
def api_patient_detail(pid):
    allowed_pids = _get_guardian_patients()
    if allowed_pids is not None and pid not in allowed_pids:
        return jsonify({}), 403
    
    sid, summary = _live_session(allowed_pids)
    if sid and summary.get("total_ticks", 0) > 0:
        window = _safe(live_db.get_patient_window, sid, pid, 20, default=[])
        preds  = _safe(live_db.get_latest_predictions, sid, patient_ids=[pid], default=[])
        pred   = next((p for p in preds if p["patient_id"] == pid), None)
        latest = window[-1] if window else {}
        if pred and isinstance(pred.get("top_factors"), str):
            import json as _json
            try: pred["top_factors"] = _json.loads(pred["top_factors"])
            except Exception: pass
        return jsonify({
            "patient_id":     pid,
            "current_status": {
                "worst_status":   latest.get("status"),
                "sim_date":       latest.get("tick_time"),
                "avg_heart_rate": latest.get("heart_rate"),
                "avg_systolic":   latest.get("systolic_bp"),
                "avg_diastolic":  latest.get("diastolic_bp"),
                "avg_temp":       latest.get("body_temp"),
                "avg_spo2":       latest.get("spo2"),
                "confidence":     latest.get("confidence"),
            },
            "latest_prediction": pred,
            "context_metrics":   {},
            "vitals_trend": {
                "dates":       [r.get("tick_time", "") for r in window],
                "heart_rate":  [r.get("heart_rate")   for r in window],
                "systolic_bp": [r.get("systolic_bp")  for r in window],
                "body_temp":   [r.get("body_temp")    for r in window],
                "spo2":        [r.get("spo2")         for r in window],
            },
            "history":           window,
        })

    detail = _safe(api.get_patient_detail_sync, pid, default={})
    with _pred_lock:
        pred = _pred_cache.get(pid)
    if pred is None:
        stored = detail.get('predictions', [])
        pred   = stored[0] if stored else None
    return jsonify({
        'patient_id':        pid,
        'current_status':    detail.get('current_status'),
        'latest_prediction': pred,
        'context_metrics':   {},
        'vitals_trend':      detail.get('vitals_trend'),
        'history':           detail.get('history', []),
        'predictions':       detail.get('predictions', []),
        'critical_timeline': detail.get('critical_timeline', []),
    })


@app.route("/api/alerts")
def api_alerts():
    allowed_pids = _get_guardian_patients()
    sid, _ = _live_session(allowed_pids)
    return jsonify(_scoped_alerts(sid, allowed_pids))



@app.route("/api/live")
def api_live():
    allowed_pids = _get_guardian_patients()
    sid, summary = _live_session(allowed_pids)
    if not sid:
        return jsonify({'session_id': None, 'summary': {}, 'predictions': [], 'tick_series': [],
                        'tick_stats': {}, 'message': 'No live session found. Waiting for data...'})
    return jsonify({
        'session_id': sid, 'summary': summary,
        'predictions': live_db.get_latest_predictions(sid, limit=200, patient_ids=allowed_pids),
        'tick_series': live_db.get_tick_series(sid, n=40, patient_ids=allowed_pids),
        'tick_stats': live_db.get_latest_tick_stats(sid, patient_ids=allowed_pids),
    })


@app.route("/api/patient/<int:patient_id>")
def api_patient_search(patient_id):
    allowed_pids = _get_guardian_patients()
    if allowed_pids is not None and patient_id not in allowed_pids:
        return jsonify({"error": "Unauthorized to view this patient", "found": False}), 403
    try:
        live_db.init_live_db()
        sid, _ = _live_session(allowed_pids)
        if not sid:
            return jsonify({"error": "No live session active", "found": False}), 404
        
        patient = live_db.search_live_patient(sid, patient_id)
        if not patient:
            return jsonify({"error": f"Patient {patient_id} not found in session", "found": False}), 404
        
        return jsonify({"found": True, "session_id": sid, "patient": patient})
    except Exception as exc:
        return jsonify({"error": str(exc), "found": False}), 500


@app.route("/api/system")
def api_system():
    allowed_pids = _get_guardian_patients()
    sid, summary = _live_session(allowed_pids)
    return jsonify(_scoped_health(allowed_pids, sid, summary))



_TRAINING_DISABLED_MESSAGE = (
    "Dashboard training is disabled. Stop Hearth, follow docs/training.md "
    "to train from the command line, then restart the launcher."
)


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify({"error": _TRAINING_DISABLED_MESSAGE,
                    "enabled": False, "training": False}), 410


@app.route("/api/training_status")
def api_training_status():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify({"enabled": False, "training": False,
                    "message": _TRAINING_DISABLED_MESSAGE})


def _load_html():
    p = _path('hearth_dashboard.html')
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return '<h2 style="font-family:sans-serif;padding:40px">hearth_dashboard.html not found next to hearth_gui.py</h2>'



def run_gui(port: int = 8050, open_browser: bool = True):
    if open_browser:
        def _open():
            import time
            time.sleep(1.4)
            webbrowser.open(f"http://localhost:{port}/landing")
        threading.Thread(target=_open, daemon=True).start()
    print(f"\n{'='*55}")
    print(f"  Hearth AI — Web Dashboard")
    print(f"{'='*55}")
    print(f"  URL  :  http://localhost:{port}")
    print(f"  Stop :  Ctrl+C")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


def run_gui_in_thread(port: int = 8050, open_browser: bool = True) -> threading.Thread:
    import time as _time

    def _flask():
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_flask, daemon=True)
    t.start()

    if open_browser:
        def _open():
            _time.sleep(2.0)
            webbrowser.open(f"http://localhost:{port}/landing")
        threading.Thread(target=_open, daemon=True).start()

    print(f"\n{'='*55}")
    print(f"  Hearth AI — Web Dashboard (background)")
    print(f"{'='*55}")
    print(f"  URL  :  http://localhost:{port}")
    print(f"  Stop :  Ctrl+C")
    print(f"{'='*55}\n")
    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hearth AI Web Dashboard")
    parser.add_argument("--port",       type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    run_gui(port=args.port, open_browser=not args.no_browser)
