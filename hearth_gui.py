import argparse
import json
import os
import sqlite3
import sys
import threading
import time as _time
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from flask import Flask, jsonify, render_template_string, request, session

except ImportError:
    print("[ERROR] Flask not installed.  Run:  pip install flask")
    sys.exit(1)

import api
import data_logger as live_db
from tabnet_engine import get_engine as _get_engine

import auth_db
import secrets

app = Flask(__name__)
app.json.sort_keys = False
app.secret_key = os.environ.get('HEARTH_SESSION_KEY', secrets.token_hex(16))
auth_db.init_db()

_training_in_progress = False
_training_message = ""
_training_lock = threading.Lock()

_TRAINING_ALLOWED_ENDPOINTS = {
    'api_auth_status', 'api_login', 'api_logout', 'api_signup',
    'api_training_status', 'api_retrain', 'index', 'landing', 'static',
}

@app.before_request
def _check_training_lockdown():
    if not _training_in_progress:
        return None
    endpoint = request.endpoint or ''
    if endpoint in _TRAINING_ALLOWED_ENDPOINTS:
        return None
    return jsonify({
        "error": "Model training in progress. Please wait for the downtime to end.",
        "training": True,
        "message": _training_message,
    }), 503

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
            if len(window) < 3:
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
            except Exception:
                pass
        with _pred_lock:
            _pred_cache.clear()
            _pred_cache.update(new_cache)
    except Exception as exc:
        print(f'[WARN] pred cache: {exc}')

def _pred_cache_worker():
    _time.sleep(4)
    while True:
        _refresh_pred_cache()
        _time.sleep(8)

threading.Thread(target=_pred_cache_worker, daemon=True).start()




def _safe(fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"[WARN] GUI: {exc}")
        return default


def _live_session():
    try:
        live_db.init_live_db()
        sid = live_db.get_latest_session()
        if sid:
            return sid, live_db.get_session_summary(sid)
    except Exception:
        pass
    return None, {}


def _get_guardian_patients():
    if session.get('role') == 'guardian':
        return auth_db.get_guardian_patients(session.get('username'))
    return None

def _filter_patients(patients_list, allowed_pids):
    if allowed_pids is None:
        return patients_list
    return [p for p in patients_list if p.get('patient_id') in allowed_pids]


@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.json or {}
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'guardian')
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
    p = os.path.join(BASE_DIR, 'hearth_landing.html')
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
    sid, summary = _live_session()
    health = _safe(api.get_system_health_sync, default={})
    allowed_pids = _get_guardian_patients()

    if sid and summary.get("total_ticks", 0) > 0:
        tick_stats = _safe(live_db.get_latest_tick_stats, sid, default={})
        patients   = _safe(live_db.get_latest_patient_states, sid, default=[])
        patients   = _filter_patients(patients, allowed_pids)
        preds      = _safe(live_db.get_latest_predictions, sid, default=[])
        live_preds = _filter_patients(list(preds), allowed_pids)
        live_preds.sort(key=lambda x: x.get("risk_score", 0) or 0, reverse=True)
        leaderboard = live_preds[:8]

        alerts = []
        try:
            LIVE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hearth_live.db')
            with sqlite3.connect(LIVE_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                recent = conn.execute(
                    "SELECT * FROM live_tick_results WHERE session_id=? AND status='Critical' ORDER BY tick DESC LIMIT 100",
                    (sid,)
                ).fetchall()
                alerts = [{"patient_id": r["patient_id"], "alert_type": "critical", "type": "critical", "sim_date": r["tick_time"], "confidence": r["confidence"]} for r in recent]
                alerts = _filter_patients(alerts, allowed_pids)
        except Exception:
            pass

        return jsonify({
            "live_mode":   True,
            "session_id":  sid,
            "patients":    patients,
            "tick_stats":  tick_stats,
            "day_summary": tick_stats,
            "leaderboard": leaderboard,
            "recent_alerts": alerts[:12],
            "system_health": health,
            "summary":     summary,
        })

    data          = _safe(api.get_dashboard_data_sync, default={})
    patients_data = _filter_patients(data.get("patients", []), allowed_pids)
    all_alerts    = (data.get("recent_alerts", []) + data.get("predictive_alerts", []))
    all_alerts    = _filter_patients(all_alerts, allowed_pids)
    with _pred_lock:
        cache_snap = dict(_pred_cache)
    for p in patients_data:
        pred = cache_snap.get(int(p.get('patient_id', 0)))
        if pred:
            p['risk_score'] = pred['risk_score']
            p['risk_label'] = pred.get('risk_label', p.get('risk_label', '—'))
    lb_all = sorted(cache_snap.values(), key=lambda x: x.get('risk_score', 0) or 0, reverse=True)
    lboard = _filter_patients(list(lb_all), allowed_pids)[:10]
    if not lboard:
        lboard = _safe(lambda: api._run_sync(api.get_risk_leaderboard(limit=100)), default=[])
        lboard = _filter_patients(lboard, allowed_pids)[:10]
    return jsonify({
        "live_mode":     False,
        "patients":      patients_data,
        "day_summary":   data.get("day_summary"),
        "leaderboard":   lboard,
        "recent_alerts": all_alerts[:12],
        "system_health": health,
        "summary":       {},
    })


@app.route("/api/patients")
def api_patients():
    sid, summary = _live_session()
    allowed_pids = _get_guardian_patients()
    if sid and summary.get("total_ticks", 0) > 0:
        patients = _safe(live_db.get_latest_patient_states, sid, default=[])
        patients = _filter_patients(patients, allowed_pids)
        preds_map = {p["patient_id"]: p for p in _safe(live_db.get_latest_predictions, sid, default=[])}
        for pt in patients:
            pred = preds_map.get(pt["patient_id"])
            pt["risk_score"] = pred["risk_score"] if pred else 0.0
            pt["risk_label"] = pred["risk_label"] if pred else "—"
            pt["worst_status"] = pt.get("status", "Unknown")
            pt["sim_date"]  = pt.get("tick_time", "")
            pt["avg_heart_rate"]  = pt.get("heart_rate")
            pt["avg_systolic"]    = pt.get("systolic_bp")
            pt["avg_diastolic"]   = pt.get("diastolic_bp")
            pt["avg_temp"]        = pt.get("body_temp")
            pt["avg_spo2"]        = pt.get("spo2")
        return jsonify(sorted(patients, key=lambda p: p.get("risk_score") or 0, reverse=True))

    data = _safe(api.get_dashboard_data_sync, default={})
    patients_data = _filter_patients(data.get("patients", []), allowed_pids)
    with _pred_lock:
        cache_snap = dict(_pred_cache)
    for p in patients_data:
        pred = cache_snap.get(int(p.get('patient_id', 0)))
        if pred:
            p['risk_score'] = pred['risk_score']
            p['risk_label'] = pred.get('risk_label', p.get('risk_label', '—'))
    return jsonify(sorted(patients_data, key=lambda p: p.get('risk_score') or 0, reverse=True))

@app.route("/api/patients/<int:pid>")
def api_patient_detail(pid):
    allowed_pids = _get_guardian_patients()
    if allowed_pids is not None and pid not in allowed_pids:
        return jsonify({}), 403
    
    sid, summary = _live_session()
    if sid and summary.get("total_ticks", 0) > 0:
        window = _safe(live_db.get_patient_window, sid, pid, 20, default=[])
        preds  = _safe(live_db.get_latest_predictions, sid, default=[])
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
    sid, summary = _live_session()
    allowed_pids = _get_guardian_patients()
    if sid and summary.get("total_ticks", 0) > 0:
        preds    = _safe(live_db.get_latest_predictions, sid, default=[])
        preds    = _filter_patients(preds, allowed_pids)
        patients = _safe(live_db.get_latest_patient_states, sid, default=[])
        patients = _filter_patients(patients, allowed_pids)
        crit     = [p for p in patients if p.get("status") == "Critical"]

        crit_alerts = [
            {"patient_id": p["patient_id"], "alert_type": "critical", "type": "critical",
             "sim_date": p.get("tick_time",""), "confidence": p.get("confidence"),
             "risk_label": None, "top_factors": []}
            for p in sorted(crit, key=lambda x: x.get("confidence") or 0, reverse=True)
        ]
        
        pred_alerts = []
        for p in preds:
            if (p.get("risk_score") or 0) >= 0.5:
                fac = p.get("top_factors", [])
                if isinstance(fac, str):
                    import json as _j
                    try: fac = _j.loads(fac)
                    except Exception: fac = []
                pred_alerts.append({
                    "patient_id": p["patient_id"], "alert_type": "predictive", "type": "predictive",
                    "sim_date": p.get("computed_at_time",""),
                    "confidence": p.get("risk_score"), "risk_label": p.get("risk_label"),
                    "top_factors": fac,
                })
                
        combined = crit_alerts + pred_alerts
        combined.sort(key=lambda x: x.get("sim_date", ""), reverse=True)
        return jsonify(combined)

    alerts = _safe(lambda: api._run_sync(api.get_alerts(limit=500)), default=[])
    if not alerts:
        db_rows = _safe(lambda: live_db.get_alerts_from_db(limit=200), default=[])
        alerts = []
        for r in db_rows:
            det = r.get('details') if isinstance(r.get('details'), dict) else {}
            alerts.append({
                'patient_id': r['patient_id'],
                'alert_type': r['alert_type'],
                'type':       r['alert_type'],
                'timestamp':  r.get('timestamp', ''),
                'sim_date':   r.get('timestamp', ''),
                'confidence': det.get('risk_score') if r['alert_type'] == 'predictive' else r.get('confidence'),
                'risk_label': r.get('severity'),
                'top_factors': det.get('top_factors', []) if isinstance(det, dict) else [],
            })
    alerts = _filter_patients(alerts, allowed_pids)[:60]
    return jsonify(alerts)



@app.route("/api/live")
def api_live():
    try:
        live_db.init_live_db()
        sid = live_db.get_latest_session()
        allowed_pids = _get_guardian_patients()
        if not sid:
            return jsonify({"session_id": None, "summary": {}, "predictions": [], "tick_series": [],
                            "message": "No live session found. Waiting for data..."})
        summary    = live_db.get_session_summary(sid)
        preds      = live_db.get_latest_predictions(sid, limit=500)
        preds      = _filter_patients(preds, allowed_pids)[:200]
        tick_series = live_db.get_tick_series(sid, n=40)
        tick_stats  = live_db.get_latest_tick_stats(sid)
        return jsonify({
            "session_id":  sid,
            "summary":     summary,
            "predictions": preds,
            "tick_series": tick_series,
            "tick_stats":  tick_stats,
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "session_id": None, "summary": {}, "predictions": [], "tick_series": []}), 500


@app.route("/api/patient/<int:patient_id>")
def api_patient_search(patient_id):
    allowed_pids = _get_guardian_patients()
    if allowed_pids is not None and patient_id not in allowed_pids:
        return jsonify({"error": "Unauthorized to view this patient", "found": False}), 403
    try:
        live_db.init_live_db()
        sid = live_db.get_latest_session()
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
    sid, _ = _live_session()
    health  = _safe(api.get_system_health_sync, default={})
    try:
        lv_info = live_db.get_session_summary(sid) if sid else {}
        live_size = os.path.getsize(live_db.LIVE_DB_PATH) / (1024*1024) if os.path.exists(live_db.LIVE_DB_PATH) else 0
    except Exception:
        lv_info, live_size = {}, 0
    health["live_ticks"]   = lv_info.get("total_ticks", 0)
    health["live_session"] = sid or "—"
    health["live_size_mb"] = round(live_size, 2)
    return jsonify(health)



def _retrain_worker(num_patients, num_days, readings_per_hour):
    global _training_in_progress, _training_message
    import data_logger as _dl
    from data_generator import generate_to_db

    try:
        _training_message = "Generating training data…"
        print(f"[RETRAIN] Generating {num_patients} patients × {num_days} days "
              f"@ {readings_per_hour} rdgs/hr")

        train_path = os.path.join(BASE_DIR, "hearth_sensor.db")
        orig_path = _dl.SENSOR_DB_PATH
        try:
            _dl.SENSOR_DB_PATH = train_path
            if os.path.exists(train_path):
                os.remove(train_path)
            generate_to_db(num_patients, num_days, readings_per_hour, seed=42)
        finally:
            _dl.SENSOR_DB_PATH = orig_path

        _training_message = "Training model…"
        print("[RETRAIN] Starting model training…")
        engine = _get_engine()
        engine.train_from_db(max_samples=500_000, epochs=30, batch_size=2048)

        _training_message = "Training complete!"
        print("[RETRAIN] Training complete.")
    except Exception as exc:
        _training_message = f"Training failed: {exc}"
        print(f"[RETRAIN] ERROR: {exc}")
    finally:
        with _training_lock:
            _training_in_progress = False


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    global _training_in_progress, _training_message
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    with _training_lock:
        if _training_in_progress:
            return jsonify({"error": "Training already in progress"}), 409
        _training_in_progress = True
    data = request.json or {}
    num_patients = max(1, min(int(data.get('num_patients', 50)), 5000))
    num_days = max(1, min(int(data.get('num_days', 30)), 730))
    readings_per_hour = max(1, min(int(data.get('readings_per_hour', 15)), 20))
    _training_message = "Initialising…"
    t = threading.Thread(
        target=_retrain_worker,
        args=(num_patients, num_days, readings_per_hour),
        daemon=True,
    )
    t.start()
    return jsonify({"message": "Training started", "training": True})


@app.route("/api/training_status")
def api_training_status():
    return jsonify({
        "training": _training_in_progress,
        "message": _training_message,
    })



def _load_html():
    p = os.path.join(BASE_DIR, 'hearth_dashboard.html')
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
