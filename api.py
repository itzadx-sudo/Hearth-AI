from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pydantic import BaseModel, Field, field_validator

import data_logger
import alert_engine
from tabnet_engine import get_engine, CHECKPOINT_PATH
from patient_predictor import engineer_features_from_window


class SensorReading(BaseModel):
    patient_id:   int             = Field(..., gt=0)
    timestamp:    str
    heart_rate:   Optional[float] = Field(None, ge=30,   le=250)
    systolic_bp:  Optional[float] = Field(None, ge=50,   le=300)
    diastolic_bp: Optional[float] = Field(None, ge=30,   le=200)
    body_temp:    Optional[float] = Field(None, ge=32.0, le=43.0)
    spo2:         Optional[float] = Field(None, ge=50.0, le=100.0)
    activity:     Optional[int]   = Field(None, ge=0, le=5)

    @field_validator("activity", mode="before")
    @classmethod
    def coerce_activity(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            mapping = {
                "active":   3,
                "resting":  0,
                "sedentary": 0,
                "light":    1,
                "moderate": 3,
                "vigorous": 4,
                "strenuous": 5,
            }
            coerced = mapping.get(v.strip().lower())
            if coerced is not None:
                return coerced
            try:
                return int(v)
            except ValueError:
                raise ValueError(
                    f"activity must be an integer 0–5 or one of "
                    f"{list(mapping.keys())}; got '{v}'"
                )
        return v

    @field_validator("diastolic_bp")
    @classmethod
    def dbp_less_than_sbp(cls, v, info):
        sbp = info.data.get("systolic_bp")
        if v is not None and sbp is not None and v >= sbp:
            raise ValueError(f"diastolic_bp ({v}) must be less than systolic_bp ({sbp})")
        return v


def validate_sensor_reading(reading: dict) -> SensorReading:
    try:
        return SensorReading(**reading)
    except Exception as exc:
        raise ValueError(f"Invalid sensor reading: {exc}") from exc


# overview endpoint
async def get_dashboard_data(sim_date=None) -> dict:
    patients, high_risk, alerts, pred_alerts = await asyncio.gather(
        asyncio.to_thread(data_logger.get_all_patients_latest),
        asyncio.to_thread(data_logger.get_high_risk_patients),
        asyncio.to_thread(alert_engine.get_alerts_sync, 20),
        asyncio.to_thread(alert_engine.get_predictive_alerts_sync, 20),
    )
    day_summary = None
    if sim_date:
        day_summary = await asyncio.to_thread(data_logger.get_day_overview, sim_date)
    elif patients:
        day_summary = await asyncio.to_thread(data_logger.get_day_overview,
                                              max(p["sim_date"] for p in patients))
    return {
        "patients":          patients,
        "day_summary":       day_summary,
        "recent_alerts":     alerts,
        "high_risk":         high_risk,
        "predictive_alerts": pred_alerts,
    }


# patient detail
async def get_patient_detail(patient_id, history_days: int = 30) -> dict:
    history     = await asyncio.to_thread(data_logger.get_patient_history, patient_id, history_days)
    predictions = await asyncio.to_thread(data_logger.get_predictions_for_patient, patient_id, 10)
    critical_tl = await asyncio.to_thread(data_logger.get_critical_timeline, patient_id, history_days)
    return {
        "current_status":    history[-1] if history else None,
        "history":           history,
        "predictions":       predictions,
        "critical_timeline": critical_tl,
        "vitals_trend": {
            "dates":        [h["sim_date"] for h in history],
            "heart_rate":   [h.get("avg_heart_rate") for h in history],
            "systolic_bp":  [h.get("avg_systolic") for h in history],
            "diastolic_bp": [h.get("avg_diastolic") for h in history],
            "body_temp":    [h.get("avg_temp") for h in history],
            "spo2":         [h.get("avg_spo2") for h in history],
        },
    }


async def get_system_health() -> dict:
    patients     = await asyncio.to_thread(data_logger.get_all_patient_ids_in_db2)
    dates        = await asyncio.to_thread(data_logger.get_dates_available)
    model_status = "loaded" if os.path.exists(CHECKPOINT_PATH) else "missing"
    db_size_mb   = round(
        sum(os.path.getsize(p) for p in (data_logger.SENSOR_DB_PATH, data_logger.RESULTS_DB_PATH)
            if os.path.exists(p)) / (1024 * 1024), 2,
    )
    return {
        "total_patients":   len(patients),
        "days_processed":   len(dates),
        "model_status":     model_status,
        "predictor_status": model_status,
        "db_size_mb":       db_size_mb,
    }


async def trigger_prediction(patient_id) -> dict:
    engine = get_engine()
    if not engine.is_ready:
        return {"error": "Prediction model not loaded."}
    return await asyncio.to_thread(engine.predict_risk, str(patient_id),
                                   data_logger.get_rolling_window(patient_id, 7) or [])


async def get_alerts(limit: int = 50, alert_type: str = None) -> list:
    if alert_type == "critical":
        return alert_engine.get_alerts_sync(limit=limit)
    if alert_type == "predictive":
        return alert_engine.get_predictive_alerts_sync(limit=limit)
    combined = alert_engine.get_alerts_sync() + alert_engine.get_predictive_alerts_sync()
    combined.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
    return combined[:limit] if limit is not None else combined


async def get_low_confidence_patients(sim_date=None) -> list:
    if sim_date is None:
        patients = await asyncio.to_thread(data_logger.get_all_patients_latest)
        if not patients:
            return []
        sim_date = max(p["sim_date"] for p in patients)
    return await asyncio.to_thread(data_logger.get_low_confidence_patients, sim_date)


async def get_risk_leaderboard(limit: int = 10) -> list:
    rows = await asyncio.to_thread(data_logger.get_all_latest_predictions, limit)
    return [{"patient_id": r["patient_id"], "risk_label": r["risk_label"],
             "risk_score": r["risk_score"], "sim_date": r["sim_date"]} for r in rows]


async def get_sudden_changes(sim_date=None) -> list:
    if sim_date is None:
        patients = await asyncio.to_thread(data_logger.get_all_patients_latest)
        if not patients:
            return []
        sim_date = max(p["sim_date"] for p in patients)
    return await asyncio.to_thread(data_logger.get_sudden_changes, sim_date)


async def get_patient_context_metrics(patient_id) -> Optional[dict]:
    rows = await asyncio.to_thread(data_logger.get_rolling_window, patient_id, 7)
    if not rows or len(rows) < 3:
        return None

    feats = engineer_features_from_window(rows)

    # bucket a metric into severity tiers for the frontend badges
    def _sev(val, warn_lo, danger_lo):
        if val >= danger_lo: return "danger"
        if val >= warn_lo:   return "warning"
        return "normal"

    result = {
        "resting_high_hr_days":  {"value": (v := feats.get("resting_high_hr_days",  0)), "severity": _sev(v, 1, 3)},
        "low_spo2_days":         {"value": (v := feats.get("low_spo2_days",         0)), "severity": _sev(v, 1, 3)},
        "fever_days":            {"value": (v := feats.get("fever_days",            0)), "severity": _sev(v, 1, 3)},
        "spo2_temp_danger_days": {"value": (v := feats.get("spo2_temp_danger_days", 0)), "severity": _sev(v, 1, 2)},
    }

    v = feats.get("critical_escalation", 0)
    result["critical_escalation"] = {"value": round(v, 4),
                                     "severity": "danger" if v > 0.5 else "warning" if v > 0 else "normal"}

    v = feats.get("activity_decline", 0)
    result["activity_decline"] = {"value": round(v, 4),
                                  "severity": "normal" if v >= 0 else "warning" if v >= -0.05 else "danger"}

    v = feats.get("hr_activity_mismatch", 0)
    result["hr_activity_mismatch"] = {"value": round(v, 2),
                                      "severity": "normal" if v < 80 else "warning" if v <= 100 else "danger"}

    result["max_systolic_slope"] = {"value": round(feats.get("max_systolic_slope", 0), 4)}
    result["spo2_min_slope"]     = {"value": round(feats.get("spo2_min_slope",     0), 4)}

    v = feats.get("critical_ratio", 0)
    result["critical_ratio"] = {"value": round(v, 4),
                                "severity": "normal" if v < 0.05 else "warning" if v <= 0.15 else "danger"}

    v = feats.get("confidence_trend", 0)
    result["confidence_trend"] = {"value": round(v, 4),
                                  "severity": "normal" if v >= 0 else "warning" if v >= -0.03 else "danger"}

    for vital in ["heart_rate", "systolic_bp", "diastolic_bp", "body_temp", "spo2"]:
        result[f"{vital}_slope"] = {"value": round(feats.get(f"{vital}_slope", 0), 4)}

    return result


# patient lookup
async def lookup_patient(patient_id) -> dict:
    detail  = await get_patient_detail(patient_id)
    context = await get_patient_context_metrics(patient_id)
    stored_preds = detail.get("predictions", [])
    pred = stored_preds[0] if stored_preds else None
    if pred is None:
        fresh = await trigger_prediction(patient_id)
        if fresh and "error" not in fresh:
            pred = fresh
    return {
        "patient_id":        patient_id,
        "current_status":    detail.get("current_status"),
        "latest_prediction": pred,
        "context_metrics":   context,
        "vitals_trend":      detail.get("vitals_trend"),
        "history":           detail.get("history", []),
        "predictions":       detail.get("predictions", []),
        "critical_timeline": detail.get("critical_timeline", []),
    }


# sync wrappers
# flask is sync but our API layer is async, so we bridge with a thread pool
_sync_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def _run_sync(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # already in an event loop (e.g. jupyter), run in separate thread
            return _sync_executor.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def get_dashboard_data_sync(sim_date=None):               return _run_sync(get_dashboard_data(sim_date))
def get_patient_detail_sync(patient_id, history_days=30): return _run_sync(get_patient_detail(patient_id, history_days))
def get_system_health_sync():                             return _run_sync(get_system_health())
def trigger_prediction_sync(patient_id):                  return _run_sync(trigger_prediction(patient_id))
def lookup_patient_sync(patient_id):                      return _run_sync(lookup_patient(patient_id))
