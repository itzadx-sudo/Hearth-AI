import asyncio
import collections
from datetime import datetime
import threading

from data import logger as data_logger
from config import MAX_ALERTS, DEBOUNCE_THRESHOLD

_alert_queue:             collections.deque = collections.deque(maxlen=MAX_ALERTS)
_predictive_queue:        collections.deque = collections.deque(maxlen=MAX_ALERTS)
_patient_critical_streak: dict              = {}
_patient_last_tick: dict                    = {}

_queue_lock      = threading.Lock()
_predictive_lock = threading.Lock()
_streak_lock     = threading.Lock()


async def check_and_alert(patient_id, result, *, session_id=None, tick=None, tick_time=None):
    key = (session_id, patient_id)
    fire = False
    with _streak_lock:
        if session_id is not None and tick is not None:
            last_tick = _patient_last_tick.get(key)
            if last_tick is not None and tick <= last_tick:
                return
            if last_tick is not None and tick != last_tick + 1:
                _patient_critical_streak.pop(key, None)
            _patient_last_tick[key] = tick
        if result.get("status") != "Critical":
            _patient_critical_streak.pop(key, None)
            return
        streak = _patient_critical_streak.get(key, 0) + 1
        _patient_critical_streak[key] = streak
        if streak >= DEBOUNCE_THRESHOLD:
            fire = True
            _patient_critical_streak[key] = 0

    if not fire:
        return

    alert = {
        "timestamp":  tick_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id": patient_id,
        "status":     result["status"],
        "confidence": round(result.get("confidence", 0.0), 4),
        "vitals":     result.get("input_used", {}),
    }

    if session_id is not None:
        alert["session_id"] = session_id
        alert["tick"] = tick
        data_logger.store_live_alert(session_id, tick, alert["timestamp"], patient_id,
                                     "critical", result["status"],
                                     confidence=alert["confidence"], details={"vitals": alert["vitals"]})
    else:
        data_logger.store_alert(patient_id, alert_type="critical",
                                severity=result["status"],
                                details={"vitals": alert["vitals"], "confidence": alert["confidence"]})

    with _queue_lock:
        _alert_queue.append(alert)
    return alert


async def get_alerts(limit=None) -> list:
    with _queue_lock:
        alerts = list(reversed(_alert_queue))
    return alerts[:limit] if limit is not None else alerts


async def alert_count() -> int:
    with _queue_lock:
        return len(_alert_queue)


async def add_predictive_alert(patient_id, prediction_result, *, session_id=None, tick=None, tick_time=None):
    if prediction_result.get("risk_label") != "HIGH RISK":
        return

    alert = {
        "timestamp":   tick_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id":  patient_id,
        "risk_label":  "HIGH RISK",
        "risk_score":  prediction_result.get("risk_score"),
        "top_factors": prediction_result.get("top_factors"),
    }

    details = {"risk_score": alert["risk_score"], "top_factors": alert["top_factors"]}
    if session_id is not None:
        alert["session_id"] = session_id
        alert["tick"] = tick
        data_logger.store_live_alert(session_id, tick, alert["timestamp"], patient_id,
                                     "predictive", "HIGH RISK", confidence=alert["risk_score"],
                                     details=details)
    else:
        data_logger.store_alert(patient_id, alert_type="predictive", severity="HIGH RISK",
                                details=details)

    with _predictive_lock:
        _predictive_queue.append(alert)
    return alert


async def get_predictive_alerts(limit=None) -> list:
    with _predictive_lock:
        alerts = list(reversed(_predictive_queue))
    return alerts[:limit] if limit is not None else alerts


async def predictive_alert_count() -> int:
    with _predictive_lock:
        return len(_predictive_queue)


# non-async versions for flask routes
def get_alerts_sync(limit=None) -> list:
    with _queue_lock:
        alerts = list(reversed(_alert_queue))
    return alerts[:limit] if limit is not None else alerts

def get_predictive_alerts_sync(limit=None) -> list:
    with _predictive_lock:
        alerts = list(reversed(_predictive_queue))
    return alerts[:limit] if limit is not None else alerts

def alert_count_sync() -> int:
    with _queue_lock:
        return len(_alert_queue)

def predictive_alert_count_sync() -> int:
    with _predictive_lock:
        return len(_predictive_queue)


if __name__ == "__main__":
    async def _self_test():
        fake = {"status": "Critical", "confidence": 0.93, "input_used": {}}
        await check_and_alert(42, fake)
        await check_and_alert(42, fake)
        await check_and_alert(42, fake)
        await check_and_alert(7,  {"status": "Healthy", "confidence": 0.99})
        print(f"[OK] alert_engine self-test passed. Buffered alerts: {await alert_count()}")
        for a in await get_alerts():
            print(a)
    asyncio.run(_self_test())
