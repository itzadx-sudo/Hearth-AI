import asyncio
import collections
from datetime import datetime
import threading

from data import logger as data_logger
from config import MAX_ALERTS, DEBOUNCE_THRESHOLD

_alert_queue:             collections.deque = collections.deque(maxlen=MAX_ALERTS)
_predictive_queue:        collections.deque = collections.deque(maxlen=MAX_ALERTS)
_patient_critical_streak: dict              = {}

_queue_lock      = threading.Lock()
_predictive_lock = threading.Lock()
_streak_lock     = threading.Lock()


async def check_and_alert(patient_id, result):
    if result.get("status") != "Critical":
        with _streak_lock:
            _patient_critical_streak.pop(patient_id, None)
        return

    # count consecutive criticals, reset after firing
    fire = False
    with _streak_lock:
        streak = _patient_critical_streak.get(patient_id, 0) + 1
        _patient_critical_streak[patient_id] = streak
        if streak >= DEBOUNCE_THRESHOLD:
            fire = True
            _patient_critical_streak[patient_id] = 0

    if not fire:
        return

    alert = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id": patient_id,
        "status":     result["status"],
        "confidence": round(result.get("confidence", 0.0), 4),
        "vitals":     result.get("input_used", {}),
    }

    data_logger.store_alert(patient_id, alert_type="critical",
                            severity=result["status"], details=result.get("input_used", {}))

    with _queue_lock:
        _alert_queue.append(alert)


async def get_alerts(limit=None) -> list:
    with _queue_lock:
        alerts = list(reversed(_alert_queue))
    return alerts[:limit] if limit is not None else alerts


async def alert_count() -> int:
    with _queue_lock:
        return len(_alert_queue)


async def add_predictive_alert(patient_id, prediction_result):
    if prediction_result.get("risk_label") != "HIGH RISK":
        return

    alert = {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id":  patient_id,
        "risk_label":  "HIGH RISK",
        "risk_score":  prediction_result.get("risk_score"),
        "top_factors": prediction_result.get("top_factors"),
    }

    data_logger.store_alert(patient_id, alert_type="predictive", severity="HIGH RISK",
                            details={"risk_score": prediction_result.get("risk_score"),
                                     "top_factors": prediction_result.get("top_factors")})

    with _predictive_lock:
        _predictive_queue.append(alert)


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