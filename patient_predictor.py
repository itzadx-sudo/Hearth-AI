from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tabnet_engine import get_engine, TabNetEngine

WINDOW_DAYS    = 7
LOOKAHEAD_DAYS = 2

RISK_CONFIDENCE_THRESHOLD = 0.55


def _slope(values):
    clean = [v for v in values if v is not None and not np.isnan(float(v))]
    if len(clean) < 2:
        return 0.0
    x = np.arange(len(clean), dtype=float)
    return float(np.polyfit(x, clean, 1)[0])


def _compute_mrv(values):
    clean = [v for v in values if v is not None and not np.isnan(float(v))]
    if len(clean) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(clean))))


def _get_series(df_or_list, candidates):
    if isinstance(df_or_list, list):
        import pandas as pd
        df = pd.DataFrame(df_or_list)
    else:
        df = df_or_list
    for col in candidates:
        if col in df.columns:
            return [
                float(v) if v is not None and str(v) not in ("nan", "None") else None
                for v in df[col]
            ]
    return [None] * len(df)


def engineer_features_from_window(window):
    import pandas as pd

    df = pd.DataFrame(window) if isinstance(window, list) else window.copy()
    features = {}

    _SEQ_COL_MAP = {
        "heart_rate":   ["avg_heart_rate",  "heart_rate"],
        "systolic_bp":  ["avg_systolic",    "systolic_bp"],
        "diastolic_bp": ["avg_diastolic",   "diastolic_bp"],
        "body_temp":    ["avg_temp",        "body_temp"],
        "spo2":         ["avg_spo2",        "spo2"],
    }
    _AGG_FUNCS = ["mean", "std", "min", "max"]

    for vital, candidates in _SEQ_COL_MAP.items():
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            for agg in _AGG_FUNCS:
                features[f"{vital}_{agg}"] = 0.0
            features[f"{vital}_slope"] = 0.0
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        valid  = series.dropna()
        if len(valid) == 0:
            for agg in _AGG_FUNCS:
                features[f"{vital}_{agg}"] = 0.0
            features[f"{vital}_slope"] = 0.0
        else:
            features[f"{vital}_mean"]  = float(valid.mean())
            features[f"{vital}_std"]   = float(valid.std()) if len(valid) > 1 else 0.0
            features[f"{vital}_min"]   = float(valid.min())
            features[f"{vital}_max"]   = float(valid.max())
            features[f"{vital}_slope"] = _slope(valid.tolist())

    max_hr    = _get_series(df, ["max_heart_rate"])
    max_sys   = _get_series(df, ["max_systolic"])
    max_temp  = _get_series(df, ["max_temp"])
    min_spo2  = _get_series(df, ["min_spo2"])
    act_ratio = _get_series(df, ["activity_ratio"])
    crit_cnt  = _get_series(df, ["critical_count"])
    total_r   = _get_series(df, ["total_readings"])
    avg_conf  = _get_series(df, ["avg_confidence"])

    features["resting_high_hr_days"] = int(sum(
        1 for hr, ar in zip(max_hr, act_ratio)
        if hr is not None and ar is not None and hr > 110 and ar < 0.3
    ))
    features["low_spo2_days"] = int(sum(
        1 for s in min_spo2 if s is not None and s < 93
    ))
    features["fever_days"] = int(sum(
        1 for t in max_temp if t is not None and t > 37.8
    ))
    features["spo2_temp_danger_days"] = int(sum(
        1 for s, t in zip(min_spo2, max_temp)
        if s is not None and t is not None and s < 93 and t > 37.8
    ))
    features["critical_escalation"]  = _slope(crit_cnt)
    features["activity_decline"]     = _slope(act_ratio)

    mismatches = [
        hr * (1.0 - ar) for hr, ar in zip(max_hr, act_ratio)
        if hr is not None and ar is not None
    ]
    features["hr_activity_mismatch"] = sum(mismatches) / len(mismatches) if mismatches else 0.0
    features["max_systolic_slope"]   = _slope(max_sys)
    features["spo2_min_slope"]       = _slope(min_spo2)

    t_crit  = sum(c for c in crit_cnt if c is not None)
    t_total = sum(r for r in total_r  if r is not None)
    features["critical_ratio"]   = t_crit / t_total if t_total > 0 else 0.0
    features["confidence_trend"] = _slope(avg_conf)

    _SEQ_VITALS = ["heart_rate", "systolic_bp", "diastolic_bp", "body_temp", "spo2"]
    for vital in _SEQ_VITALS:
        col  = next((c for c in _SEQ_COL_MAP[vital] if c in df.columns), None)
        vals = pd.to_numeric(df[col], errors="coerce").tolist() if col else []
        features[f"{vital}_mrv"] = _compute_mrv(vals)

    mean_hr  = features.get("heart_rate_mean", 0.0)
    mean_sys = features.get("systolic_bp_mean", 1.0)
    features["shock_index"] = mean_hr / max(mean_sys, 1.0)

    act_col  = next((c for c in ["activity_ratio", "dominant_activity"] if c in df.columns), None)
    mean_act = float(pd.to_numeric(df[act_col], errors="coerce").mean()) if act_col else 0.0
    features["activity_adjusted_hr"] = mean_hr / max(mean_act, 0.1)

    return features


class PredictionEngine:
    def __init__(self):
        self._engine: TabNetEngine = get_engine()

    @property
    def is_ready(self) -> bool:
        return self._engine.is_ready

    def run_prediction_with_rows(self, patient_id, window_rows: List[dict], sim_date=None) -> Optional[dict]:
        result = self._engine.predict_risk(str(patient_id), window_rows)
        if result is None:
            return None
        risk_score = result.get("risk_score", 1.0)
        if risk_score < RISK_CONFIDENCE_THRESHOLD:
            result["low_confidence"] = True
            result["note"] = (
                f"Risk score ({risk_score:.2f}) is below the confidence threshold "
                f"({RISK_CONFIDENCE_THRESHOLD}). Treat as indicative only."
            )
        else:
            result["low_confidence"] = False
        return result

    def run_prediction(self, patient_id) -> Optional[dict]:
        import data_logger
        window = data_logger.get_rolling_window(patient_id, WINDOW_DAYS)
        if not window:
            return {"error": f"No history found for patient {patient_id}."}
        result = self.run_prediction_with_rows(patient_id, window)
        if result is None:
            return {"error": "Prediction engine not ready — train the model first."}
        return result

    def train(self, **kwargs):
        self._engine.train_from_db(**kwargs)


PatientPredictor = PredictionEngine


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the 7-day risk predictor.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=500000)
    args = parser.parse_args()
    PredictionEngine().train(epochs=args.epochs, batch_size=args.batch_size,
                             max_samples=args.max_samples)
    print("[OK] Risk predictor training complete.")

