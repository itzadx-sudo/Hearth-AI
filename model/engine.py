from __future__ import annotations

import math
import os
import threading
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
from paths import _data_path
from constants import news2_score
from config import (EXERTION_BIAS_HR, EXERTION_BIAS_SBP, RING_BUFFER_SIZE,
                    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR,
                    DEFAULT_WEIGHT_DECAY, RISK_LABEL_THRESHOLD,
                    LOW_CONFIDENCE_THRESHOLD)
from constants import (VITALS, N_VITALS, N_FEATURES, VITAL_BOUNDS,
                       LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES, CLINICAL_MEDIANS)
from .layers import (FocalLoss, GhostBatchNorm, GLUBlock,
                     AttentiveTransformer, FeatureTransformer, TabNet)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()

_DEVICE_LABEL: dict = {
    "mps":  "GPU (unified memory)",
    "cuda": f"NVIDIA CUDA — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU'}",
    "cpu":  "CPU (no GPU available)",
}
print(f"[ENGINE] Compute device: {_DEVICE_LABEL.get(DEVICE.type, DEVICE.type)}")

CHECKPOINT_PATH = _data_path("hearth_tabnet.pth")


def normalize_vitals_tanaka(reading: dict) -> dict:
    act = reading.get("activity")
    if isinstance(act, str):
        is_active = act.strip().lower() == "active"
        intensity_scale = 1.0
    else:
        is_active = isinstance(act, (int, float)) and act >= 3
        intensity_scale = 0.7 + (act - 3) * 0.20 if is_active else 0.0
    if not is_active:
        return reading
    normalized = reading.copy()
    hr  = reading.get("heart_rate")
    sbp = reading.get("systolic_bp")
    if hr  is not None and not math.isnan(hr):
        normalized["heart_rate"]  = max(25.0, hr  - EXERTION_BIAS_HR  * intensity_scale)
    if sbp is not None and not math.isnan(sbp):
        normalized["systolic_bp"] = max(50.0, sbp - EXERTION_BIAS_SBP * intensity_scale)
    return normalized



class DeviceRingBuffer:
    
    N_BASE_FEATURES = N_VITALS + 1
    IDX_HR   = 0
    IDX_SPO2 = 4
    
    def __init__(self, buffer_size: int = RING_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self._buffers: Dict[str, Tuple[torch.Tensor, int]] = {}
        self._lock = threading.Lock()
        fallback_values = [
            _normalize(CLINICAL_MEDIANS[k], k) if k in VITAL_BOUNDS
            else (float(CLINICAL_MEDIANS.get(k, 0.0)) / 5.0 if k == "activity"
                  else float(CLINICAL_MEDIANS.get(k, 0.0)))
            for k in ["heart_rate", "systolic_bp", "diastolic_bp", "body_temp", "spo2", "activity"]
        ]
        self._fallback = torch.tensor(fallback_values, dtype=torch.float32, device=DEVICE)

    def _get_or_create_buffer(self, patient_id: str) -> Tuple[torch.Tensor, int]:
        if patient_id not in self._buffers:
            buffer = torch.full((self.buffer_size, self.N_BASE_FEATURES), float("nan"),
                                dtype=torch.float32, device=DEVICE)
            self._buffers[patient_id] = (buffer, 0)
        return self._buffers[patient_id]

    def push_and_impute(self, patient_id: str, reading: torch.Tensor) -> torch.Tensor:
        with self._lock:
            buffer, pos = self._get_or_create_buffer(patient_id)

            # push new reading into circular buffer
            base_reading = reading[:self.N_BASE_FEATURES]
            buffer[pos] = base_reading
            self._buffers[patient_id] = (buffer, (pos + 1) % self.buffer_size)

            # fill NaN gaps with rolling median from this patient's history
            medians = torch.nanmedian(buffer, dim=0).values
            medians = torch.where(torch.isnan(medians), self._fallback, medians)

            nan_mask = torch.isnan(base_reading)
            imputed_base = torch.where(nan_mask, medians, base_reading)

            # delta features: deviation from patient's recent baseline, normalized to match training
            # training uses raw BPM delta / 50.0 and raw SpO2 delta / 10.0
            # VITAL_BOUNDS for HR: [20, 220] → range 200; for SpO2: [50, 100] → range 50
            # normalized delta × range gives approximate raw delta → divide by training scale
            delta_hr   = (imputed_base[self.IDX_HR]   - medians[self.IDX_HR])   * 200.0 / 50.0
            delta_spo2 = (imputed_base[self.IDX_SPO2] - medians[self.IDX_SPO2]) * 50.0  / 10.0
            
            return torch.cat([imputed_base, torch.tensor([delta_hr, delta_spo2], device=DEVICE)])

    def batch_impute(self, patient_id: str, readings: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.push_and_impute(patient_id, readings[i])
                            for i in range(readings.size(0))])

    def clear_patient(self, patient_id: str):
        with self._lock:
            self._buffers.pop(patient_id, None)



def derive_severity(reading: dict) -> str:
    def _f(key):
        v = reading.get(key)
        if v is None: return None
        try:
            fv = float(v)
            return None if math.isnan(fv) else fv
        except (TypeError, ValueError):
            return None

    hr   = _f("heart_rate")  or CLINICAL_MEDIANS["heart_rate"]
    sbp  = _f("systolic_bp") or CLINICAL_MEDIANS["systolic_bp"]
    temp = _f("body_temp")   or CLINICAL_MEDIANS["body_temp"]
    spo2 = _f("spo2")        or CLINICAL_MEDIANS["spo2"]
    act  = reading.get("activity", 0)
    if isinstance(act, str):
        is_active = act.strip().lower() == "active"
    else:
        is_active = isinstance(act, (int, float)) and act >= 3

    score, max_single = news2_score(hr, sbp, temp, spo2, is_active)
    
    if score >= 5 or max_single >= 3:
        base_label = "Critical"
    elif score >= 2:
        base_label = "Unhealthy"
    else:
        base_label = "Healthy"

    # tachycardia + hypotension pattern: only escalate when base is Unhealthy
    eff_hr  = max(hr  - 15.0, 25.0) if is_active else hr
    eff_sbp = max(sbp - 15, 50.0) if is_active else sbp
    if base_label == "Unhealthy" and eff_hr > 90 and eff_sbp < 110:
        return "Critical"
    return base_label


def derive_severity_vectorized(df) -> "pd.Series":
    import pandas as pd

    def _col(name):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(CLINICAL_MEDIANS.get(name, 0))
        return pd.Series(CLINICAL_MEDIANS.get(name, 0), index=df.index)

    hr   = _col("heart_rate")
    sbp  = _col("systolic_bp")
    temp = _col("body_temp")
    spo2 = _col("spo2")

    act_raw   = df.get("activity", pd.Series(0, index=df.index))
    act_numeric = pd.to_numeric(act_raw, errors="coerce").fillna(0)
    act_str     = act_raw.astype(str).str.strip().str.lower()
    is_active = (act_numeric >= 3) | (act_str == "active")

    eff_hr  = np.where(is_active, hr  - 15.0, hr ).clip(25, 220)
    eff_sbp = np.where(is_active, sbp - 15, sbp).clip(50, 280)

    def _hr_score(h):
        s = np.zeros(len(df), dtype=int)
        s = np.where((h <= 40) | (h >= 131), 3, s)
        s = np.where((h >= 111) & (h <= 130), 2, s)
        s = np.where(((h >= 41) & (h <= 50)) | ((h >= 91) & (h <= 110)), 1, s)
        return s

    def _sbp_score(b):
        s = np.zeros(len(df), dtype=int)
        s = np.where((b <= 90) | (b >= 220), 3, s)
        s = np.where((b >= 91) & (b <= 100),  2, s)
        s = np.where((b >= 101) & (b <= 110), 1, s)
        return s

    def _spo2_score(o):
        s = np.zeros(len(df), dtype=int)
        s = np.where(o <= 91, 3, s)
        s = np.where((o >= 92) & (o <= 93), 2, s)
        s = np.where((o >= 94) & (o <= 95), 1, s)
        return s

    def _temp_score(t):
        s = np.zeros(len(df), dtype=int)
        s = np.where(t <= 35.0, 3, s)
        s = np.where((t >= 35.1) & (t <= 36.0), 1, s)
        s = np.where((t >= 38.1) & (t <= 39.0), 1, s)
        s = np.where(t >= 39.1, 2, s)
        return s

    total      = _hr_score(eff_hr) + _sbp_score(eff_sbp) + _spo2_score(spo2) + _temp_score(temp)
    max_single = np.maximum.reduce([_hr_score(eff_hr), _sbp_score(eff_sbp),
                                    _spo2_score(spo2), _temp_score(temp)])

    base_label = pd.Series(np.where(
        (total >= 5) | (max_single >= 3), "Critical",
        np.where(total >= 2, "Unhealthy", "Healthy")
    ), index=df.index)

    eff_hr_v  = np.where(is_active, (hr  - 15.0).clip(lower=25), hr)
    eff_sbp_v = np.where(is_active, (sbp - 15).clip(lower=50), sbp)
    # tachycardia + hypotension: only escalate when NEWS2 already flags concern (total >= 1)
    geri_critical = (base_label == "Unhealthy") & (eff_hr_v > 90) & (eff_sbp_v < 110)
    result = base_label.copy()
    result[geri_critical] = "Critical"
    return result


# min-max scale to [0, 1] using clinical bounds
def _normalize(value: float, vital: str) -> float:
    lo, hi = VITAL_BOUNDS[vital]
    return float(np.clip((value - lo) / (hi - lo + 1e-8), 0.0, 1.0))


def reading_to_vec(reading: dict) -> Optional[np.ndarray]:
    vec = []
    for v in VITALS:
        val = reading.get(v)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            vec.append(float("nan"))
        else:
            try:
                vec.append(_normalize(float(val), v))
            except (TypeError, ValueError):
                vec.append(float("nan"))
    act = reading.get("activity", 0)
    if isinstance(act, str):
        act = 3 if act.strip().lower() == "active" else 0
    try:
        act_norm = float(act) / 5.0
    except (TypeError, ValueError):
        act_norm = 0.0
    vec.append(max(0.0, min(1.0, act_norm)))
    return np.array(vec, dtype=np.float32)


def reading_to_tensor(reading: dict, device: torch.device = DEVICE) -> torch.Tensor:
    return torch.tensor(reading_to_vec(reading), dtype=torch.float32, device=device)


class TabNetEngine:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_PATH
        self.model: Optional[TabNet] = None
        self.is_ready = False
        self._lock = threading.Lock()
        self.ring_buffer = DeviceRingBuffer()
        if os.path.exists(self.checkpoint_path):
            self._load()

    def _load(self):
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE, weights_only=False)
            self.model = TabNet(
                input_dim  = checkpoint.get("input_dim",  N_FEATURES),
                output_dim = checkpoint.get("output_dim", NUM_CLASSES),
                n_d        = checkpoint.get("n_d", 32),
                n_a        = checkpoint.get("n_a", 32),
                n_steps    = checkpoint.get("n_steps", 5),
            ).to(DEVICE)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.is_ready = True

            try:
                with torch.no_grad():
                    _dummy = torch.zeros(1, N_FEATURES, device=DEVICE)
                    self.model(_dummy)
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] Failed to load model: {e}")
            self.is_ready = False

    def _save(self, config: dict = None):
        if self.model is None:
            return
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": N_FEATURES, "output_dim": NUM_CLASSES,
            "n_d": self.model.n_d, "n_a": self.model.n_a, "n_steps": self.model.n_steps,
        }
        if config:
            checkpoint.update(config)
        torch.save(checkpoint, self.checkpoint_path)

    def classify_reading(self, patient_id: str, reading: dict) -> dict:
        raw_tensor = reading_to_tensor(reading)
        imputed    = self.ring_buffer.push_and_impute(patient_id, raw_tensor)

        if not self.is_ready or self.model is None:
            status = derive_severity(reading)
            idx    = LABEL_TO_IDX[status]
            probs  = [0.08, 0.08, 0.08]
            # Use LOW_CONFIDENCE_THRESHOLD + small margin so rule-based predictions
            # are treated the same as model predictions by the downgrade logic
            probs[idx] = LOW_CONFIDENCE_THRESHOLD + 0.05
            feature_names_full = VITALS + ["activity", "delta_hr", "delta_spo2"]
            return {
                "status": status, "confidence": probs[idx],
                "probabilities": {IDX_TO_LABEL[i]: round(probs[i], 4) for i in range(3)},
                "attention": {v: round(1.0 / len(feature_names_full), 4) for v in feature_names_full},
                "model": "rule-based-NEWS2",
            }

        self.model.eval()
        with torch.no_grad():
            x             = imputed.unsqueeze(0)
            status_logits, _, attention = self.model(x, return_attention=True)
            probs         = F.softmax(status_logits, dim=1).squeeze(0)
            idx           = probs.argmax().item()

        feature_names  = VITALS + ["activity"]
        attention_dict = {name: round(float(attention[0, i]), 4) for i, name in enumerate(feature_names)}

        return {
            "status": IDX_TO_LABEL[idx], "confidence": float(probs[idx]),
            "probabilities": {IDX_TO_LABEL[i]: round(float(probs[i]), 4) for i in range(3)},
            "attention": attention_dict, "model": "Hearth Model",
        }

    def classify_batch_fast(self, readings: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(readings)
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        X = np.zeros((n, N_FEATURES), dtype=np.float32)
        for i, r in enumerate(readings):
            for j, vital in enumerate(VITALS):
                val = r.get(vital)
                if val is not None:
                    try:
                        fval = float(val)
                        if not np.isnan(fval):
                            lo, hi = VITAL_BOUNDS[vital]
                            X[i, j] = np.clip((fval - lo) / (hi - lo + 1e-8), 0.0, 1.0)
                        else:
                            X[i, j] = np.nan
                    except (TypeError, ValueError):
                        X[i, j] = np.nan
                else:
                    X[i, j] = np.nan
            act = r.get("activity", 0)
            if isinstance(act, str):
                act = 3 if act.strip().lower() == "active" else 0
            try:
                X[i, N_VITALS] = max(0.0, min(1.0, float(act) / 5.0))
            except (TypeError, ValueError):
                X[i, N_VITALS] = 0.0

        for j in range(N_FEATURES):
            col      = X[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                valid      = col[~nan_mask]
                fill_val   = np.median(valid) if len(valid) > 0 else 0.5
                X[nan_mask, j] = fill_val

        if not self.is_ready or self.model is None:
            return (np.zeros(n, dtype=np.int64),
                    np.full(n, 0.8, dtype=np.float32),
                    np.full((n, N_FEATURES), 1.0 / N_FEATURES, dtype=np.float32))

        X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        self.model.eval()
        with torch.no_grad():
            status_logits, _, att = self.model(X_tensor, return_attention=True)
            probs        = F.softmax(status_logits, dim=1)
            indices      = probs.argmax(dim=1)
            confidences  = probs.gather(1, indices.unsqueeze(1)).squeeze(1)

        return (
            indices.cpu().numpy(),
            confidences.cpu().numpy(),
            att.cpu().numpy() if att is not None else np.full((n, N_FEATURES), 1.0 / N_FEATURES),
        )

    def classify_patient_batch(self, patient_id: str, readings: List[dict]) -> List[dict]:
        if not readings:
            return []
        if not self.is_ready or self.model is None:
            return [self.classify_reading(patient_id, r) for r in readings]

        raw_tensors = torch.stack([reading_to_tensor(r) for r in readings])
        imputed     = self.ring_buffer.batch_impute(patient_id, raw_tensors)

        self.model.eval()
        with torch.no_grad():
            status_logits, risk_logits, attention = self.model(imputed, return_attention=True)
            probs   = F.softmax(status_logits, dim=1)
            indices = probs.argmax(dim=1)

        feature_names = VITALS + ["activity"]
        results = []
        for i in range(len(readings)):
            idx = indices[i].item()
            p   = probs[i]
            att = attention[i] if attention is not None else None
            results.append({
                "status":      IDX_TO_LABEL[idx],
                "confidence":  float(p[idx]),
                "probabilities": {IDX_TO_LABEL[k]: round(float(p[k]), 4) for k in range(3)},
                "attention":   {name: round(float(att[j]), 4) if att is not None else 0.167
                                for j, name in enumerate(feature_names)},
                "model": "Hearth Model",
            })
        return results

    def impute_all_patients(
        self,
        patient_reading_indices: Dict[str, List[int]],
        normalized_readings: List[dict],
    ) -> torch.Tensor:
        n = len(normalized_readings)
        N_BASE = DeviceRingBuffer.N_BASE_FEATURES

        X_np = np.full((n, N_BASE), np.nan, dtype=np.float32)
        for i, r in enumerate(normalized_readings):
            for j, vital in enumerate(VITALS):
                val = r.get(vital)
                if val is not None:
                    try:
                        fval = float(val)
                        if not math.isnan(fval):
                            lo, hi = VITAL_BOUNDS[vital]
                            X_np[i, j] = max(0.0, min(1.0, (fval - lo) / (hi - lo + 1e-8)))
                    except (TypeError, ValueError):
                        pass
            act = r.get("activity", 0)
            if isinstance(act, str):
                act = 3 if act.strip().lower() == "active" else 0
            try:
                X_np[i, N_VITALS] = max(0.0, min(1.0, float(act) / 5.0))
            except (TypeError, ValueError):
                X_np[i, N_VITALS] = 0.0

        X_raw = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)

        imputed_all = torch.zeros((n, N_FEATURES), dtype=torch.float32, device=DEVICE)
        temp_ring_buffer = DeviceRingBuffer()
        for pid, idxs in patient_reading_indices.items():
            for global_i in idxs:
                imputed_all[global_i] = temp_ring_buffer.push_and_impute(
                    pid, X_raw[global_i]
                )

        return imputed_all

    # 7-day lookahead risk prediction
    def predict_risk(self, patient_id: str, window_rows: List[dict]) -> Optional[dict]:
        if len(window_rows) < 7 or not self.is_ready or self.model is None:
            return None

        # daily summary rows use different column names than raw readings
        mapped_rows = []
        for r in window_rows:
            mapped = r.copy()
            if "avg_heart_rate" in mapped and "heart_rate" not in mapped:
                mapped["heart_rate"]   = mapped.pop("avg_heart_rate")
                mapped["systolic_bp"]  = mapped.pop("avg_systolic", None)
                mapped["diastolic_bp"] = mapped.pop("avg_diastolic", None)
                mapped["body_temp"]    = mapped.pop("avg_temp", None)
                mapped["spo2"]         = mapped.pop("avg_spo2", None)
                mapped["activity"]     = mapped.pop("dominant_activity", 0)
            # apply the same exertion bias used by real-time classification
            mapped = normalize_vitals_tanaka(mapped)
            mapped_rows.append(mapped)

        raw_tensors = torch.stack([reading_to_tensor(r) for r in mapped_rows])

        imputed = self.ring_buffer.batch_impute(patient_id, raw_tensors)

        self.model.eval()
        with torch.no_grad():
            status_logits, risk_logits, attention = self.model(imputed, return_attention=True)
            risk_probs = torch.sigmoid(risk_logits.squeeze(-1))
            risk_prob  = float(risk_probs.mean().item())
            peak_day   = int(risk_probs.argmax().item())
            peak_score = float(risk_probs.max().item())

        feature_names = VITALS + ["activity", "delta_hr", "delta_spo2"]
        att = attention[peak_day].cpu().numpy() if attention is not None else np.zeros(N_FEATURES)
        top_factors = [feature_names[i] for i in np.argsort(att)[-3:][::-1]]

        return {
            "risk_label":  "HIGH RISK" if risk_prob >= RISK_LABEL_THRESHOLD else "LOW RISK",
            "risk_score":  round(risk_prob, 4),
            "peak_score":  round(peak_score, 4),
            "peak_day":    peak_day,
            "model":       "Hearth Model",
            "top_factors": top_factors,
        }

    def train_from_db(self, max_samples: int = 500_000,
                      epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE):
        import pandas as pd
        from data import logger as data_logger

        print("\n" + "=" * 60)
        print("  HEARTH AI — Model Training")
        print("=" * 60)
        print(f"  Device: {_DEVICE_LABEL.get(DEVICE.type, DEVICE.type)}")
        print(f"  Batch Size: {batch_size}")

        print("\n[1/4] Loading training data...")
        dates = data_logger.get_dates_available()
        if not dates:
            print("[ERROR] No data in database. Generate data first.")
            return

        all_readings = []
        for date in dates[:60]:
            all_readings.extend(data_logger.get_readings_for_date(date))
            if len(all_readings) >= max_samples:
                break

        if len(all_readings) < 1000:
            print(f"[ERROR] Insufficient data ({len(all_readings)} readings).")
            return
        print(f"       Loaded {len(all_readings):,} readings")


        print("\n[2/4] Building feature matrix with temporal deltas...")
        df = pd.DataFrame(all_readings)
        df["status"] = derive_severity_vectorized(df)

        df = df.reset_index(drop=True)
        y_risk_arr = np.zeros(len(df), dtype=np.float32)
        for _, grp in df.groupby("patient_id", sort=False):
            idxs     = grp.index.tolist()
            statuses = grp["status"].values
            for ii, orig_idx in enumerate(idxs):
                lookahead = statuses[ii + 1 : ii + 8]
                if "Critical" in lookahead:
                    y_risk_arr[orig_idx] = 1.0

        ROLLING_WINDOW = 10
        
        delta_hr_arr = np.zeros(len(df), dtype=np.float32)
        delta_spo2_arr = np.zeros(len(df), dtype=np.float32)

        for pid, grp in df.groupby("patient_id", sort=False):
            idxs = grp.index.tolist()
            hr_vals   = grp["heart_rate"].fillna(CLINICAL_MEDIANS["heart_rate"]).values
            spo2_vals = grp["spo2"].fillna(CLINICAL_MEDIANS["spo2"]).values
            
            for i, orig_idx in enumerate(idxs):
                start = max(0, i - ROLLING_WINDOW + 1)
                hr_window   = hr_vals[start:i+1]
                spo2_window = spo2_vals[start:i+1]
                
                hr_median   = float(np.median(hr_window))
                spo2_median = float(np.median(spo2_window))
                
                delta_hr_arr[orig_idx]   = hr_vals[i] - hr_median
                delta_spo2_arr[orig_idx] = spo2_vals[i] - spo2_median

        df["delta_hr"]   = delta_hr_arr
        df["delta_spo2"] = delta_spo2_arr

        _BASE_COLS = VITALS
        _ROLLING_IMP_WINDOW = RING_BUFFER_SIZE
        _nan_filled_total = 0

        for pid, grp in df.groupby("patient_id", sort=False):
            idxs = grp.index.tolist()
            for col_name in _BASE_COLS:
                vals = grp[col_name].values.copy().astype(float)
                for i in range(len(vals)):
                    if np.isnan(vals[i]):
                        start = max(0, i - _ROLLING_IMP_WINDOW + 1)
                        window = vals[start:i]
                        valid_in_window = window[~np.isnan(window)]
                        if len(valid_in_window) > 0:
                            vals[i] = float(np.median(valid_in_window))
                        else:
                            vals[i] = CLINICAL_MEDIANS.get(col_name, 0.0)
                        _nan_filled_total += 1
                df.loc[idxs, col_name] = vals

        if _nan_filled_total:
            print(f"       Per-patient rolling-median imputation: "
                  f"{_nan_filled_total:,} NaN cells filled")

        X_list, y_status_list, y_risk_list = [], [], []
        records = df.to_dict("records")
        for orig_idx, row in enumerate(records):
            vec = reading_to_vec(row)
            if vec is not None:
                delta_hr_norm   = row["delta_hr"] / 50.0
                delta_spo2_norm = row["delta_spo2"] / 10.0
                full_vec = np.concatenate([vec, [delta_hr_norm, delta_spo2_norm]])
                X_list.append(full_vec)
                y_status_list.append(LABEL_TO_IDX.get(row["status"], 0))
                y_risk_list.append(y_risk_arr[orig_idx])

        X        = np.vstack(X_list)
        y_status = np.array(y_status_list, dtype=np.int64)
        y_risk   = np.array(y_risk_list,   dtype=np.float32)

        _feature_keys = VITALS + ["activity", "delta_hr", "delta_spo2"]
        _norm_fallback = np.array([
            _normalize(CLINICAL_MEDIANS[k], k) if k in VITAL_BOUNDS
            else float(CLINICAL_MEDIANS.get(k, 0.0))
            for k in _feature_keys
        ], dtype=np.float32)

        nan_cols_filled = 0
        for j in range(N_FEATURES):
            col      = X[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                X[nan_mask, j] = float(_norm_fallback[j])
                nan_cols_filled += 1

        if nan_cols_filled:
            print(f"       Final NaN safety net: {nan_cols_filled} feature(s) "
                  f"still had gaps (filled with CLINICAL_MEDIANS fallback)")

        print(f"       Features: {X.shape[0]:,} samples × {X.shape[1]} features")

        unique, counts = np.unique(y_status, return_counts=True)
        print("       Class distribution:")
        for idx, cnt in zip(unique, counts):
            print(f"         {IDX_TO_LABEL[idx]}: {cnt:,} ({cnt/len(y_status)*100:.1f}%)")

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_status_train, y_status_test, y_risk_train, y_risk_test = train_test_split(
            X, y_status, y_risk, test_size=0.2, random_state=42, stratify=y_status
        )

        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.pipeline import Pipeline as ImbPipeline

            unique_tr, counts_tr = np.unique(y_status_train, return_counts=True)
            max_minority = int(max(counts_tr[unique_tr != 0]) if len(counts_tr) > 1 else counts_tr[0])
            healthy_target = max(max_minority * 2, int(counts_tr[unique_tr == 0][0] * 0.4)) if 0 in unique_tr else max_minority * 2
            minority_target = max(max_minority, int(healthy_target * 0.5))

            over_strategy = {}
            under_strategy = {}
            for idx, cnt in zip(unique_tr, counts_tr):
                if idx == 0:
                    under_strategy[idx] = min(int(cnt), healthy_target)
                else:
                    over_strategy[idx] = max(int(cnt), minority_target)

            steps = []
            if over_strategy:
                steps.append(('over', SMOTE(sampling_strategy=over_strategy, random_state=42, k_neighbors=min(5, min(counts_tr[unique_tr != 0]) - 1) if min(counts_tr[unique_tr != 0]) > 1 else 1)))
            if under_strategy:
                steps.append(('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)))

            if steps:
                X_train_aug = np.column_stack([X_train, y_risk_train])
                pipeline = ImbPipeline(steps=steps)
                X_resampled, y_status_train = pipeline.fit_resample(X_train_aug, y_status_train)
                X_train      = X_resampled[:, :-1]
                y_risk_train = X_resampled[:, -1].astype(np.float32)

                unique_rs, counts_rs = np.unique(y_status_train, return_counts=True)
                print("       Class distribution after resampling:")
                for idx, cnt in zip(unique_rs, counts_rs):
                    print(f"         {IDX_TO_LABEL[idx]}: {cnt:,} ({cnt/len(y_status_train)*100:.1f}%)")
        except ImportError:
            print("       [WARN] imbalanced-learn not installed — skipping SMOTE. "
                  "Install with: pip install imbalanced-learn")
        except Exception as e:
            print(f"       [WARN] SMOTE resampling failed ({e}), continuing with original data.")

        _pin = DEVICE.type == "cuda"
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train,        dtype=torch.float32),
                torch.tensor(y_status_train, dtype=torch.long),
                torch.tensor(y_risk_train,   dtype=torch.float32),
            ),
            batch_size=batch_size, shuffle=True, pin_memory=_pin, num_workers=0,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test,        dtype=torch.float32),
                torch.tensor(y_status_test, dtype=torch.long),
                torch.tensor(y_risk_test,   dtype=torch.float32),
            ),
            batch_size=batch_size, shuffle=False, pin_memory=_pin, num_workers=0,
        )

        print("\n[3/4] Initializing model...")
        self.model = TabNet(
            input_dim=N_FEATURES, output_dim=NUM_CLASSES,
            n_d=32, n_a=32, n_steps=5, gamma=1.5, virtual_batch_size=256,
        ).to(DEVICE)

        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        print("\n[4/4] Training with Focal Loss...")
        best_recall = 0.0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_x, batch_y_status, batch_y_risk in train_loader:
                batch_x        = batch_x.to(DEVICE)
                batch_y_status = batch_y_status.to(DEVICE)
                batch_y_risk   = batch_y_risk.to(DEVICE)
                optimizer.zero_grad()
                status_logits, risk_logits, _ = self.model(batch_x)
                loss_status = criterion(status_logits, batch_y_status)
                loss_risk   = F.binary_cross_entropy_with_logits(risk_logits.squeeze(-1), batch_y_risk)
                loss        = loss_status + loss_risk
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            self.model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch_x, batch_y_status, _ in test_loader:
                    status_logits, _, _ = self.model(batch_x.to(DEVICE))
                    all_preds.extend(status_logits.argmax(dim=1).cpu().numpy())
                    all_labels.extend(batch_y_status.numpy())

            all_preds  = np.array(all_preds)
            all_labels = np.array(all_labels)
            accuracy   = (all_preds == all_labels).mean()

            critical_mask   = all_labels == 2
            critical_recall = (all_preds[critical_mask] == 2).mean() if critical_mask.sum() > 0 else 0.0
            avg_loss        = total_loss / len(train_loader)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"       Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"Acc: {accuracy*100:.1f}% | Critical Recall: {critical_recall*100:.1f}%")

            if critical_recall > best_recall:
                best_recall = critical_recall
                self._save({"best_critical_recall": best_recall})

            scheduler.step(critical_recall)

        self._load()
        self.is_ready = True

        self.model.eval()
        train_preds_all, train_labels_all = [], []
        with torch.no_grad():
            for batch_x, batch_y_status, _ in train_loader:
                status_logits, _, _ = self.model(batch_x.to(DEVICE))
                train_preds_all.extend(status_logits.argmax(dim=1).cpu().numpy())
                train_labels_all.extend(batch_y_status.numpy())
        train_preds_arr  = np.array(train_preds_all)
        train_labels_arr = np.array(train_labels_all)
        train_accuracy   = (train_preds_arr == train_labels_arr).mean()

        self.model.eval()
        att_sum   = np.zeros(N_FEATURES, dtype=np.float64)
        att_count = 0
        with torch.no_grad():
            for batch_x, _, _ in test_loader:
                _, _, att_batch = self.model(batch_x.to(DEVICE), return_attention=True)
                if att_batch is not None:
                    att_sum   += att_batch.cpu().numpy().sum(axis=0)
                    att_count += att_batch.shape[0]
        feature_names_full = VITALS + ["activity", "delta_hr", "delta_spo2"]
        if att_count > 0:
            avg_importance = att_sum / att_count
        else:
            avg_importance = np.ones(N_FEATURES) / N_FEATURES

        from sklearn.metrics import (
            confusion_matrix, classification_report,
            precision_score, recall_score, f1_score,
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        class_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

        cm       = confusion_matrix(all_labels, all_preds)
        prec_w   = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        rec_w    = recall_score(all_labels,    all_preds, average="weighted", zero_division=0)
        f1_w     = f1_score(all_labels,        all_preds, average="weighted", zero_division=0)

        prec_per = precision_score(all_labels, all_preds, average=None,       zero_division=0)
        rec_per  = recall_score(all_labels,    all_preds, average=None,       zero_division=0)

        healthy_idx = LABEL_TO_IDX["Healthy"]
        tp_h = cm[healthy_idx, healthy_idx]
        fn_h = cm[healthy_idx, :].sum() - tp_h
        fp_h = cm[:, healthy_idx].sum() - tp_h
        tn_h = cm.sum() - tp_h - fn_h - fp_h
        healthy_specificity = tn_h / (tn_h + fp_h) if (tn_h + fp_h) > 0 else 0.0

        critical_idx    = LABEL_TO_IDX["Critical"]
        overall_crit_recall = rec_per[critical_idx]

        test_accuracy  = accuracy
        overfit_gap    = train_accuracy - test_accuracy

        print("\n" + "=" * 60)
        print("  HEARTH AI — Post-Training Evaluation Report")
        print("=" * 60)

        print("\n  [Accuracy]")
        print(f"    Training Accuracy:   {train_accuracy*100:.2f}%")
        print(f"    Testing  Accuracy:   {test_accuracy*100:.2f}%")
        print(f"    Overall  Accuracy:   {test_accuracy*100:.2f}%")
        print(f"    Overfitting Gap:     {overfit_gap*100:+.2f}%  "
              f"({'minimal' if abs(overfit_gap) < 0.05 else 'moderate' if abs(overfit_gap) < 0.10 else 'significant'})")

        print(f"\n  [Weighted Metrics]")
        print(f"    Weighted F1 Score:   {f1_w:.4f}")
        print(f"    Weighted Precision:  {prec_w:.4f}")
        print(f"    Weighted Recall:     {rec_w:.4f}")

        print(f"\n  [Per-Class Precision & Recall]")
        print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10}  Note")
        print(f"  {'-'*50}")
        for i, cname in enumerate(class_names):
            note = ""
            if prec_per[i] == max(prec_per) and rec_per[i] == max(rec_per):
                note = "<-- most reliable"
            elif prec_per[i] == min(prec_per) or rec_per[i] == min(rec_per):
                note = "<-- least reliable"
            print(f"  {cname:<12} {prec_per[i]:>10.4f} {rec_per[i]:>10.4f}  {note}")

        combined = prec_per + rec_per
        most_reliable  = class_names[int(np.argmax(combined))]
        least_reliable = class_names[int(np.argmin(combined))]
        print(f"\n    Most  Reliable Class: {most_reliable}")
        print(f"    Least Reliable Class: {least_reliable}")

        print(f"\n  [Healthy Class Specificity]")
        print(f"    Specificity:         {healthy_specificity:.4f}  "
              f"(TN={tn_h:,}  FP={fp_h:,})")

        print(f"\n  [Overall Critical Recall]")
        print(f"    Critical Recall:     {overall_crit_recall*100:.2f}%  "
              f"(best across all epochs: {best_recall*100:.2f}%)")

        print(f"\n  [Feature Importance]")
        sorted_feats = sorted(zip(feature_names_full, avg_importance),
                              key=lambda x: x[1], reverse=True)
        print(f"  {'Feature':<16} {'Importance':>10}")
        print(f"  {'-'*30}")
        for fname, fimp in sorted_feats:
            print(f"  {fname:<16} {fimp:>10.4f}")

        print(f"\n  [False Alarm Rate  (FAR = FP / (FP + TN), per class)]")
        print(f"  {'Class':<12} {'FAR':>8} {'FP':>8} {'TN':>10}  Interpretation")
        print(f"  {'-'*58}")
        for i, cname in enumerate(class_names):
            tp_i = cm[i, i]
            fn_i = cm[i, :].sum() - tp_i
            fp_i = cm[:, i].sum() - tp_i
            tn_i = cm.sum() - tp_i - fn_i - fp_i
            far_i = fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0.0
            interp = (
                "very low false alarm" if far_i < 0.02 else
                "low false alarm"      if far_i < 0.05 else
                "moderate false alarm" if far_i < 0.10 else
                "HIGH false alarm rate"
            )
            print(f"  {cname:<12} {far_i:>8.4f} {fp_i:>8,} {tn_i:>10,}  {interp}")

        print("\n" + "=" * 60)
        print("  Training Complete!")
        print("=" * 60)


_engine_instance: Optional[TabNetEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> TabNetEngine:
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = TabNetEngine()
    return _engine_instance


TRIAGE_SEQ_LEN = 96
SEQ_LEN        = RING_BUFFER_SIZE


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Hearth Model")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch-size",  type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=500000)
    args = parser.parse_args()
    get_engine().train_from_db(
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
