# Model architecture and clinical reference constants — fixed medical/ML values.
# Operational settings (ports, thresholds, training params) live in config.py.

from typing import Dict, List, Tuple

# vital sign feature set
VITALS: List[str] = ["heart_rate", "systolic_bp", "diastolic_bp", "body_temp", "spo2"]
N_VITALS           = len(VITALS)
N_FEATURES         = N_VITALS + 3   # + activity, delta_hr, delta_spo2

# clinical normalisation bounds (min-max scaling to [0, 1])
VITAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "heart_rate":   (20.0,  220.0),
    "systolic_bp":  (50.0,  280.0),
    "diastolic_bp": (25.0,  160.0),
    "body_temp":    (33.0,   43.0),
    "spo2":         (50.0,  100.0),
}

# classification labels
LABEL_TO_IDX: Dict[str, int] = {"Healthy": 0, "Unhealthy": 1, "Critical": 2}
IDX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES = 3

# population-level fallback medians (used when no patient history exists)
CLINICAL_MEDIANS: Dict[str, float] = {
    "heart_rate":   72.0,
    "systolic_bp":  120.0,
    "diastolic_bp": 80.0,
    "body_temp":    36.6,
    "spo2":         97.0,
    "activity":     1.5,
    "delta_hr":     0.0,
    "delta_spo2":   0.0,
}

def news2_score(hr: float, sbp: float, temp: float, spo2: float,
                is_active: bool = False,
                exertion_bias_hr: float = 15.0,
                exertion_bias_sbp: float = 15.0) -> Tuple[int, int]:
    eff_hr  = max(hr  - exertion_bias_hr, 25.0) if is_active else hr
    eff_sbp = max(sbp - exertion_bias_sbp, 50.0) if is_active else sbp
    score, max_single = 0, 0

    if eff_hr <= 40 or eff_hr >= 131:
        s = 3
    elif 111 <= eff_hr <= 130:
        s = 2
    elif (41 <= eff_hr <= 50) or (91 <= eff_hr <= 110):
        s = 1
    else:
        s = 0
    score += s
    max_single = max(max_single, s)

    if spo2 <= 91:
        s = 3
    elif 92 <= spo2 <= 93:
        s = 2
    elif 94 <= spo2 <= 95:
        s = 1
    else:
        s = 0
    score += s
    max_single = max(max_single, s)

    if eff_sbp <= 90 or eff_sbp >= 220:
        s = 3
    elif 91 <= eff_sbp <= 100:
        s = 2
    elif 101 <= eff_sbp <= 110:
        s = 1
    else:
        s = 0
    score += s
    max_single = max(max_single, s)

    if temp <= 35.0:
        s = 3
    elif 35.1 <= temp <= 36.0:
        s = 1
    elif 38.1 <= temp <= 39.0:
        s = 1
    elif temp >= 39.1:
        s = 2
    else:
        s = 0
    score += s
    max_single = max(max_single, s)

    return score, max_single
