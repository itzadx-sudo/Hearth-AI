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
