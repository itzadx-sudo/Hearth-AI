# Central configuration for Hearth AI — all tunable constants live here.

# network
SERVER_HOST         = "127.0.0.1"
SERVER_PORT         = 65432
DEVICE_ADAPTER_PORT = 65431
DASHBOARD_PORT      = 8050
MAX_MSG_BYTES       = 10 * 1024 * 1024   # 10 MB hard cap on incoming TCP frames

# inference
PREDICT_EVERY_N          = 1     # run inference every N ticks
LOW_CONFIDENCE_THRESHOLD = 0.55  # below this, Critical is downgraded to Unhealthy
CRITICAL_CONF_THRESHOLD  = LOW_CONFIDENCE_THRESHOLD  # highlight all non-downgraded Criticals
RISK_LABEL_THRESHOLD     = 0.55  # risk_score >= this → "HIGH RISK" label (matches confidence threshold)

# context metric severity thresholds (used in server/api.py get_patient_context_metrics)
CTX_HIGH_HR_WARN   = 1;  CTX_HIGH_HR_DANGER   = 3   # resting_high_hr_days
CTX_LOW_SPO2_WARN  = 1;  CTX_LOW_SPO2_DANGER  = 3   # low_spo2_days
CTX_FEVER_WARN     = 1;  CTX_FEVER_DANGER      = 3   # fever_days
CTX_DANGER_WARN    = 1;  CTX_DANGER_DANGER     = 2   # spo2_temp_danger_days
CTX_CRIT_ESC_WARN  = 0.0; CTX_CRIT_ESC_DANGER  = 0.5  # critical_escalation slope
CTX_ACT_DECLINE_WARN = -0.05; CTX_ACT_DECLINE_DANGER = -0.10  # activity_decline
CTX_HR_MISMATCH_WARN = 80;  CTX_HR_MISMATCH_DANGER = 100     # hr_activity_mismatch
CTX_CRIT_RATIO_WARN  = 0.05; CTX_CRIT_RATIO_DANGER  = 0.15  # critical_ratio
CTX_CONF_TREND_WARN  = -0.03; CTX_CONF_TREND_DANGER = -0.10  # confidence_trend

# tanaka exertion normalisation
EXERTION_BIAS_HR  = 15.0   # HR reduction applied when patient is active
EXERTION_BIAS_SBP = 15.0   # systolic BP reduction applied when patient is active

# ring buffer
RING_BUFFER_SIZE = 10   # per-patient rolling window depth for NaN imputation

# alert queue & debounce
MAX_ALERTS         = 100  # in-memory alert queue capacity
DEBOUNCE_THRESHOLD = 3    # consecutive critical ticks required before firing alert

# sudden-change detection (day-over-day deltas)
HR_CHANGE_THRESHOLD     = 20
SPO2_CHANGE_THRESHOLD   = 5
TEMP_CHANGE_THRESHOLD   = 1.0
SYS_BP_CHANGE_THRESHOLD = 25

# 7-day risk predictor
WINDOW_DAYS               = 7
LOOKAHEAD_DAYS            = 2
RISK_CONFIDENCE_THRESHOLD = 0.55

# training defaults
DEFAULT_EPOCHS       = 30
DEFAULT_BATCH_SIZE   = 2048
DEFAULT_LR           = 0.02
DEFAULT_WEIGHT_DECAY = 1e-5
