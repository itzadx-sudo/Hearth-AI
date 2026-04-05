# Central configuration for Hearth AI — all tunable constants live here.

# network
SERVER_HOST         = "127.0.0.1"
SERVER_PORT         = 65432
DEVICE_ADAPTER_PORT = 65431
DASHBOARD_PORT      = 8050
MAX_MSG_BYTES       = 10 * 1024 * 1024   # 10 MB hard cap on incoming TCP frames

# inference
PREDICT_EVERY_N          = 1     # run inference every N ticks
CRITICAL_CONF_THRESHOLD  = 0.75  # minimum confidence to fire a critical alert
LOW_CONFIDENCE_THRESHOLD = 0.55  # below this, Critical is downgraded to Unhealthy

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
