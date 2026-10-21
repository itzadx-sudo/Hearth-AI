# Configuration and runtime data

[Back to README](../README.md)

## Defaults

`config.py` contains network defaults, inference thresholds, ring-buffer settings, context thresholds, and training defaults:

```python
SERVER_PORT              = 65432
DEVICE_ADAPTER_PORT      = 65431
LOW_CONFIDENCE_THRESHOLD = 0.55   # live Critical predictions below this become Unhealthy
CRITICAL_CONF_THRESHOLD  = 0.55   # confidence floor for highlighting Critical patients
RISK_LABEL_THRESHOLD     = 0.55   # model risk score threshold for HIGH RISK
PREDICT_EVERY_N          = 1      # live risk inference cadence after seven ticks
DEBOUNCE_THRESHOLD       = 3     # consecutive live Critical ticks per alert
WINDOW_DAYS              = 7     # historical lookback requested by the predictor wrapper
```

## Configuration limits

`WINDOW_DAYS` controls the predictor wrapper; the live risk window is fixed at seven ticks. Several dashboard risk highlights use a fixed threshold of `0.5`. The launcher prompts for its dashboard port instead of using `DASHBOARD_PORT`. `LOOKAHEAD_DAYS` is unused in training-label generation.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `HEARTH_LIVE_MODE` | Set to `1` for live server storage and live simulation; the launcher sets it automatically |
| `LIVE_N_PATIENTS` | Live simulator patient count, default `50`; set from the launcher prompt |
| `LIVE_TICK_SECONDS` | Live simulator tick interval, default `2.0`; set from the launcher prompt |
| `HEARTH_SENSOR_DB` | Override the sensor database used by `data/logger.py` and the trainer |
| `HEARTH_SESSION_KEY` | Flask session signing key; otherwise a random key is generated on startup |

`constants.py` defines vital bounds, label mapping, fallback medians, and NEWS2-style severity rules.

## Runtime data

`paths.py` creates a persistent data directory and copies bundled files there only when no runtime copy exists:

| Platform | Default data directory |
| --- | --- |
| macOS | `~/Library/Application Support/HearthAI/` |
| Windows | `%APPDATA%\HearthAI\` (home directory fallback if `APPDATA` is unset) |
| Linux / other | `~/.hearthai/` |

Runtime files include `hearth_tabnet.pth`, `hearth_sensor.db`, `hearth_results.db`, `hearth_live.db`, and `user.db`. Live readings, predictions, and alert events use `hearth_live.db`; historical summaries, predictions, and alerts use `hearth_results.db`; authentication uses `user.db`. An existing runtime checkpoint takes precedence over the bundled checkpoint.

The standalone generator uses files under `data/`, and dashboard retraining generates `hearth_sensor.db` in the repository root. These paths differ from the default persistent sensor database.

See [Training](training.md#dashboard-retraining) for dashboard retraining paths.
