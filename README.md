# Hearth AI

A home-care patient monitoring demo that streams synthetic vitals over TCP, uses a custom TabNet model to classify patient status (Healthy / Unhealthy / Critical), and displays live charts, alerts, and rolling risk scores through a Flask web dashboard.

---

## What it does

- **Live monitoring** — an IoT simulator streams heart rate, blood pressure, temperature, SpO2, and activity at a configurable interval
- **TabNet inference** — a custom dual-head model classifies patient status; its risk head is also used for rolling risk estimates. If the model cannot load, the server uses a NEWS2-style rule-based status fallback
- **Live alerts** — the dashboard derives critical and predictive alerts from stored patient states and risk predictions. A separate debounced alert helper exists, but the live server does not call it
- **Rolling risk estimates** — live predictions begin after seven ticks and use the latest seven readings. Historical predictions use daily summaries. The current implementation does not establish a seven-day forecast horizon
- **Web dashboard** — Flask-served dashboard with live charts, alert feed, and per-patient drill-down
- **Explainability** — averaged TabNet attention weights and top features are exposed for inspecting model attention
- **Accounts and roles** — guardian signup/login, admin demo access, patient assignment, and display-name editing
- **Admin retraining** — a dashboard action generates synthetic training data and starts model training, with a status endpoint and temporary API downtime; see the current limitations below

---

## Architecture

```
main.py                  launcher — starts the AI server, simulator, and dashboard
│
├── model/
│   ├── layers.py        FocalLoss, GhostBatchNorm, GLUBlock, AttentiveTransformer, TabNet
│   ├── engine.py        training loop, checkpoint management, normalisation
│   └── predictor.py     historical feature summaries + wrapper around the shared engine
│
├── server/
│   ├── ai_server.py     TCP server — receives IoT frames, runs inference, writes results
│   ├── alert_engine.py  async alert queue with debounce logic
│   ├── api.py           validation, dashboard data, and prediction helpers
│   └── device_adapter.py  bridges raw device TCP to the AI server format
│
├── data/
│   ├── logger.py        SQLite read/write layer for live and historical data
│   └── generator.py     synthetic patient data generator for training
│
├── iot/
│   └── simulator.py     simulates sensor devices sending vitals over TCP
│
├── auth/
│   └── db.py            SQLite-backed accounts and guardian/patient assignments
│
├── ui/
│   └── gui.py           Flask app — serves the dashboard, exposes the API
│
├── config.py            operational defaults and thresholds
├── constants.py         feature names, label map, bounds, medians, and severity rules
├── paths.py             asset paths and persistent runtime data directory
├── hearth_landing.html  landing page
├── hearth_dashboard.html  dashboard and client-side API calls
├── static/              logo asset
├── hearth_tabnet.pth     bundled model checkpoint
└── HearthAI.spec         PyInstaller specification (currently uses stale module paths)
```

---

## Setup

**Requirements:** Python 3.11+, pip. The pinned NumPy 2.4.4 and pandas 3.0.2 releases require Python 3.11 or newer.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `flask`, `matplotlib`, `seaborn`, `pydantic`

The dashboard loads Chart.js from jsDelivr and fonts from Google Fonts, so those assets need internet access. The engine selects Apple MPS, then CUDA, then CPU according to availability.

---

## Running

**Current launch blocker:** `main.py` references `VERSION` without defining or importing it. The command below currently raises `NameError` before any services start. The following describes the launch flow once that code issue is resolved.

```bash
python main.py
```

The repository includes `hearth_tabnet.pth`. When no runtime copy exists, `paths.py` copies the bundled checkpoint into the persistent data directory. The launcher prompts for training only if the checkpoint file is absent:

```
[T] Train model now
[Q] Quit

Number of patients  (default 200):
Number of days      (default 60):
Sensor reads/hour   (default 12):
```

With an existing checkpoint, or after the inline training step, the launcher asks:

```text
Live patients  [1-500]  (default 50):
Tick rate (s)  [0.5-30] (default 2.0):
Dashboard port          (default 8050):
```

It starts the AI server, waits for its TCP port to accept connections, starts the live simulator, and serves the dashboard in a background thread. The browser opens `http://localhost:8050/landing` by default; the dashboard itself is at `http://localhost:8050/`. A custom port replaces `8050` in both URLs.

Press **Ctrl+C** in the launcher terminal to stop the session. The Flask app binds to `0.0.0.0`, while the AI server defaults to `127.0.0.1`.

### Dashboard access

- Use **Admin Demo** to log in to the built-in demo account (`admin` / `adminpassword`); the browser creates it if needed. Admin controls include guardian assignment and retraining.
- Create a **Guardian** account using signup, then have an admin assign patient IDs to it. Guardian filtering is applied to the main patient and alert views, but is not enforced consistently across all API routes.
- Display names can be edited through the dashboard. The password-reset button currently displays a message only; no reset email is sent.

### Module commands

Additional module entry points exist. Run them from the repository root with `python -m`:

```bash
# Dashboard only; does not start the AI server or simulator
python -m ui.gui --port 8050 --no-browser

# Train from the configured sensor database
python -m model.engine --epochs 30 --batch-size 2048 --max-samples 500000

# Interactive synthetic training/test data generation
python -m data.generator
```

The standalone generator writes `data/hearth_sensor.db` or `data/hearth_test.db`, while the engine normally reads the persistent sensor database. To train on the standalone generator's training output, set `HEARTH_SENSOR_DB` to its path before starting the trainer:

```bash
# macOS/Linux
HEARTH_SENSOR_DB="$PWD/data/hearth_sensor.db" python -m model.engine
```

Training requires at least 1,000 readings. Inline generation through the launcher uses the configured sensor database instead.

---

## Model

The classifier is a from-scratch TabNet implementation — no external TabNet library. Key design choices:

- **Dual head** — status logits are converted to three-class probabilities with softmax; risk logits are converted to scores with sigmoid
- **FocalLoss** — down-weights easy examples and applies class weights of 0.1 / 0.3 / 0.6 for Healthy / Unhealthy / Critical
- **GhostBatchNorm** — splits batches into virtual sub-batches to keep BN statistics stable at large batch sizes
- **Attention weights** — inference can request feature attention averaged across the decision steps; the dashboard displays these weights and top features

Input features: heart rate, systolic BP, diastolic BP, body temperature, SpO2, activity level, Δ heart rate, Δ SpO2 (8 features total).

Risk training labels indicate whether a Critical status occurs in the next seven readings for a patient. At inference, `predict_risk()` averages risk-head scores over the supplied window. Live mode supplies seven ticks; historical mode supplies daily summary rows. `model/predictor.py` wraps the same engine rather than loading a separate prediction model. UI labels that say “7-day” or “7-day ahead” should therefore not be interpreted as an implemented seven-day forecast.

---

## Configuration

`config.py` contains network defaults, inference thresholds, ring-buffer settings, context thresholds, and training defaults:

```python
SERVER_PORT              = 65432
DEVICE_ADAPTER_PORT      = 65431
LOW_CONFIDENCE_THRESHOLD = 0.55   # live Critical predictions below this become Unhealthy
CRITICAL_CONF_THRESHOLD  = 0.55   # confidence floor for highlighting Critical patients
RISK_LABEL_THRESHOLD     = 0.55   # model risk score threshold for HIGH RISK
PREDICT_EVERY_N          = 1      # live risk inference cadence after seven ticks
DEBOUNCE_THRESHOLD       = 3     # used by the separate alert helper, not the live server
WINDOW_DAYS              = 7     # historical lookback requested by the predictor wrapper
```

Some behavior remains hardcoded: live risk windows are seven ticks, parts of the dashboard use `0.5` for risk highlighting, and the launcher prompts for its own dashboard port. Changing `WINDOW_DAYS`, `DASHBOARD_PORT`, or `RISK_LABEL_THRESHOLD` does not update every code path. `LOOKAHEAD_DAYS` is defined but is not used to construct the training labels.

Environment variables:

| Variable | Purpose |
| --- | --- |
| `HEARTH_LIVE_MODE` | Set to `1` for live server storage and live simulation; the launcher sets it automatically |
| `LIVE_N_PATIENTS` | Live simulator patient count, default `50`; set from the launcher prompt |
| `LIVE_TICK_SECONDS` | Live simulator tick interval, default `2.0`; set from the launcher prompt |
| `HEARTH_SENSOR_DB` | Override the sensor database used by `data/logger.py` and the trainer |
| `HEARTH_SESSION_KEY` | Flask session signing key; otherwise a random key is generated on startup |

`constants.py` defines vital bounds, label mapping, fallback medians, and NEWS2-style severity rules.

---

## Runtime data

`paths.py` creates a persistent data directory and copies bundled files there only when no runtime copy exists:

| Platform | Default data directory |
| --- | --- |
| macOS | `~/Library/Application Support/HearthAI/` |
| Windows | `%APPDATA%\HearthAI\` (home directory fallback if `APPDATA` is unset) |
| Linux / other | `~/.hearthai/` |

Runtime files include `hearth_tabnet.pth`, `hearth_sensor.db`, `hearth_results.db`, `hearth_live.db`, and `user.db`. Live sessions use `hearth_live.db`; historical summaries, predictions, and alerts use `hearth_results.db`; authentication uses `user.db`. An existing runtime checkpoint takes precedence over the bundled checkpoint.

The standalone generator uses files under `data/`, and dashboard retraining generates `hearth_sensor.db` in the repository root. These paths differ from the default persistent sensor database.

---

## Project structure notes

- The launcher runs the IoT simulator and AI server as separate subprocesses; Flask runs in a thread in the launcher process.
- Flask REST routes are defined in `ui/gui.py`. `server/api.py` provides supporting functions and sensor validation.
- Historical result writes in `data/logger.py` use a daemon-thread queue. Live tick and prediction writes are synchronous. Authentication and some dashboard queries access SQLite directly.
- Simulator-to-server messages use a 4-byte big-endian length prefix followed by a JSON payload containing `sim_date` and `readings`. The AI server caps frames at 10 MB.
- The optional device adapter accepts an unframed JSON list, translates vendor field names, and forwards framed messages to the AI server. The launcher does not start this adapter.

---
