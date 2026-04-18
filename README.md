# Hearth AI

A home-care patient monitoring system that streams live vitals from IoT sensors, runs real-time TabNet inference to classify patient status (Healthy / Unhealthy / Critical), and surfaces alerts and a 7-day risk score through a web dashboard.

---

## What it does

- **Live monitoring** — IoT simulator streams heart rate, blood pressure, temperature, and SpO2 to a TCP server every few seconds
- **TabNet inference** — a custom dual-head TabNet model classifies each patient as Healthy, Unhealthy, or Critical on every tick and produces a continuous risk score
- **Alert engine** — debounced alerts fire when a patient crosses the Critical threshold for several consecutive ticks
- **7-day risk prediction** — a separate predictor looks back over a rolling window and forecasts deterioration risk
- **Web dashboard** — Flask-served dashboard with live charts, alert feed, and per-patient drill-down
- **Explainability** — attention weights from the TabNet steps are exposed so you can see which vitals drove each classification

---

## Architecture

```
main.py                  entry point — launches all subprocesses and the GUI
│
├── model/
│   ├── layers.py        FocalLoss, GhostBatchNorm, GLUBlock, AttentiveTransformer, TabNet
│   ├── engine.py        training loop, checkpoint management, normalisation
│   └── predictor.py     7-day risk feature engineering + inference
│
├── server/
│   ├── ai_server.py     TCP server — receives IoT frames, runs inference, writes results
│   ├── alert_engine.py  async alert queue with debounce logic
│   ├── api.py           Flask REST API endpoints
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
│   └── db.py            user authentication (SQLite-backed)
│
├── ui/
│   └── gui.py           Flask app — serves the dashboard, exposes the API
│
├── config.py            all tunable operational constants (ports, thresholds, etc.)
└── constants.py         fixed clinical/ML constants (feature names, label map, medians)
```

---

## Setup

**Requirements:** Python 3.10+, pip

```bash
git clone <repo-url>
cd hearth-ai
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `flask`, `matplotlib`, `seaborn`, `pydantic`

---

## Running

```bash
python main.py
```

On first run, no model checkpoint exists. The CLI will prompt you to train one:

```
[T] Train model now
[Q] Quit

Number of patients  (default 200):
Number of days      (default 60):
Sensor reads/hour   (default 12):
```

Once training completes, the dashboard opens automatically at `http://127.0.0.1:8050`.

Subsequent runs skip straight to the dashboard using the saved checkpoint (`hearth_tabnet.pth`).

---

## Model

The classifier is a from-scratch TabNet implementation — no external TabNet library. Key design choices:

- **Dual head** — one softmax head for 3-class status, one sigmoid head for continuous risk score
- **FocalLoss** — down-weights easy Healthy samples so the model doesn't ignore the minority Critical class
- **GhostBatchNorm** — splits batches into virtual sub-batches to keep BN statistics stable at large batch sizes
- **Attention weights** — each forward pass returns per-step feature importances, used for XAI on the dashboard

Input features: heart rate, systolic BP, diastolic BP, body temperature, SpO2, activity level, Δ heart rate, Δ SpO2 (8 features total).

---

## Configuration

`config.py` — change ports, alert thresholds, debounce counts, training defaults:

```python
SERVER_PORT              = 65432
CRITICAL_CONF_THRESHOLD  = 0.55   # confidence floor — Critical predictions below this get downgraded to Unhealthy
DEBOUNCE_THRESHOLD       = 3      # consecutive critical ticks before alert fires
WINDOW_DAYS              = 7      # lookback window for risk prediction
```

`constants.py` — clinical bounds, label mapping, fallback medians. Only change these if you have a clinical reason to.

---

## Project structure notes

- `main.py` is the only entry point — do not run submodules directly
- The IoT simulator and AI server run as separate subprocesses so they can be killed and restarted independently
- All SQLite I/O goes through `data/logger.py` — writes are queued through a daemon thread to avoid blocking the inference loop
- The TCP framing protocol is length-prefixed (4-byte big-endian header + JSON payload)

---
