# Architecture

[Back to README](../README.md)

## Source map

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
│   ├── alert_engine.py  debounced Critical alerts and predictive alert events
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
└── HearthAI.spec         PyInstaller specification
```

## Processes and data flow

- The launcher runs the IoT simulator and AI server as separate subprocesses; Flask runs in a thread in the launcher process.
- Flask REST routes are defined in `ui/gui.py`. `server/api.py` provides supporting functions and sensor validation.
- Historical result writes in `data/logger.py` use a daemon-thread queue. Live tick, prediction, and alert writes are synchronous. Authentication and some dashboard queries access SQLite directly.
- Simulator-to-server messages use a 4-byte big-endian length prefix followed by a JSON payload containing `sim_date` and `readings`. The AI server caps frames at 10 MB.
- The optional device adapter accepts an unframed JSON list, translates vendor field names, and forwards framed messages to the AI server. The launcher does not start this adapter.

To run only the dashboard from the repository root:

```bash
python -m ui.gui --port 8050 --no-browser
```

This does not start the AI server or simulator. The Flask app binds to `0.0.0.0`; the AI server defaults to `127.0.0.1:65432`.

## Packaging

`HearthAI.spec` bundles the application modules, HTML pages, static assets, model checkpoint, and the `server/ai_server.py` and `iot/simulator.py` scripts used by the frozen launcher. Paths are relative to the specification file.

Install PyInstaller separately from the application dependencies, then build from the repository root:

```bash
python -m pip install pyinstaller
python -m PyInstaller --clean --noconfirm HearthAI.spec
```

The console executable is written to `dist/HearthAI` (`dist/HearthAI.exe` on Windows). Build on the target operating system.

See [Configuration and runtime data](configuration.md) for database locations and [Model](model.md) for inference behavior.


## API access

Patient API routes require login. Guardian lists, details, alerts, summaries, trends, and health counters include only assigned patients. Admin sessions can access all patients. Public signup creates guardian accounts; the demo admin account is created at server startup. Assignment, guardian listings, and training status require an admin session. Dashboard training is disabled; `/api/retrain` returns HTTP 410 for admins and starts no worker.
