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

## Processes and data flow

- The launcher runs the IoT simulator and AI server as separate subprocesses; Flask runs in a thread in the launcher process.
- Flask REST routes are defined in `ui/gui.py`. `server/api.py` provides supporting functions and sensor validation.
- Historical result writes in `data/logger.py` use a daemon-thread queue. Live tick and prediction writes are synchronous. Authentication and some dashboard queries access SQLite directly.
- Simulator-to-server messages use a 4-byte big-endian length prefix followed by a JSON payload containing `sim_date` and `readings`. The AI server caps frames at 10 MB.
- The optional device adapter accepts an unframed JSON list, translates vendor field names, and forwards framed messages to the AI server. The launcher does not start this adapter.

To run only the dashboard from the repository root:

```bash
python -m ui.gui --port 8050 --no-browser
```

This does not start the AI server or simulator. The Flask app binds to `0.0.0.0`; the AI server defaults to `127.0.0.1:65432`.

## Packaging

`HearthAI.spec` still names files and imports from an older flat layout, including `ai_server.py`, `iot_simulator.py`, and `hearth_gui`. These paths do not match the current module structure, so the specification needs updating before packaging.

See [Configuration and runtime data](configuration.md) for database locations and [Model](model.md) for inference behavior.
