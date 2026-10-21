# Hearth AI

A home-care patient monitoring demo that streams synthetic vitals, classifies patient status with a custom TabNet model, and displays live charts, alerts, and risk scores in a Flask dashboard.

## Features

- **Simulated vitals:** heart rate, blood pressure, temperature, SpO2, and activity streamed over TCP.
- **Patient assessment:** Healthy / Unhealthy / Critical classification, rolling risk scores, and model attention views.
- **Live dashboard:** patient charts, debounced Critical alerts, risk alerts, and per-patient details.
- **Accounts and administration:** guardian accounts with assigned-patient access, patient assignment, and admin retraining.

## Quick start

**Requirements:** Python 3.11+ and pip. Run commands from the repository root.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The launcher asks for the patient count (default **50**), tick interval (**2 seconds**), and dashboard port (**8050**). Press Enter to accept each default.

A model checkpoint is bundled. If the runtime checkpoint is missing, the launcher offers to generate data and train a model; see [Training](docs/training.md).

The launcher starts the AI server, simulator, and dashboard, then opens the landing page:

- Landing page: [http://localhost:8050/landing](http://localhost:8050/landing)
- Dashboard: [http://localhost:8050/](http://localhost:8050/)

Use your chosen port if you changed it. Press **Ctrl+C** in the launcher terminal to stop.

Dashboard charts and fonts load from external services and need internet access.

## Dashboard access

- **Admin Mode:** use the dashboard button for the demo account (`admin` / `adminpassword`). The server creates the demo account at startup if needed. Admins can assign patients and start retraining.
- **Guardian:** sign up, then ask an admin to assign patient IDs to your account.
- Display names can be edited in the dashboard.

## Architecture

```text
Simulator → TCP server → TabNet → SQLite → Flask dashboard
```

| Location | Purpose |
| --- | --- |
| `main.py` | Starts the server, simulator, and dashboard |
| `model/` | Model layers, training, and risk inference |
| `server/`, `iot/` | Sensor transport, inference server, and simulation |
| `data/` | Synthetic data generation and storage |
| `ui/`, `auth/` | Dashboard, accounts, and patient assignments |
| `config.py`, `constants.py`, `paths.py` | Defaults, feature definitions, and runtime paths |

See [Architecture](docs/architecture.md) for the source map, process layout, and TCP format.

## Demo data

Training and live demo data are synthetic.

## Further documentation

- [Architecture](docs/architecture.md): files, processes, storage writes, and TCP framing.
- [Model](docs/model.md): features, model design, risk semantics, and fallback behavior.
- [Training](docs/training.md): launcher training, standalone commands, and retraining limitations.
- [Configuration and runtime data](docs/configuration.md): defaults, environment variables, database locations, and checkpoint precedence.
