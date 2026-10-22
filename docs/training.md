# Training

[Back to README](../README.md)

Run commands from the repository root with the virtual environment activated. A checkpoint is bundled, so training is optional.

## Standalone commands

1. Stop Hearth with **Ctrl+C** in the launcher terminal.
2. Generate training data:

```bash
python -m data.generator
```

Choose **1 (Training Data)**. The generator writes `data/hearth_sensor.db`. Training requires at least 1,000 readings.

3. Train from that same database:

```bash
# macOS/Linux
HEARTH_SENSOR_DB="$PWD/data/hearth_sensor.db" python -m model.engine --epochs 30 --batch-size 2048 --max-samples 500000
```

```powershell
# Windows PowerShell
$env:HEARTH_SENSOR_DB = "$PWD/data/hearth_sensor.db"
python -m model.engine --epochs 30 --batch-size 2048 --max-samples 500000
Remove-Item Env:HEARTH_SENSOR_DB
```

The trainer saves the checkpoint in the persistent runtime directory. It does not replace the bundled checkpoint. See [Configuration and runtime data](configuration.md#runtime-data) for locations.

4. After training completes, restart Hearth:

```bash
python main.py
```

The AI server loads the saved runtime checkpoint at startup.

## Launcher training

If no checkpoint is available, `python main.py` offers **T** to generate synthetic data and train, or **Q** to quit. The launcher asks for the patient count, days, and readings per hour. Generation and training both use the configured sensor database and finish before the AI server starts.

## Dashboard retraining

Dashboard training is disabled. Use the standalone commands above.

`POST /api/retrain` returns HTTP 410 for admins without generating data or starting training. `/api/training_status` reports `enabled: false` and `training: false`. Both endpoints require an admin session. Dashboard API routes remain available after a rejected training request.
