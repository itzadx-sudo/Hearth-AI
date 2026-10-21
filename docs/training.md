# Training

[Back to README](../README.md)

Run commands from the repository root with the virtual environment activated. A checkpoint is bundled, so training is optional when that checkpoint is available.

## Launcher training

`python main.py` offers training only if the runtime checkpoint file is absent:

```text
[T] Train model now
[Q] Quit

Number of patients  (default 200):
Number of days      (default 60):
Sensor reads/hour   (default 12):
```

Choose `T` to generate synthetic data and train, or `Q` to quit. Inline generation uses the configured sensor database.

## Standalone commands

```bash
# Interactive synthetic training/test data generation
python -m data.generator

# Train from the configured sensor database
python -m model.engine --epochs 30 --batch-size 2048 --max-samples 500000
```

The standalone generator writes `data/hearth_sensor.db` or `data/hearth_test.db`, while the engine normally reads the persistent sensor database. To train on the standalone generator's training output, set `HEARTH_SENSOR_DB` to that file:

```bash
# macOS/Linux
HEARTH_SENSOR_DB="$PWD/data/hearth_sensor.db" python -m model.engine
```

```powershell
# Windows PowerShell
$env:HEARTH_SENSOR_DB = "$PWD/data/hearth_sensor.db"
python -m model.engine
```

Training requires at least 1,000 readings. See [Configuration and runtime data](configuration.md#runtime-data) for persistent database locations and checkpoint precedence.

## Dashboard retraining

Admins can generate synthetic training data and start training from the dashboard. `/api/training_status` exposes progress. Most API routes return HTTP 503 during training; authentication, training status/control, landing pages, and static assets remain available.

Current limitations:

- The retraining worker generates `hearth_sensor.db` in the repository root, then restores the sensor database path before calling the trainer. The trainer can therefore read a different database from the newly generated one.
- Retraining does not automatically reload the model already held by the separate AI server process. Restart the launcher after successful training to load the saved checkpoint there.
