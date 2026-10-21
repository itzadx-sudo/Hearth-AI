# Model

[Back to README](../README.md)

## Design and inputs

The classifier is a from-scratch TabNet implementation — no external TabNet library. Key design choices:

- **Dual head** — status logits are converted to three-class probabilities with softmax; risk logits are converted to scores with sigmoid
- **FocalLoss** — down-weights easy examples and applies class weights of 0.1 / 0.3 / 0.6 for Healthy / Unhealthy / Critical
- **GhostBatchNorm** — splits batches into virtual sub-batches to keep BN statistics stable at large batch sizes
- **Attention weights** — inference can request feature attention averaged across the decision steps; the dashboard displays these weights and top features

Input features: heart rate, systolic BP, diastolic BP, body temperature, SpO2, activity level, Δ heart rate, Δ SpO2 (8 features total).

The engine selects Apple MPS, then CUDA, then CPU according to availability.

## Risk scores

Risk training labels indicate whether a Critical status occurs in the next seven readings for a patient. At inference, `predict_risk()` averages risk-head scores over the supplied window. Live predictions begin after seven ticks and use the latest seven readings; historical mode supplies daily summary rows. `model/predictor.py` wraps the same engine rather than loading a separate prediction model. UI labels that say “7-day” or “7-day ahead” should therefore not be interpreted as an implemented seven-day forecast.

## Fallback and alerts

If the model cannot load, the server uses a NEWS2-style rule-based status fallback. `constants.py` defines its severity rules, vital bounds, label mapping, and fallback medians.

The dashboard derives critical and predictive alerts from stored patient states and risk predictions. `server/alert_engine.py` contains a separate debounced alert helper; the live server does not call it.

See [Training](training.md) for training commands and [Configuration](configuration.md) for thresholds and their limits.
