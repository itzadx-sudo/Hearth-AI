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

Risk training labels indicate whether a Critical status occurs in the next seven readings for a patient. At inference, `predict_risk()` averages risk-head scores over the supplied window. Live predictions begin after seven ticks and use the latest seven readings; historical mode supplies daily summary rows. `model/predictor.py` wraps the same engine rather than loading a separate prediction model. These scores describe the supplied reading window; they do not establish a seven-day forecast horizon.

## Fallback and alerts

If the model cannot load, the live server uses a NEWS2-style rule-based status fallback. `constants.py` defines the NEWS2 scoring rules, vital bounds, label mapping, and fallback medians.

The live-server fallback fills the existing result fields with fixed values: confidence `0.80` and equal attention weights of `1/8` per feature. These are placeholders, not learned probabilities or feature explanations. A confidence of `0.80` exceeds the default `0.55` downgrade threshold, so a rule-based Critical result is retained. A related engine fallback explicitly uses the threshold plus `0.05` to keep rule-based outputs compatible with this gate; the live server does not document why it chose exactly `0.80`. The value is not a measured 80% accuracy claim.

`predict_risk()` requires a loaded model, so fallback mode does not produce learned risk estimates.

The dashboard derives critical and predictive alerts from stored patient states and risk predictions. `server/alert_engine.py` contains a separate debounced alert helper; the live server does not call it.

See [Training](training.md) for training commands and [Configuration](configuration.md) for thresholds and their limits.
