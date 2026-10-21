# Model

[Back to README](../README.md)

## Design and inputs

`model/layers.py` implements a dual-head TabNet in PyTorch:

- **Dual head** — status logits are converted to three-class probabilities with softmax; risk logits are converted to scores with sigmoid
- **FocalLoss** — down-weights easy examples and applies class weights of 0.1 / 0.3 / 0.6 for Healthy / Unhealthy / Critical
- **GhostBatchNorm** — splits training batches into virtual sub-batches
- **Attention weights** — attention masks are averaged across decision steps; the dashboard displays the weights and top features

Input features: heart rate, systolic BP, diastolic BP, body temperature, SpO2, activity level, Δ heart rate, Δ SpO2 (8 features total).

The engine selects Apple MPS, then CUDA, then CPU according to availability.

## Risk scores

Risk training labels mark whether a Critical status occurs in the next seven readings for a patient. `predict_risk()` averages risk-head scores across the input window. Live estimates start after seven ticks and use the latest seven readings. Historical estimates use daily summaries. `model/predictor.py` uses the shared engine. The training lookahead is seven readings, not seven calendar days.

## Fallback and alerts

If the model cannot load, the live server uses a NEWS2-style rule-based status fallback. `constants.py` defines the NEWS2 scoring rules, vital bounds, label mapping, and fallback medians.

The fallback assigns fixed confidence `0.80` and uniform attention weights (`1/8` per feature). Both are placeholder values. The confidence stays above the default `0.55` downgrade threshold, preserving rule-based Critical classifications.

Risk estimates are unavailable in fallback mode because `predict_risk()` requires a loaded model.

The dashboard derives critical and predictive alerts from stored patient states and risk predictions. `server/alert_engine.py` contains a separate debounced alert helper; the live server does not call it.

See [Training](training.md) for training commands and [Configuration](configuration.md) for thresholds and their limits.
