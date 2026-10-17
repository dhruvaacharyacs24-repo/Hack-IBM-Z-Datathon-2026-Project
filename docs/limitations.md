# Limitations and Responsible Use

CardioSentinel is an experimental prototype developed for the IBM Z Datathon 2026.

## Technical Scope

- The application analyzes selected windows from stored ECG recordings.
- The model produces class probabilities that are presented using the application's configured categories and decision rules.
- The Critical threshold is an application-level rule, not a clinically established threshold.
- Grad-CAM provides a model-attribution visualization; it does not establish medical causation or prove a prediction is correct.
- Prediction history is maintained within the application session and is not guaranteed to persist across sessions or server restarts.

## Evaluation

Performance claims must be based on documented, reproducible evaluation results. Dataset performance alone does not demonstrate generalization to unseen hospitals, devices, or patient populations.

## Clinical Disclaimer

The system is not clinically validated, is not a medical device, and must not be used to diagnose, treat, or make decisions about an individual's care.

## Future Work

Potential work includes more rigorous patient-independent evaluation, broader validation, verified streaming integration, deployment security, monitoring, and collaboration with qualified clinical experts.
