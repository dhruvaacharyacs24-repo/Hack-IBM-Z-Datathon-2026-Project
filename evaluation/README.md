# Evaluation

The directory "\role_b_review" contains scripts and utilities for evaluating the ECG classification model.

## Files

- `evaluate_model.py` — evaluation entry point, subject to verification against its actual implementation.
- `metrics.py` — metric utilities, subject to verification against its actual implementation.

## Planned Evaluation Outputs

Where supported by the implementation, report:

- Confusion matrix
- Per-class precision and recall
- Per-class and macro F1-score
- Overall accuracy
- ROC-AUC, only when the class setup and evaluation method support a valid calculation
- Evaluation dataset size and class distribution

## Reproducibility

Document the model checkpoint, data split, label mapping, preprocessing, and exact command used for each evaluation run.

Do not commit placeholder or fabricated metrics. Add actual results only after the evaluation script completes successfully and the output has been reviewed.

## Important

Evaluation results on a particular dataset do not establish clinical validity or guarantee performance on real-world patient data.
