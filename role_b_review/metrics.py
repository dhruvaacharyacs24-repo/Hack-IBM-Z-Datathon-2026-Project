
import warnings

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

CLASS_NAMES = ["Normal", "Watch", "Critical"]
CLASS_IDS = [0, 1, 2]

# Allow tiny floating-point errors in probability values.
PROB_TOL = 1e-6
SUM_TOL = 1e-4


def report(y_true, y_pred, probs):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probs = np.asarray(probs, dtype=float)

    # 1. Validate shapes and labels.
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have equal, nonzero length."
        )

    if probs.shape != (len(y_true), 3):
        raise ValueError(
            "probs must have shape (number of samples, 3)."
        )

    if not np.isin(y_true, CLASS_IDS).all():
        raise ValueError("y_true contains unknown class IDs.")

    if not np.isin(y_pred, CLASS_IDS).all():
        raise ValueError("y_pred contains unknown class IDs.")

    # 2. Validate probabilities with floating-point tolerance.
    if not np.isfinite(probs).all():
        raise ValueError("probs must contain only finite values.")

    if np.any(probs < -PROB_TOL) or np.any(probs > 1 + PROB_TOL):
        raise ValueError(
            "probs must be between 0 and 1, within tolerance."
        )

    if not np.allclose(probs.sum(axis=1), 1.0, atol=SUM_TOL, rtol=0):
        raise ValueError("Each row of probs must sum to 1.")

    # 3. Check whether predicted labels match probability argmax.
    # Tiny excursions tolerated above are clipped for this comparison.
    probs_for_argmax = np.clip(probs, 0.0, 1.0)
    probability_predictions = np.argmax(probs_for_argmax, axis=1)
    mismatches = probability_predictions != y_pred
    mismatch_fraction = float(np.mean(mismatches))

    print(f"Probability/prediction mismatch fraction: {mismatch_fraction:.4f}")

    if np.any(mismatches):
        warnings.warn(
            "Some y_pred labels disagree with argmax(probs). "
            "Check class order and prediction generation.",
            RuntimeWarning,
            stacklevel=2,
        )

    # 4. Classification report: readable output plus numeric results.
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASS_IDS,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # 5. Confusion matrix: rows = actual, columns = predicted.
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)

    print("CONFUSION MATRIX")
    print(cm)
    print("Rows = actual classes; columns = predicted classes.")

    # 6. False-alarm rate for actual Normal windows.
    normal_mask = y_true == 0
    normal_count = int(normal_mask.sum())
    false_alarms = int(np.sum(normal_mask & (y_pred != 0)))

    false_alarm_rate = (
        false_alarms / normal_count
        if normal_count > 0
        else float("nan")
    )

    # OPTIONAL estimate: assumes 1 second of new signal per window.
    # This counts false-alarm windows, not distinct episodes.
    hours_of_new_signal = len(y_true) / 3600.0
    false_alarm_windows_per_hour = false_alarms / hours_of_new_signal

    print(f"\nFalse alarms: {false_alarms}")
    print(f"True Normal windows: {normal_count}")
    print(f"False-alarm rate: {false_alarm_rate:.4f}")
    print(
        "Estimated false-alarm windows per hour of test data (counts windows, not episodes): "
        f"{false_alarm_windows_per_hour:.4f}"
    )

    # 7. One-vs-rest AUROC independently for each class.
    # AUC needs at least one positive and one negative example.
    per_class_auc = {}
    auc_skipped = {}

    for class_id, class_name in enumerate(CLASS_NAMES):
        binary_truth = (y_true == class_id).astype(int)

        positives = int(binary_truth.sum())
        negatives = int(len(binary_truth) - positives)

        if positives == 0:
            per_class_auc[class_name] = None
            auc_skipped[class_name] = (
                "No positive examples in y_true."
            )
        elif negatives == 0:
            per_class_auc[class_name] = None
            auc_skipped[class_name] = (
                "No negative examples in y_true."
            )
        else:
            per_class_auc[class_name] = float(
                roc_auc_score(binary_truth, probs[:, class_id])
            )

    all_classes_present = set(np.unique(y_true)) == set(CLASS_IDS)

    # Macro AUROC is reported only when all three classes are present
    # and all three one-vs-rest AUROCs are calculable.
    if all_classes_present and all(
        value is not None for value in per_class_auc.values()
    ):
        macro_auc = float(np.mean(list(per_class_auc.values())))
    else:
        macro_auc = None

    print("\nONE-VS-REST AUROC")
    for class_name, auc in per_class_auc.items():
        if auc is None:
            print(f"{class_name}: skipped â€” {auc_skipped[class_name]}")
        else:
            print(f"{class_name}: {auc:.4f}")

    if macro_auc is None:
        print("Macro AUROC: unavailable (not all classes are present).")
    else:
        print(f"Macro AUROC: {macro_auc:.4f}")

    if auc_skipped:
        print("AUROC skipped reasons:", auc_skipped)

    # 8. Return machine-readable results.
    return {
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "probability_prediction_mismatch_fraction": mismatch_fraction,
        "false_alarms": false_alarms,
        "true_normal_windows": normal_count,
        "false_alarm_rate": false_alarm_rate,
        "false_alarm_windows_per_hour_estimate": (
            false_alarm_windows_per_hour
        ),
        "per_class_auc_ovr": per_class_auc,
        "auc_skipped_reasons": auc_skipped,
        "macro_auroc_ovr": macro_auc,
    }

