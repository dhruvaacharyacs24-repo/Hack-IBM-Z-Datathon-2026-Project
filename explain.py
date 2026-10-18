"""Plain-language explanations for the ECG research dashboard.

This module describes model predictions and basic signal statistics.
It does not diagnose a patient or determine clinical safety.
"""

import numpy as np
import core


def signal_facts(raw_win):
    """Return simple descriptive statistics for an ECG window."""
    x = np.asarray(raw_win, dtype=np.float64).reshape(-1)

    if x.size == 0 or not np.isfinite(x).all():
        return {
            "valid": False,
            "message": "Signal statistics are unavailable for this window."
        }

    return {
        "valid": True,
        "samples": int(x.size),
        "duration_seconds": round(x.size / core.FS, 2),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "peak_to_peak": float(np.ptp(x)),
    }


def explain(raw_win, result):
    """Create a readable explanation from a window and core.predict result."""
    facts = signal_facts(raw_win)

    if not isinstance(result, dict) or "probs" not in result:
        return (
            "An explanation could not be generated because prediction "
            "information is unavailable. This is a research tool, not "
            "a medical diagnosis."
        )

    labels = list(core.LEVELS)
    probs = np.asarray(result["probs"], dtype=float).reshape(-1)

    if len(probs) != len(labels) or not np.isfinite(probs).all():
        return (
            "An explanation could not be generated because the model "
            "probabilities are invalid."
        )

    if np.any(probs < 0) or np.any(probs > 1):
        return "An explanation could not be generated because probabilities are out of range."

    # These are the unadjusted model probabilities.
    raw_index = int(np.argmax(probs))
    raw_label = labels[raw_index]
    raw_score = float(probs[raw_index])

    # The operational label may differ because core.predict applies a threshold.
    operational_label = result.get("level", raw_label)
    critical_score = float(probs[2]) if len(probs) > 2 else 0.0
    threshold = float(getattr(core, "CRIT_T", 0.9))

    lines = [
        f"**Model result:** {operational_label}",
        (
            f"The highest raw model probability is **{raw_label}** "
            f"({raw_score:.1%})."
        ),
        (
            f"Model probabilities — Normal: {probs[0]:.1%}, "
            f"Watch: {probs[1]:.1%}, Critical: {critical_score:.1%}."
        ),
    ]

    if (
        len(labels) > 2
        and operational_label != raw_label
        and critical_score <= threshold
    ):
        lines.append(
            f"The Critical probability ({critical_score:.1%}) does not "
            f"exceed the configured {threshold:.0%} threshold, so the "
            "operational label differs from the highest raw-probability "
            "class. This threshold rule is a software setting, not a "
            "clinical safety threshold."
        )
    elif len(labels) > 2 and critical_score > threshold:
        lines.append(
            f"The Critical probability exceeds the configured "
            f"{threshold:.0%} threshold."
        )

    if facts["valid"]:
        lines.append(
            f"The analyzed window contains {facts['samples']} samples "
            f"({facts['duration_seconds']:.2f} seconds at {core.FS} Hz)."
        )
        lines.append(
            f"Basic signal statistics: standard deviation "
            f"{facts['std']:.4g}; peak-to-peak range "
            f"{facts['peak_to_peak']:.4g}. These describe the waveform "
            "and do not establish a medical condition."
        )
    else:
        lines.append(facts["message"])

    lines.append(
        "Interpretation: this is an experimental machine-learning output. "
        "It is not a confirmed diagnosis, and a Normal label does not "
        "rule out a medical problem."
    )

    return "\n\n".join(lines)
