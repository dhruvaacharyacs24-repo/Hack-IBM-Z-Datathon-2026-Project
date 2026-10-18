from pathlib import Path

import numpy as np
import wfdb

FS = 360
DEFAULT_WINDOW_SIZE = 1080
DEFAULT_HOP = 360

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "mit-bih"
    / "mit-bih-arrhythmia-database-1.0.0"
)


def resolve_record_path(record_path):
    """Resolve a record name or path to a WFDB record base path."""
    path = Path(record_path).expanduser()

    if not path.is_absolute():
        project_candidate = PROJECT_DIR / path
        data_candidate = DATA_DIR / path

        if project_candidate.with_suffix(".hea").exists():
            path = project_candidate
        elif data_candidate.with_suffix(".hea").exists():
            path = data_candidate
        else:
            path = project_candidate

    path = path.resolve()

    if not path.with_suffix(".hea").exists():
        raise FileNotFoundError(
            f"WFDB header not found: {path.with_suffix('.hea')}"
        )

    return path


def load_record(record_path):
    """Load a WFDB recording and return its signal and metadata."""
    path = resolve_record_path(record_path)
    record = wfdb.rdrecord(str(path), physical=False)

    signal = np.asarray(record.d_signal)

    if signal.ndim != 2 or signal.shape[0] == 0:
        raise ValueError(f"Unexpected signal shape: {signal.shape}")

    if int(round(record.fs)) != FS:
        raise ValueError(
            f"Expected {FS} Hz sampling, got {record.fs} Hz"
        )

    return {
        "record_path": str(path),
        "signal": signal,
        "fs": float(record.fs),
        "channels": list(record.sig_name),
        "length": int(signal.shape[0]),
        "duration_seconds": signal.shape[0] / record.fs,
    }


def get_window(signal, position, window_size=DEFAULT_WINDOW_SIZE):
    """Return a complete window, or None if it exceeds the signal."""
    signal = np.asarray(signal)
    position = int(position)
    window_size = int(window_size)

    if signal.ndim != 1:
        raise ValueError("signal must be a one-dimensional channel")

    if position < 0:
        raise ValueError("position cannot be negative")

    if window_size <= 0:
        raise ValueError("window_size must be positive")

    end = position + window_size

    if end > len(signal):
        return None

    window = signal[position:end]

    if not np.isfinite(window).all():
        raise ValueError("ECG window contains non-finite values")

    return window.astype(np.float32, copy=False)


def advance_position(
    position, hop=DEFAULT_HOP, signal_length=None
):
    """Advance playback by hop samples without exceeding the last window."""
    position = int(position)
    hop = int(hop)

    if position < 0:
        raise ValueError("position cannot be negative")

    if hop <= 0:
        raise ValueError("hop must be positive")

    next_position = position + hop

    if signal_length is not None:
        signal_length = int(signal_length)

        if signal_length < 0:
            raise ValueError("signal_length cannot be negative")

        next_position = min(
            next_position,
            max(0, signal_length - DEFAULT_WINDOW_SIZE),
        )

    return next_position


def find_first_ventricular_event(record_path):
    """Return the first annotated ventricular event, if present.

    MIT-BIH annotation symbols V, E, and ! are treated as
    ventricular-related events for this demo. This is not a
    clinical event detector.
    """
    path = resolve_record_path(record_path)
    annotation = wfdb.rdann(str(path), "atr")

    ventricular_symbols = {"V", "E", "!"}

    for sample, symbol in zip(
        annotation.sample, annotation.symbol
    ):
        if symbol in ventricular_symbols:
            return {
                "sample": int(sample),
                "time_seconds": float(sample / FS),
                "symbol": symbol,
            }

    return None
