from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core import FS, WIN, LEVELS, predict
from explain import explain
from streaming import (
    DATA_DIR,
    load_record,
    get_window,
    find_first_ventricular_event,
)
from ui_components import render_dashboard
from role_b_review.xai import gradcam


@st.cache_data(show_spinner=False)
def cached_gradcam(window_values: tuple, class_index: int):
    """Cache Grad-CAM for a specific ECG window and class."""
    return gradcam(np.asarray(window_values, dtype=np.float32), class_index)


st.set_page_config(
    page_title="ECG Research Dashboard",
    page_icon="📈",
    layout="wide",
)

st.sidebar.title("ECG Research Controls")
st.sidebar.caption("MIT-BIH Arrhythmia Database")

if not DATA_DIR.exists():
    st.error(f"Dataset directory not found: {DATA_DIR}")
    st.stop()

records = sorted(p.stem for p in DATA_DIR.glob("*.hea"))
if not records:
    st.error("No WFDB records found. Check the dataset extraction.")
    st.stop()

record_id = st.sidebar.selectbox("ECG record", records, index=0)

@st.cache_data(show_spinner="Loading ECG record...")
def cached_load_record(path):
    loaded = load_record(path)
    # Cache serializable signal data and metadata.
    return {
        "record_path": loaded["record_path"],
        "signal": loaded["signal"],
        "fs": loaded["fs"],
        "channels": loaded["channels"],
        "length": loaded["length"],
        "duration_seconds": loaded["duration_seconds"],
    }

try:
    record = cached_load_record(str(DATA_DIR / record_id))
except Exception as exc:
    st.error(f"Could not load record {record_id}: {exc}")
    st.stop()

channels = record["channels"]
channel = st.sidebar.selectbox(
    "ECG channel",
    channels,
    help="Choose one channel from the selected recording.",
)
channel_index = channels.index(channel)
signal = np.asarray(record["signal"][:, channel_index], dtype=np.float32)

max_start = max(0, len(signal) - WIN)
position = st.sidebar.slider(
    "Window start (seconds)",
    min_value=0.0,
    max_value=max_start / FS,
    value=0.0,
    step=1.0,
)
start_sample = min(int(position * FS), max_start)
window = get_window(signal, start_sample, WIN)

st.title("ECG Arrhythmia Research Dashboard")

st.markdown("""
<style>
.ecg-pulse-banner {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 14px 20px;
    margin: 10px 0 22px 0;
    border: 1px solid rgba(50,230,226,.24);
    border-radius: 14px;
    background: linear-gradient(105deg,#13253d,#0a1428);
    box-shadow: 0 0 22px rgba(50,230,226,.06);
}
.ecg-heart {
    color: #32e6e2;
    font-size: 28px;
    animation: ecgHeartPulse 1.35s ease-in-out infinite;
    text-shadow: 0 0 12px rgba(50,230,226,.5);
}
.ecg-banner-title {
    color: #f1f7ff;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.2px;
}
.ecg-banner-subtitle {
    color: #91a9c7;
    font-size: 12px;
    margin-top: 5px;
}
.ecg-monitor-line { flex: 1; min-width: 80px; height: 42px; }
.ecg-monitor-line path {
    fill: none;
    stroke: #32e6e2;
    stroke-width: 2.2;
    stroke-linejoin: round;
    stroke-linecap: round;
    stroke-dasharray: 360;
    stroke-dashoffset: 360;
    animation: ecgTrace 2.8s linear infinite;
    filter: drop-shadow(0 0 3px rgba(50,230,226,.45));
}
@keyframes ecgHeartPulse {
    0%,100% { transform: scale(1); opacity: .85; }
    12% { transform: scale(1.17); opacity: 1; }
    24% { transform: scale(1); }
}
@keyframes ecgTrace {
    0% { stroke-dashoffset: 360; }
    75%,100% { stroke-dashoffset: 0; }
}
@media (prefers-reduced-motion: reduce) {
    .ecg-heart, .ecg-monitor-line path { animation: none; }
}
</style>
<div class="ecg-pulse-banner">
  <div class="ecg-heart" aria-hidden="true">&#9829;</div>
  <div>
    <div class="ecg-banner-title">CARDIAC SIGNAL ANALYSIS</div>
    <div class="ecg-banner-subtitle">
      Recorded MIT-BIH ECG · Experimental CNN inference
    </div>
  </div>
  <svg class="ecg-monitor-line" viewBox="0 0 360 42"
       preserveAspectRatio="none" aria-hidden="true">
    <path d="M0 22 L65 22 L78 20 L88 22 L100 22
             L112 5 L125 37 L140 13 L151 22
             L185 22 L197 19 L207 22 L235 22
             L247 8 L260 34 L273 15 L285 22 L360 22"/>
  </svg>
</div>
""", unsafe_allow_html=True)


st.caption(
    "Experimental model demonstration using MIT-BIH ECG recordings. "
    "Not a medical device; predictions are not clinically validated."
)


try:
    with st.spinner("Running CNN inference..."):
        result = predict(window)
except Exception as exc:
    st.error(f"Model inference failed: {exc}")
    st.stop()

probabilities = np.asarray(result["probs"], dtype=float)
if probabilities.size != len(LEVELS):
    st.error("Model output does not match the configured class labels.")
    st.stop()

probability_map = {
    label: float(probability)
    for label, probability in zip(LEVELS, probabilities)
}

# Explain the same window and prediction used by the dashboard.
explanation_text = explain(window, result)

# Unified summary cards: one place for the recording and inference metadata.
st.markdown("""
<style>
.ecg-card-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 16px 0 22px 0;
}
.ecg-card {
    background: linear-gradient(145deg, #172642, #0d172b);
    border: 1px solid #314568;
    border-radius: 14px;
    padding: 16px;
    min-height: 100px;
}
.ecg-card-label {
    color: #a9bad4;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 10px;
}
.ecg-card-value {
    color: #f3f7ff;
    font-size: 23px;
    font-weight: 650;
    overflow-wrap: anywhere;
}
.ecg-card-note {
    color: #9aabc4;
    font-size: 12px;
    margin-top: 7px;
}
@media (max-width: 750px) {
    .ecg-card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="ecg-card-grid">
      <div class="ecg-card">
        <div class="ecg-card-label">ECG record</div>
        <div class="ecg-card-value">{record_id}</div>
        <div class="ecg-card-note">MIT-BIH recording</div>
      </div>
      <div class="ecg-card">
        <div class="ecg-card-label">Channel</div>
        <div class="ecg-card-value">{channel}</div>
        <div class="ecg-card-note">{FS} Hz sampling rate</div>
      </div>
      <div class="ecg-card">
        <div class="ecg-card-label">Window duration</div>
        <div class="ecg-card-value">{WIN / FS:.1f} seconds</div>
        <div class="ecg-card-note">{WIN:,} samples</div>
      </div>
      <div class="ecg-card">
        <div class="ecg-card-label">Highest raw probability</div>
        <div class="ecg-card-value">{LEVELS[int(np.argmax(probabilities))]}</div>
        <div class="ecg-card-note">Class with highest softmax score</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load reference annotations from MIT-BIH. These are dataset annotations,
# not predictions generated by the CNN.
events = []
try:
    annotations = __import__("wfdb").rdann(record["record_path"], "atr")
    window_end = start_sample + WIN
    for sample, symbol in zip(annotations.sample, annotations.symbol):
        if start_sample <= sample < window_end:
            events.append({
                "time": float((sample - start_sample) / FS),
                "label": f"Reference annotation: {symbol}",
            })
except Exception as exc:
    st.warning(f"Reference annotations unavailable: {exc}")

risk_indicator = round(float(1.0 - probabilities[0]), 4)

# Keep a session-level history across Streamlit reruns.
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

history_entry = {
    "Record": record_id,
    "Channel": channel,
    "Start (s)": round(start_sample / FS, 2),
    "End (s)": round((start_sample + WIN) / FS, 2),
    "Predicted class": result["level"],
    "Normal probability": round(float(probabilities[0]), 4),
    "Watch probability": round(float(probabilities[1]), 4),
    "Critical probability": round(float(probabilities[2]), 4),
    "Inference latency (ms)": round(float(result["latency_ms"]), 2),
}

# Record each distinct selected window once, not once per Streamlit rerun.
history_key = (record_id, channel, start_sample)
if (
    not st.session_state["prediction_history"]
    or st.session_state["prediction_history"][-1].get("_key") != history_key
):
    history_entry["_key"] = history_key
    st.session_state["prediction_history"].append(history_entry)

history_rows = [
    {key: value for key, value in row.items() if key != "_key"}
    for row in st.session_state["prediction_history"]
]

dashboard_data = {
    "time": np.arange(WIN, dtype=float) / FS,
    "signal": window,
    "events": events,
    "alerts": [
        {
            "title": "Experimental CNN output",
            "message": (
                f"Predicted class: {result['level']}. "
                "This is a model output, not a clinical diagnosis."
            ),
            "level": "warning",
        },
        {
            "title": "Reference annotations",
            "message": (
                f"{len(events)} reference annotation(s) occur in this window. "
                "They come from the dataset, not from the CNN."
            ),
            "level": "info",
        },
    ],
    "risk_score": f"{risk_indicator:.1%}",
    "risk_label": "1 − P(Normal), experimental indicator",
    "probabilities": probability_map,
    "explanation": explanation_text,
    "history": pd.DataFrame(history_rows),
    "heart_rate": "Not calculated",
    "signal_status": "MIT-BIH recorded signal",
    "recorded_data": True,
}



# ECG WAVEFORM — REAL MIT-BIH SAMPLES
import plotly.graph_objects as go

ecg_amplitude = np.asarray(window, dtype=np.float64).reshape(-1)
ecg_time_seconds = np.arange(ecg_amplitude.size, dtype=np.float64) / float(FS)

if ecg_amplitude.size != WIN:
    st.error(f"Expected {WIN} ECG samples; received {ecg_amplitude.size}.")
elif not np.isfinite(ecg_amplitude).all():
    st.error("The selected ECG window contains non-finite samples.")
else:
    st.subheader("Recorded ECG Waveform")
    st.caption(
        f"MIT-BIH record {record_id} · Channel {channel} · "
        f"{FS} Hz · {ecg_amplitude.size} samples · "
        f"{ecg_amplitude.size / FS:.1f} seconds"
    )

    st.write(
        f"Signal range: {ecg_amplitude.min():.1f} to "
        f"{ecg_amplitude.max():.1f} digital units"
    )

    show_attention = st.checkbox(
        "Show Grad-CAM attention overlay",
        value=False,
        help=(
            "Highlights time points associated with the selected class. "
            "This experimental visualization is not a medical finding."
        ),
        key="show_gradcam_overlay",
    )

    attention = None
    if show_attention:
        try:
            with st.spinner("Calculating Grad-CAM attention..."):
                attention = cached_gradcam(
                    tuple(ecg_amplitude.astype(np.float32).tolist()),
                    int(np.argmax(probabilities)),
                )
        except Exception as exc:
            st.warning(f"Grad-CAM could not be calculated: {exc}")

    direct_fig = go.Figure()
    direct_fig.add_trace(
        go.Scattergl(
            x=ecg_time_seconds.tolist(),
            y=ecg_amplitude.tolist(),
            mode="lines",
            name="Recorded ECG",
            line=dict(color="#32E6E2", width=1.5),
            hovertemplate=(
                "Time: %{x:.3f} s<br>"
                "Amplitude: %{y:.1f}<extra></extra>"
            ),
        )
    )
    if attention is not None:
        direct_fig.add_trace(
            go.Scattergl(
                x=ecg_time_seconds.tolist(),
                y=ecg_amplitude.tolist(),
                mode="markers",
                name="Grad-CAM attention",
                marker=dict(
                    size=5,
                    color=np.asarray(attention).tolist(),
                    colorscale="Viridis",
                    cmin=0,
                    cmax=1,
                    opacity=0.85,
                    colorbar=dict(title="Attention"),
                ),
                hovertemplate=(
                    "Time: %{x:.3f} s<br>"
                    "Amplitude: %{y:.1f}<br>"
                    "Attention: %{marker.color:.3f}<extra></extra>"
                ),
            )
        )

    direct_fig.update_layout(
        title="ECG Signal — 3-Second Window",
        xaxis_title="Time (seconds)",
        yaxis_title="Digital amplitude (ADC units)",
        height=420,
        template="plotly_dark",
        margin=dict(l=60, r=25, t=65, b=55),
        showlegend=False,
    )
    direct_fig.update_xaxes(range=[0, (ecg_amplitude.size - 1) / FS])
    direct_fig.update_yaxes(
        range=[
            float(ecg_amplitude.min()) - 20,
            float(ecg_amplitude.max()) + 20,
        ],
        autorange=False,
    )
    st.plotly_chart(
        direct_fig,
        use_container_width=True,
        key="mit_bih_ecg_waveform",
    )

# Focused results section: no second waveform or repeated metric dashboard.
st.divider()
st.subheader("Model output")

raw_top_index = int(np.argmax(probabilities))
raw_top_class = LEVELS[raw_top_index]

from core import CRIT_T

st.markdown("### Prediction summary")

r1, r2, r3 = st.columns(3)
r1.metric("Highest raw probability", raw_top_class)
r2.metric("Threshold-adjusted class", result["level"])
r3.metric("Inference latency", f"{result['latency_ms']:.2f} ms")

st.caption(
    "**Highest raw probability:** The category the model gives "
    "the highest probability."
)
st.caption(
    "**Threshold-adjusted class:** The final category chosen after "
    "applying the Critical threshold rule."
)

critical_probability = float(probabilities[2])

st.markdown("#### Critical probability vs. threshold")
st.progress(min(max(critical_probability, 0.0), 1.0))
st.caption(
    f"Critical probability: {critical_probability:.1%} · "
    f"Critical threshold: {CRIT_T:.0%} "
    "(the Critical score must be above this level to remain eligible)."
)

if critical_probability <= CRIT_T:
    st.info(
        f"Critical probability ({critical_probability:.1%}) is at or "
        f"below the {CRIT_T:.0%} threshold, so Critical is suppressed "
        "when selecting the threshold-adjusted class."
    )

st.warning(
    "Research use only — this model is not clinically validated. "
    "A below-threshold score does not rule out an abnormality."
)

st.subheader("Prediction probabilities")
prob_fig = go.Figure()
prob_fig.add_trace(
    go.Bar(
        x=LEVELS,
        y=probabilities.tolist(),
        marker_color=["#39C7B7", "#E8B75A", "#E77D8D"],
        text=[f"{value:.1%}" for value in probabilities],
        textposition="outside",
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    )
)
prob_fig.update_layout(
    template="plotly_dark",
    height=330,
    yaxis_title="Softmax probability",
    yaxis=dict(range=[0, 1.12], tickformat=".0%"),
    xaxis_title="Model class",
    showlegend=False,
    margin=dict(l=45, r=25, t=35, b=50),
)
st.plotly_chart(prob_fig, use_container_width=True, key="final_probability_chart")

st.subheader("Reference annotations")
st.info(
    f"{len(events)} reference annotation(s) fall inside this selected window. "
    "These are dataset annotations, not CNN predictions."
)

st.subheader("Model explanation")
st.markdown(explanation_text)

with st.expander("Understanding the technical terms", expanded=True):
    st.markdown("""
    - **Class Probability:** A score showing how strongly the AI model favours each category — Normal, Watch, or Critical. It is not the actual chance of having a disease.
    - **Threshold-adjusted prediction:** A software rule that can change how a prediction is classified when a probability crosses a set cutoff. This threshold is experimental, not a medically validated limit.
    - **Signal Statistics:** Basic measurements, such as the minimum and maximum signal values, that describe the ECG segment being analysed.
    - **Grad-CAM (Gradient-weighted Class Activation Mapping):** An explainable-AI technique that highlights parts of the ECG waveform that influenced the model's prediction. Highlighted regions are not necessarily abnormal.
    - **Inference Latency:** The time the computer takes to process the ECG segment and produce its prediction.
    - **Reference Annotations:** Labels supplied with the dataset for comparison. They are separate from the model's own prediction.
    - **Risk Indicator:** An experimental score derived from the model's Normal-class probability. It is not a clinically validated measure of a person's medical risk.

    **Important:** This is an experimental AI prototype, not a diagnostic tool. A Normal prediction does not rule out a medical problem.
    """)

st.caption(
    "This explanation summarizes model output and basic signal statistics. "
    "It is experimental and is not a medical diagnosis."
)

with st.expander("Technical details", expanded=False):
    st.write(f"Dataset record path: {record['record_path']}")
    st.write(f"Selected channel: {channel}")
    st.write(f"Sampling rate: {FS} Hz")
    st.write(f"Input window: {WIN} samples")
    st.write(f"Window starts at sample: {start_sample}")
    st.write(f"Window starts at: {start_sample / FS:.2f} seconds")
    st.write(f"Signal minimum: {float(np.min(window)):.2f} digital units")
    st.write(f"Signal maximum: {float(np.max(window)):.2f} digital units")
    st.write("Model checkpoint: ecg_cnn_v2.pt")
    st.write("Inference device: CPU")
    st.write(f"Threshold-adjusted class: {result['level']}")
    st.write(f"Highest raw probability class: {raw_top_class}")
    st.json(probability_map)
