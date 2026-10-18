"""Reusable Streamlit UI components for the ECG dashboard.

This module only presents data supplied by its caller.  It does not load ECG
records, manage streaming state, run inference, or generate explanations.
"""

from __future__ import annotations

import math
from html import escape
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


def _inject_dashboard_styles() -> None:
    """Apply focused, framework-safe styling to the Streamlit dashboard shell."""
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at 12% -8%, rgba(94, 70, 220, .25), transparent 32%),
              radial-gradient(circle at 88% 0%, rgba(30, 207, 219, .16), transparent 28%),
              linear-gradient(145deg, #050b1a 0%, #09142a 56%, #0a1023 100%);
            color: #eaf3ff;
          }
          [data-testid="stHeader"] { background: transparent; }
          [data-testid="stToolbar"] { right: 1rem; }
          h1, h2, h3 { color: #f2f7ff !important; letter-spacing: -.02em; }
          h1 { text-shadow: 0 0 28px rgba(115, 92, 255, .28); }
          [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(109, 92, 255, .24) !important;
            background: rgba(10, 20, 42, .62);
          }
          [data-testid="stDataFrame"] {
            border: 1px solid rgba(47, 222, 218, .18);
            border-radius: 12px;
            overflow: hidden;
          }
          div.stButton > button {
            border: 1px solid rgba(103, 217, 255, .5);
            border-radius: 10px;
            background: linear-gradient(120deg, #5c46dc, #198fbd);
            color: #f8fbff;
            box-shadow: 0 8px 20px rgba(31, 106, 194, .24);
            transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
          }
          div.stButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.13);
            box-shadow: 0 11px 25px rgba(111, 87, 255, .34);
          }
          div.stButton > button:active { transform: translateY(1px) scale(.985); }
          div.stButton > button:focus-visible { outline: 3px solid rgba(64, 232, 224, .7); outline-offset: 2px; }
          [data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid rgba(98, 116, 255, .25);
            background: rgba(14, 25, 50, .78);
          }
          @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _install_cinematic_blackout() -> None:
    """Install one browser-side blackout timer without involving Streamlit state.

    Uses a JS setInterval so the blackout fires reliably every 30 seconds
    rather than waiting through a slow CSS keyframe cycle.
    """
    components.html(
        """
        <script>
          (() => {
            try {
              const host = window.parent;
              if (!host || host.__ecgBlackoutInstalled ||
                  host.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

              // Build the full-screen overlay once.
              const overlay = host.document.createElement('div');
              overlay.setAttribute('aria-hidden', 'true');
              Object.assign(overlay.style, {
                position: 'fixed', inset: '0', zIndex: '2147483647',
                background: '#000000', opacity: '0', pointerEvents: 'none',
                transition: 'opacity 0.3s ease-in-out'
              });
              host.document.body.appendChild(overlay);
              host.__ecgBlackoutInstalled = true;

              // Fire the blackout flash: fade in, hold briefly, fade out.
              function triggerBlackout() {
                overlay.style.opacity = '1';
                setTimeout(() => { overlay.style.opacity = '0'; }, 300);
              }

              // Fire every 30 seconds, starting immediately.
              triggerBlackout();
              setInterval(triggerBlackout, 30000);
            } catch (error) {
              // The dashboard remains fully usable if an embedding blocks parent access.
            }
          })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def _numeric_series(values: Any) -> np.ndarray | None:
    """Return a one-dimensional numeric array, or ``None`` when unusable."""
    if values is None or isinstance(values, (str, bytes)):
        return None

    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    return array if array.size else None


def _event_details(event: Any) -> tuple[float, str] | None:
    """Extract a marker time and optional label from a supported event value."""
    if isinstance(event, Mapping):
        event_time = event.get("time", event.get("timestamp"))
        label = event.get("label", event.get("event", "Event"))
    else:
        event_time = event
        label = "Event"

    try:
        numeric_time = float(event_time)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric_time):
        return None
    return numeric_time, str(label)


def render_ecg_waveform(
    time: Sequence[float] | np.ndarray,
    signal: Sequence[float] | np.ndarray,
    events: Sequence[Any] | None = None,
) -> None:
    """Render an interactive ECG waveform and optional event markers.

    Args:
        time: Numeric time values, usually in seconds.
        signal: Numeric ECG amplitude values corresponding to ``time``.
        events: Optional event timestamps or mappings with ``time`` (or
            ``timestamp``) and optional ``label`` keys.

    Returns:
        None. The chart or an explanatory Streamlit message is rendered.
    """
    time_values = _numeric_series(time)
    signal_values = _numeric_series(signal)
    if time_values is None or signal_values is None:
        st.info("ECG waveform data is not available.")
        return

    length = min(time_values.size, signal_values.size)
    if length == 0:
        st.info("ECG waveform data is not available.")
        return
    if time_values.size != signal_values.size:
        st.warning("Waveform time and signal lengths differ; displaying their shared range.")

    time_values = time_values[:length]
    signal_values = signal_values[:length]
    valid = np.isfinite(time_values) & np.isfinite(signal_values)
    if not valid.any():
        st.info("ECG waveform contains no finite values to display.")
        return

    time_values = time_values[valid]
    signal_values = signal_values[valid]
    figure = go.Figure(
        go.Scatter(
            x=time_values,
            y=signal_values,
            mode="lines",
            name="ECG signal",
            line={"color": "#32E6E2", "width": 2.4},
            hovertemplate="Time: %{x:.3f} s<br>Amplitude: %{y:.3f}<extra></extra>",
        )
    )

    marker_times: list[float] = []
    marker_labels: list[str] = []
    for event in ([] if events is None else events):
        details = _event_details(event)
        if details is not None:
            marker_times.append(details[0])
            marker_labels.append(details[1])
    if marker_times:
        nearest_indices = [int(np.abs(time_values - event_time).argmin()) for event_time in marker_times]
        figure.add_trace(
            go.Scatter(
                x=marker_times,
                y=[signal_values[index] for index in nearest_indices],
                mode="markers",
                name="Events",
                text=marker_labels,
                marker={"color": "#FF788A", "size": 10, "symbol": "diamond", "line": {"color": "#FFE4E9", "width": 1}},
                hovertemplate="%{text}<br>Time: %{x:.3f} s<extra></extra>",
            )
        )

    figure.update_layout(
        title="ECG Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Signal amplitude",
        hovermode="x unified",
        margin={"l": 50, "r": 20, "t": 50, "b": 45},
        height=380,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,16,39,0.72)",
        legend={"orientation": "h", "y": 1.1},
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(89, 225, 226, 0.14)", gridwidth=1)
    figure.update_yaxes(
        showgrid=True,
        gridcolor="rgba(139, 111, 255, 0.16)",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="rgba(89, 225, 226, 0.28)",
    )
    st.plotly_chart(figure, use_container_width=True)


def render_alert_cards(alerts: Sequence[Any] | None) -> None:
    """Render supplied alert or status messages as Streamlit status cards.

    Each item can be a string or a mapping with optional ``title``, ``message``,
    and ``level`` (``info``, ``success``, ``warning``, or ``error``) keys.
    """
    st.subheader("Alerts")
    if not alerts:
        st.info("No alerts to display.")
        return

    display_methods = {
        "info": st.info,
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
    }
    for alert in alerts:
        if isinstance(alert, Mapping):
            title = str(alert.get("title", "Status"))
            message = alert.get("message", alert.get("text", ""))
            level = str(alert.get("level", "info")).lower()
            content = f"{title}: {message}" if message else title
        else:
            level = "info"
            content = str(alert)
        display_methods.get(level, st.info)(content)


def render_risk_score(score: Any, label: Any) -> None:
    """Render a caller-supplied risk score and category without interpretation.

    Args:
        score: A displayable score value, or ``None`` if it is unavailable.
        label: The caller-supplied category associated with ``score``.

    Returns:
        None. A Streamlit metric is rendered.
    """
    st.subheader("Risk Score")
    score_text = "Unavailable" if score is None else str(score)
    label_text = "Not provided" if label is None or str(label).strip() == "" else str(label)
    st.metric(label="Score", value=score_text, help=f"Category: {label_text}")
    st.caption(f"Category: {label_text}")


def render_probabilities(probabilities: Mapping[str, Any] | None) -> None:
    """Display class probabilities without using st.dataframe."""
    st.subheader("Prediction Probabilities")

    if not probabilities or not isinstance(probabilities, Mapping):
        st.info("No prediction probabilities to display.")
        return

    rows = []
    for label, value in probabilities.items():
        try:
            number = float(value)
            if not math.isfinite(number):
                number = None
        except (TypeError, ValueError):
            number = None
        rows.append((str(label), number))

    for label, number in rows:
        if number is None:
            st.write(f"**{label}:** unavailable")
        else:
            st.write(f"**{label}:** {number:.4f} ({number:.1%})")

    valid = [(label, value) for label, value in rows
             if value is not None]
    if valid:
        chart = go.Figure(go.Bar(
            x=[label for label, _ in valid],
            y=[value for _, value in valid],
            marker_color="#9D7CFF",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        ))
        chart.update_layout(
            title="Model output probabilities",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            height=280,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(7,16,39,0.72)",
        )
        st.plotly_chart(chart, use_container_width=True)


def render_explanation(explanation: Any) -> None:
    """Render caller-supplied explanation text without generating new content."""
    st.subheader("Explanation")
    if explanation is None or str(explanation).strip() == "":
        st.info("No explanation is available.")
        return
    st.write(str(explanation))


def render_history(history: pd.DataFrame | Sequence[Mapping[str, Any]] | None) -> None:
    """Display prediction history without using st.dataframe."""
    st.subheader("Event History")

    if history is None:
        st.info("No event history to display.")
        return

    try:
        if isinstance(history, pd.DataFrame):
            records = history.to_dict(orient="records")
        else:
            records = list(history)
    except (TypeError, ValueError):
        st.warning("Event history could not be displayed.")
        return

    if not records:
        st.info("No event history to display.")
        return

    for index, record in enumerate(records, start=1):
        with st.expander(f"Prediction record {index}"):
            if isinstance(record, Mapping):
                for key, value in record.items():
                    st.write(f"**{key}:** {value}")
            else:
                st.write(str(record))


def _render_metric_cards(data: Mapping[str, Any]) -> None:
    """Render isolated, animated presentation cards for supplied dashboard values."""
    events = data.get("events")
    event_count = events.size if isinstance(events, np.ndarray) else len(events or [])
    recorded = bool(data.get("recorded_data"))
    cards = [
        ("Heart rate", data.get("heart_rate", "Not calculated"), "No rate calculation is performed by this UI."),
        (
            "Signal status",
            data.get("signal_status", "Not supplied"),
            "Recorded source" if recorded else "Mock display value",
        ),
        ("Model risk indicator", data.get("risk_score", "--"), "Experimental model-derived indicator; not clinically validated"),
        ("Events in window", event_count, "Markers supplied by the caller"),
    ]
    card_markup = "".join(
        "<article class='metric-card'>"
        "<span class='status-dot' aria-hidden='true'></span>"
        f"<p>{escape(str(label))}</p><strong>{escape(str(value))}</strong>"
        f"<small>{escape(str(note))}</small></article>"
        for label, value, note in cards
    )
    components.html(
        f"""
        <style>
          * {{ box-sizing: border-box; }}
          body {{ margin: 0; background: transparent; color: #e5edf6; font-family: sans-serif; }}
          .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
          .metric-card {{ position: relative; overflow: hidden; min-height: 118px; padding: 18px;
            border: 1px solid rgba(108, 101, 255, .35); border-radius: 15px;
            background: linear-gradient(140deg, rgba(74, 52, 151, .76), rgba(12, 27, 55, .96) 68%);
            box-shadow: 0 10px 26px rgba(0, 0, 0, .27), inset 0 1px 0 rgba(255, 255, 255, .08);
            transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease; }}
          .metric-card:nth-child(2) {{ border-color: rgba(47, 226, 213, .38); background: linear-gradient(140deg, rgba(12, 113, 126, .76), rgba(10, 25, 53, .96) 68%); }}
          .metric-card:nth-child(3) {{ border-color: rgba(255, 120, 138, .42); background: linear-gradient(140deg, rgba(139, 55, 97, .72), rgba(30, 23, 54, .96) 68%); }}
          .metric-card:nth-child(4) {{ border-color: rgba(85, 195, 255, .38); background: linear-gradient(140deg, rgba(26, 86, 151, .72), rgba(12, 25, 55, .96) 68%); }}
          .metric-card:hover {{ transform: translateY(-3px); border-color: rgba(151, 236, 255, .7); box-shadow: 0 15px 32px rgba(0, 0, 0, .32), 0 0 22px rgba(63, 228, 221, .12); }}
          .metric-card p {{ margin: 0 0 10px; color: #d0ddf1; font-size: .78rem; text-transform: uppercase; letter-spacing: .055em; }}
          .metric-card strong {{ display: block; color: #f8fbff; font-size: 1.32rem; line-height: 1.2; transition: color .3s ease, text-shadow .3s ease; text-shadow: 0 0 18px rgba(210, 243, 255, .2); }}
          .metric-card small {{ display: block; margin-top: 10px; color: #afc3df; font-size: .72rem; }}
          .status-dot {{ position: absolute; top: 16px; right: 16px; width: 8px; height: 8px; border-radius: 50%;
            background: #45f4d4; box-shadow: 0 0 12px rgba(69, 244, 212, .88); }}
          @media (max-width: 680px) {{ .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
          @media (prefers-reduced-motion: reduce) {{ .metric-card {{ transition: none; }} }}
        </style>
        <section class='metrics' aria-label='Dashboard metrics'>{card_markup}</section>
        <script src='https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.2/anime.min.js'></script>
        <script>
          const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
          if (window.anime && !reduceMotion) {{
            let playEntrance = true;
            try {{
              playEntrance = !sessionStorage.getItem('ecg-metric-entrance');
              sessionStorage.setItem('ecg-metric-entrance', 'true');
            }} catch (error) {{ /* Render remains usable when storage is unavailable. */ }}
            if (playEntrance) {{
              anime.set('.metric-card', {{ opacity: 0, translateY: 10 }});
              anime({{ targets: '.metric-card', opacity: [0, 1], translateY: [10, 0],
                delay: anime.stagger(70), duration: 520, easing: 'easeOutCubic' }});
            }}
            anime({{ targets: '.status-dot', scale: [1, 1.12, 1], opacity: [.68, 1, .68],
              duration: 1000, easing: 'easeInOutSine', loop: true }});
          }}
        </script>
        """,
        height=142,
        scrolling=False,
    )


def render_dashboard(data: Mapping[str, Any]) -> None:
    """Render the complete dashboard using a caller-provided data dictionary.

    Expected keys are ``time``, ``signal``, ``events``, ``alerts``,
    ``risk_score``, ``risk_label``, ``probabilities``, ``explanation``, and
    ``history``. Optional ``heart_rate``, ``signal_status``, and
    ``recorded_data`` values populate and label the metric row. Only ``time``
    and ``signal`` are needed for a waveform; all other keys have sensible
    display defaults.

    Returns:
        None. Components are rendered directly through Streamlit.
    """
    if not isinstance(data, Mapping):
        st.error("Dashboard data must be provided as a dictionary.")
        return

    _inject_dashboard_styles()
    # Disabled for the live monitoring dashboard.
    st.title("ECG Dashboard")
    _render_metric_cards(data)
    st.caption("Experimental research demonstration only. Model probabilities and indicators are not clinically validated and must not be used for medical decisions.")

    # The live app renders the verified ECG waveform directly in app.py.
    # Do not render a second waveform here.

    left_column, right_column = st.columns(2)
    with left_column:
        render_alert_cards(data.get("alerts"))
        render_explanation(data.get("explanation"))
    with right_column:
        render_risk_score(data.get("risk_score"), data.get("risk_label"))
        render_probabilities(data.get("probabilities"))

    render_history(data.get("history"))

