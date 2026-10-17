# ibm-dataz_hackathon

# CardioSentinel — Explainable ECG Arrhythmia Detection

**IBM Z Datathon 2026 | Real-Time AI for Critical Decisions**

CardioSentinel is an experimental ECG analysis prototype that uses a PyTorch 1D convolutional neural network (CNN) to classify ECG signal windows and presents the results through an interactive Streamlit dashboard. It combines waveform visualization, class probabilities, signal statistics, inference latency, session history, and Grad-CAM-based explainability.

> **Disclaimer:** CardioSentinel is a research and demonstration prototype. It is not a clinically validated medical device and must not be used to diagnose or treat a medical condition.

## Problem

ECG waveforms contain patterns that can be difficult to interpret. CardioSentinel explores how machine learning and visual explanations can make ECG classification outputs easier to inspect.

## What We Built

- ECG data workflow based on the MIT-BIH Arrhythmia Database.
- Three-second ECG windows sampled at 360 Hz (1,080 samples per window).
- PyTorch 1D CNN inference using the trained model weights, where available.
- Application-level Normal, Watch, and Critical categories.
- Threshold-based adjustment of the Critical prediction.
- Interactive Streamlit dashboard for record/channel selection and window analysis.
- Class-probability display, waveform visualization, signal statistics, and inference timing.
- Grad-CAM waveform attribution to help inspect regions associated with a model prediction.
- Session-level prediction history.

The categories and threshold are prototype software rules, not clinically validated risk levels.

## Architecture

1. **Data:** MIT-BIH Arrhythmia Database.
2. **Signal input:** Select a record and channel, then extract a three-second window.
3. **Inference:** Pass the ECG window through the trained PyTorch 1D CNN.
4. **Decision logic:** Present model probabilities and apply the configured Critical threshold rule.
5. **Explainability:** Generate a Grad-CAM attribution visualization.
6. **Interface:** Display the output in Streamlit.
7. **Platform:** The application and inference workload run on IBM LinuxONE Community Cloud.

See [`docs/architecture.png`](docs/architecture.png).

## Technology Stack

- Python
- PyTorch
- Streamlit
- WFDB
- NumPy, SciPy, and Pandas
- scikit-learn, if used by the checked-in evaluation workflow

## Dataset

The project uses the MIT-BIH Arrhythmia Database from PhysioNet.

Dataset page: https://physionet.org/content/mitdb/1.0.0/

Follow the dataset's own terms and citation requirements. The dataset is not bundled with this repository by default.

## Repository Layout

- `app.py` — Streamlit dashboard.
- `core.py` — model loading, ECG-window inference, and prediction rules.
- `ui_components.py` — dashboard components.
- `streaming.py` — streaming-related functionality; inspect the implementation to understand its current scope.
- `explain.py` — explanation-related integration.
- `explainability/` — Grad-CAM implementation.
- `evaluation/` — evaluation scripts and metric utilities, subject to completion and verification.
- `models/` — instructions for obtaining or placing model weights.
- `docs/` — architecture, screenshots, pitch deck, and limitations.
- `scripts/` — helper scripts.

## Run Locally

Use a compatible Python environment. The commands below assume dependencies have been installed and the required dataset and model weights have been made available.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If you are on Windows, activate the environment using `.venv\Scripts\activate` instead.

See [`models/README.md`](models/README.md) and [`data/README.md`](data/README.md) for setup notes.

## Evaluation

Evaluation results should be reported only after the evaluation script has been run and the outputs have been checked. Do not interpret example screenshots or placeholder numbers as measured performance.

See [`evaluation/README.md`](evaluation/README.md).

## IBM LinuxONE

The application has been run on IBM LinuxONE Community Cloud in an Ubuntu environment on the `s390x` architecture. This demonstrates execution on that platform; it does not by itself establish clinical readiness, production scalability, or security certification.

## Limitations and Future Work

- Experimental model; not clinically validated.
- Performance must be assessed with appropriate held-out data and documented metrics.
- The current workflow analyzes stored ECG recordings and selected windows; it is not a verified live-patient monitoring system.
- Grad-CAM is an attribution aid, not proof that a prediction is medically correct.
- Session history is not durable storage.
- Further work may include more rigorous evaluation, robust streaming integration, deployment hardening, and clinical collaboration.

See [`docs/limitations.md`](docs/limitations.md).

## Open-Source References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- PyTorch: https://pytorch.org/
- Streamlit: https://streamlit.io/
- WFDB: https://wfdb.readthedocs.io/
- NumPy: https://numpy.org/
- SciPy: https://scipy.org/
- Pandas: https://pandas.pydata.org/
- scikit-learn: https://scikit-learn.org/ (if used in the checked-in code)

## License

See [`LICENSE`](LICENSE) for the license covering this repository's original code. Third-party datasets, libraries, and assets remain subject to their respective terms.

## Citation

See [`CITATION.cff`](CITATION.cff). Replace any placeholder team, author, or repository details before publishing.
