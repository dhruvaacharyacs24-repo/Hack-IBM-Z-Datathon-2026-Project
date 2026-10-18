#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/streamlit" ]]; then
  echo "Streamlit was not found in .venv. Install the project dependencies first."
  exit 1
fi

exec .venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501
