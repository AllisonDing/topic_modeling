#!/usr/bin/env bash
# Launch the interactive BERTopic dashboard.
#
# Uses the rapids-25.10 conda env (only one with streamlit + cuml.accel +
# bertopic + datamapplot all installed). Override with PYTHON env var if needed.
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-/home/allisond/miniconda/envs/rapids-25.10/bin/python}"

exec "$PYTHON" -m streamlit run topic_modeling_app.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-8501}" \
    --server.headless true
