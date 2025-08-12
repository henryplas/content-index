#!/usr/bin/env bash
set -euo pipefail

# Detect conda/Studio and skip venv if present
IS_CONDA=0
if command -v conda >/dev/null 2>&1 || [[ -n "${CONDA_PREFIX:-}" ]]; then
  IS_CONDA=1
fi

# 0) Ensure ffmpeg (install via conda if missing and conda is available)
if ! command -v ffmpeg >/dev/null 2>&1; then
  if [[ $IS_CONDA -eq 1 ]]; then
    echo "ffmpeg not found. Installing via conda-forge…"
    if command -v mamba >/dev/null 2>&1; then
      mamba install -y -c conda-forge ffmpeg
    else
      conda install -y -c conda-forge ffmpeg
    fi
  else
    echo "ERROR: ffmpeg not found and no conda detected. Please install ffmpeg, then rerun."
    exit 1
  fi
fi

# 1) Python env: use existing conda env if present; otherwise create venv
if [[ $IS_CONDA -eq 1 ]]; then
  echo "Using existing conda environment at: ${CONDA_PREFIX:-<unknown>}"
else
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  source .venv/bin/activate
fi

# 2) Upgrade pip and install deps
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

# 3) Ensure PYTHONPATH so 'src' is importable
export PYTHONPATH=.

# 4) Download sample public videos (Blender Open Movies)
mkdir -p data/raw
if [[ ! -f data/raw/bbb_640x360.mp4 ]]; then
  echo "Downloading Big Buck Bunny…"
  curl -L -o data/raw/bbb_640x360.mp4 https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_640x360.m4v
fi
if [[ ! -f data/raw/sintel_480p.mp4 ]]; then
  echo "Downloading Sintel…"
  curl -L -o data/raw/sintel_480p.mp4 https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4
fi
if [[ ! -f data/raw/tears_720p.mov ]]; then
  echo "Downloading Tears of Steel…"
  curl -L -o data/raw/tears_720p.mov https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov
fi

# 5) Build manifest
python - << 'PY'
from src.util.preprocess import build_manifest
build_manifest("data/raw", "artifacts/manifest.jsonl")
print("Wrote artifacts/manifest.jsonl")
PY

# 6) Preprocess (frames + audio)
python - << 'PY'
from src.util.preprocess import preprocess_assets
preprocess_assets("artifacts/manifest.jsonl", "artifacts")
print("Wrote artifacts/preprocessed.jsonl, frames/, audio/")
PY

# 7) Transcribe with Whisper (uses default 'small'; override with WHISPER_MODEL env)
python - << 'PY'
import os
from src.util.transcribe import transcribe_all
model = os.environ.get("WHISPER_MODEL","small")
transcribe_all("artifacts/preprocessed.jsonl", "artifacts/transcripts", model_name=model)
print("Transcripts in artifacts/transcripts/")
PY

# 8) Build embeddings + Faiss index
python - << 'PY'
from src.index.build_index import build
build("artifacts/preprocessed.jsonl", "artifacts/transcripts", "artifacts", alpha=0.7)
PY

# 9) Quick evaluation
python src/index/eval_small.py

# 10) Serve API
echo ""
echo "Starting API at http://127.0.0.1:8000 ..."
uvicorn src.serve.app:app --host 127.0.0.1 --port 8000
