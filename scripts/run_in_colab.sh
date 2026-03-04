#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <endpoint> [repo_url]" >&2
  exit 1
fi

ENDPOINT="$1"
REPO_URL="${2:-https://github.com/mikewong23571/number_detection.git}"

npx --yes --package=git+https://github.com/mikewong23571/colab-vscode.git#main colab-cli -- exec "$ENDPOINT" -- bash -lc "rm -rf number_detection && git clone ${REPO_URL} && cd number_detection && pip install -q uv && uv sync && uv run number-detection --epochs 1 --device cuda --max-train-samples 20000 --max-test-samples 5000"

