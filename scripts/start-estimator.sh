#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
INTERVAL="${ESTIMATOR_INTERVAL:-60}"  # seconds between estimator runs

cd "$REPO_DIR"

# Run once immediately, then loop
python3 scripts/learning_estimator.py || true
while true; do
  sleep "$INTERVAL"
  python3 scripts/learning_estimator.py || true
done
