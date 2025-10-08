#!/usr/bin/env bash
# Top-level devserver for Replit: starts the app and the autoupdater.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

# Start the app once (this will write .app.pid)
bash scripts/start-app.sh

# Start the autoupdater in background (it will pull and restart the app when needed)
bash scripts/autoupdate.sh &
# Start estimator if present
if [ -f "scripts/start-estimator.sh" ]; then
  bash scripts/start-estimator.sh &
fi
wait -n
