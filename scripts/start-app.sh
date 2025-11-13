#!/usr/bin/env bash
# Start the application and write its PID to .app.pid
# Configure APP_CMD env var (e.g. APP_CMD="python main.py" or "npm start")

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PID_FILE="${PID_FILE:-$REPO_DIR/.app.pid}"
APP_CMD="${APP_CMD:-python main.py}"

cd "$REPO_DIR"

# stop previous if still running
if [ -f "$PID_FILE" ]; then
  oldpid=$(cat "$PID_FILE" 2>/dev/null || true)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "Stopping previous app (pid $oldpid)..."
    kill "$oldpid" || true
    sleep 1
  fi
fi

echo "Starting app with command: $APP_CMD"
# start the app in background, detach so script can exit
bash -lc "$APP_CMD" &
echo $! > "$PID_FILE"
echo "App started (pid $(cat "$PID_FILE"))."