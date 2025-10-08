#!/usr/bin/env bash
# Periodically check remote BRANCH; if changed, pull, run tests, and restart the app.
# Configure via env:
# - BRANCH (default: main)
# - INTERVAL (seconds, default: 60)
# - TEST_CMD (optional, a command string to run tests)
# - GIT_REMOTE (optional)
# - REPO_DIR (optional, default: $PWD)
# - PID_FILE (optional, default: .app.pid)
# - START_SCRIPT (optional, script to start app; default: scripts/start-app.sh)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
BRANCH="${BRANCH:-main}"
INTERVAL="${INTERVAL:-60}"
TEST_CMD="${TEST_CMD:-}"
GIT_REMOTE="${GIT_REMOTE:-}"
PID_FILE="${PID_FILE:-$REPO_DIR/.app.pid}"
START_SCRIPT="${START_SCRIPT:-$REPO_DIR/scripts/start-app.sh}"

cd "$REPO_DIR"

# Optionally override origin remote URL
if [ -n "$GIT_REMOTE" ]; then
  git remote remove origin 2>/dev/null || true
  git remote add origin "$GIT_REMOTE"
fi

# ensure we have branch refs
git fetch --all --prune

echo "Auto-update watcher started (branch=$BRANCH, interval=${INTERVAL}s)."

while true; do
  # Refresh remote refs
  git fetch origin "$BRANCH" --quiet || true

  # Determine current and remote commit
  # Use the checked-out branch if HEAD is on the same branch, else compare branch names.
  LOCAL="$(git rev-parse "$BRANCH" 2>/dev/null || true)"
  REMOTE="$(git rev-parse "origin/$BRANCH" 2>/dev/null || true)"

  if [ -z "$LOCAL" ] || [ -z "$REMOTE" ]; then
    echo "Warning: could not determine local/remote refs (local=$LOCAL remote=$REMOTE). Retrying in $INTERVAL s..."
    sleep "$INTERVAL"
    continue
  fi

  if [ "$LOCAL" != "$REMOTE" ]; then
    echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") - New commit detected: local=$LOCAL remote=$REMOTE"

    # Do not override local uncommitted changes
    if [ -n "$(git status --porcelain)" ]; then
      echo "Local uncommitted changes present; skipping auto-update to avoid breaking work-in-progress."
      sleep "$INTERVAL"
      continue
    fi

    # Try a safe fast-forward pull
    echo "Pulling changes (ff-only)..."
    if git pull --ff-only origin "$BRANCH"; then
      echo "Pulled successfully."

      # Run tests if configured
      if [ -n "$TEST_CMD" ]; then
        echo "Running tests: $TEST_CMD"
        if bash -lc "$TEST_CMD"; then
          echo "Tests passed."
        else
          echo "Tests failed. Rolling back to previous commit ($LOCAL)."
          git reset --hard "$LOCAL"
          sleep "$INTERVAL"
          continue
        fi
      else
        echo "No TEST_CMD set; skipping tests."
      fi

      # Restart app gracefully by using the start script which will kill previous pid
      if [ -f "$START_SCRIPT" ]; then
        echo "Restarting app using $START_SCRIPT"
        bash "$START_SCRIPT"
      else
        echo "Start script $START_SCRIPT not found; skipping automatic restart."
      fi
    else
      echo "git pull failed (non fast-forward or conflict). Skipping restart and leaving working tree untouched."
    fi
  fi

  sleep "$INTERVAL"
done
