#!/usr/bin/env bash
# Periodically check remote BRANCH; if changed, pull, run tests, and restart the app.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
BRANCH="${BRANCH:-main}"
INTERVAL="${INTERVAL:-60}"
TEST_CMD="${TEST_CMD:-}"
GIT_REMOTE="${GIT_REMOTE:-}"
PID_FILE="${PID_FILE:-$REPO_DIR/.app.pid}"
START_SCRIPT="${START_SCRIPT:-$REPO_DIR/scripts/start-app.sh}"

cd "$REPO_DIR"
if [ -n "$GIT_REMOTE" ]; then
  git remote remove origin 2>/dev/null || true
  git remote add origin "$GIT_REMOTE"
fi
git fetch --all --prune
echo "Auto-update watcher started (branch=$BRANCH, interval=${INTERVAL}s)."

while true; do
  git fetch origin "$BRANCH" --quiet || true
  LOCAL="$(git rev-parse "$BRANCH" 2>/dev/null || true)"
  REMOTE="$(git rev-parse "origin/$BRANCH" 2>/dev/null || true)"

  if [ -z "$LOCAL" ] || [ -z "$REMOTE" ]; then
    echo "Warning: could not determine local/remote refs (local=$LOCAL remote=$REMOTE). Retrying in $INTERVAL s..."
    sleep "$INTERVAL"
    continue
  fi

  if [ "$LOCAL" != "$REMOTE" ]; then
    echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") - New commit detected: local=$LOCAL remote=$REMOTE"
    if [ -n "$(git status --porcelain)" ]; then
      echo "Local uncommitted changes present; skipping auto-update to avoid breaking work-in-progress."
      sleep "$INTERVAL"
      continue
    fi

    echo "Pulling changes (ff-only)..."
    if git pull --ff-only origin "$BRANCH"; then
      echo "Pulled successfully."
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
