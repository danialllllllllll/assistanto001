# Auto-update + safe deployment for Replit / always-on container

What this provides
- scripts/devserver.sh — launches your app and the auto-updater.
- scripts/start-app.sh — generic starter that writes a PID file.
- scripts/autoupdate.sh — loop that fetches, compares, pulls, runs tests, and restarts if safe.
- .replit — example Replit run configuration (so Replit will run the devserver script).

Principles
- Passive: script polls the Git remote periodically (INTERVAL) and only acts when remote HEAD differs.
- Safe: if there are uncommitted local changes, the update is skipped to avoid breaking learning in-progress.
- Test-before-restart: configure TEST_CMD to let the script run tests; if tests fail, it rolls back.
- Configurable: environment variables control branch, interval, test command, app command, and git remote.

How to install on Replit
1. Add these script files into `scripts/` in your repl (or repo).
2. Add the `.replit` file to the repo root (example provided).
3. Configure environment variables in Replit Secrets or the Replit UI:
   - APP_CMD — the command to run your app (default: `python main.py`).
   - TEST_CMD — optional command string to run your tests (e.g. `pytest -q` or `npm test`).
   - BRANCH — branch to track (default: `main`).
   - INTERVAL — poll interval in seconds (default: `60`).
   - GIT_REMOTE — optional, overrides `origin` remote URL.
4. Make sure "Always On" is enabled on the Replit (or keep a background process running).
5. Start the repl (Replit will run the `.replit` run command which launches the devserver).

Notes & tips
- The scripts avoid destructive behavior: they only do a `git.pull --ff-only` and they check working tree cleanliness.
- If you need a more advanced restart strategy (systemd, docker restart, graceful reload handlers), update `scripts/start-app.sh` to integrate with your app's shutdown/start hooks.
- If you prefer push-based deployment (GitHub Actions -> Replit), I can add a workflow that triggers a webhook / Replit API call; tell me if you'd like that next.