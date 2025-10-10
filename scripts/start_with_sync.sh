
#!/bin/bash
# Start training with GitHub auto-sync

echo "Starting AI Training System with GitHub Auto-Sync..."

# Start GitHub sync in background
python scripts/auto_github_sync.py &
SYNC_PID=$!

# Start training
python train_advanced_ai.py

# Cleanup on exit
kill $SYNC_PID 2>/dev/null
