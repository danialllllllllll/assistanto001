
#!/usr/bin/env python3
"""
Automated GitHub Repository Sync
Monitors code mutations and auto-commits to GitHub
"""
import os
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

class GitHubAutoSync:
    def __init__(self, repo_path='.', check_interval=300):
        self.repo_path = repo_path
        self.check_interval = check_interval
        self.last_sync = None
        
    def git_command(self, *args):
        """Execute git command"""
        try:
            result = subprocess.run(
                ['git'] + list(args),
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Git error: {e.stderr}")
            return None
    
    def has_changes(self):
        """Check if there are uncommitted changes"""
        status = self.git_command('status', '--porcelain')
        return bool(status)
    
    def commit_and_push(self, message=None):
        """Commit and push changes"""
        if not self.has_changes():
            return False
        
        # Add all changes
        self.git_command('add', '.')
        
        # Create commit message
        if not message:
            mutation_count = self.get_mutation_count()
            message = f"Auto-sync: Code mutations and training updates ({mutation_count} changes) - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Commit
        self.git_command('commit', '-m', message)
        
        # Push to remote
        result = self.git_command('push', 'origin', 'main')
        
        if result is not None:
            print(f"✓ Synced to GitHub: {message}")
            self.last_sync = datetime.now()
            return True
        return False
    
    def get_mutation_count(self):
        """Get number of mutations from training state"""
        try:
            with open('training_state.json', 'r') as f:
                state = json.load(f)
                return state.get('generation', 0)
        except:
            return 0
    
    def auto_sync_loop(self):
        """Main sync loop"""
        print("🔄 GitHub Auto-Sync Started")
        print(f"   Checking every {self.check_interval} seconds")
        
        while True:
            try:
                if self.has_changes():
                    self.commit_and_push()
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                print("\n🛑 Auto-sync stopped")
                break
            except Exception as e:
                print(f"Sync error: {e}")
                time.sleep(self.check_interval)

if __name__ == '__main__':
    syncer = GitHubAutoSync(check_interval=600)  # Sync every 10 minutes
    syncer.auto_sync_loop()
