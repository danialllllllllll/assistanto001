from typing import Optional, Dict, Any
import socket
import threading
import time
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NetworkStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_check: float = time.time()
    average_response_time: float = 0.0

class NetworkOptimizer:
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.host = host
        self.port = port
        self.stats = NetworkStats()
        self.lock = threading.Lock()
        self._health_check_thread = None
        self._should_run = True

    def start_monitoring(self):
        """Start background health monitoring"""
        if self._health_check_thread is None:
            self._health_check_thread = threading.Thread(
                target=self._monitor_health,
                daemon=True
            )
            self._health_check_thread.start()

    def stop_monitoring(self):
        """Stop background health monitoring"""
        self._should_run = False
        if self._health_check_thread:
            self._health_check_thread.join()

    def _monitor_health(self):
        """Background thread for monitoring network health"""
        while self._should_run:
            try:
                # Check if port is still available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex((self.host, self.port))

                    if result == 0:
                        with self.lock:
                            self.stats.successful_requests += 1
                    else:
                        with self.lock:
                            self.stats.failed_requests += 1

                    self.stats.total_requests += 1

            except Exception as e:
                print(f"Health check error: {str(e)}")

            time.sleep(60)  # Check every minute

    def get_network_stats(self) -> Dict[str, Any]:
        """Get current network statistics"""
        with self.lock:
            current_time = time.time()
            uptime = current_time - self.stats.last_check

            stats = {
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'host': self.host,
                'port': self.port,
                'total_requests': self.stats.total_requests,
                'successful_requests': self.stats.successful_requests,
                'failed_requests': self.stats.failed_requests,
                'uptime_seconds': uptime,
                'success_rate': (self.stats.successful_requests / 
                               max(1, self.stats.total_requests) * 100),
                'average_response_time': self.stats.average_response_time
            }

            return stats

    def update_response_time(self, response_time: float):
        """Update average response time statistics"""
        with self.lock:
            if self.stats.average_response_time == 0:
                self.stats.average_response_time = response_time
            else:
                # Exponential moving average
                alpha = 0.1
                self.stats.average_response_time = (
                    (1 - alpha) * self.stats.average_response_time + 
                    alpha * response_time
                )