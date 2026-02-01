#!/usr/bin/env python3
"""
TradingSwarmCore Watchdog
Self-healing wrapper that restarts the trading agent on crash.
"""

import os
import subprocess
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Auto-load .env file
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

# Config
AGENT_SCRIPT = Path(__file__).parent / "agent.py"
MAX_RESTARTS = 10
RESTART_WINDOW_HOURS = 1
BACKOFF_BASE_SECONDS = 30
BACKOFF_MAX_SECONDS = 300
STATUS_FILE = Path(__file__).parent / "watchdog_status.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - WATCHDOG - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / "watchdog_log.txt"),
        logging.StreamHandler()
    ]
)

class Watchdog:
    def __init__(self):
        self.restart_times = []
        self.total_restarts = 0
        self.start_time = datetime.now()
        self.current_process = None
        
    def update_status(self, status: str, extra: dict = None):
        import json
        data = {
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_restarts": self.total_restarts,
            "restarts_last_hour": len([t for t in self.restart_times 
                                       if (datetime.now() - t).total_seconds() < 3600]),
            "pid": self.current_process.pid if self.current_process else None,
            **(extra or {})
        }
        STATUS_FILE.write_text(json.dumps(data, indent=2))
        
    def should_restart(self) -> bool:
        # Clean old restart times
        cutoff = datetime.now()
        self.restart_times = [t for t in self.restart_times 
                              if (cutoff - t).total_seconds() < RESTART_WINDOW_HOURS * 3600]
        
        if len(self.restart_times) >= MAX_RESTARTS:
            logging.error(f"Max restarts ({MAX_RESTARTS}) in {RESTART_WINDOW_HOURS}h window reached. Giving up.")
            return False
        return True
    
    def get_backoff(self) -> int:
        recent = len(self.restart_times)
        backoff = min(BACKOFF_BASE_SECONDS * (2 ** recent), BACKOFF_MAX_SECONDS)
        return backoff
    
    def run_agent(self):
        logging.info(f"Starting agent: {AGENT_SCRIPT}")
        self.current_process = subprocess.Popen(
            [sys.executable, str(AGENT_SCRIPT)],
            cwd=AGENT_SCRIPT.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        self.update_status("running")
        
        # Stream output
        for line in self.current_process.stdout:
            print(line, end='')
            
        self.current_process.wait()
        return self.current_process.returncode
    
    def run(self):
        logging.info("Watchdog starting")
        self.update_status("starting")
        
        while True:
            try:
                exit_code = self.run_agent()
                logging.warning(f"Agent exited with code {exit_code}")
                self.update_status("crashed", {"last_exit_code": exit_code})
                
                if exit_code == 0:
                    logging.info("Agent exited cleanly (code 0). Stopping watchdog.")
                    self.update_status("stopped_clean")
                    break
                    
                if not self.should_restart():
                    self.update_status("stopped_max_restarts")
                    break
                    
                self.restart_times.append(datetime.now())
                self.total_restarts += 1
                backoff = self.get_backoff()
                
                logging.info(f"Restarting in {backoff}s (restart #{self.total_restarts})")
                self.update_status("waiting_restart", {"backoff_seconds": backoff})
                time.sleep(backoff)
                
            except KeyboardInterrupt:
                logging.info("Watchdog interrupted by user")
                if self.current_process:
                    self.current_process.terminate()
                self.update_status("stopped_user")
                break
            except Exception as e:
                logging.error(f"Watchdog error: {e}")
                self.update_status("error", {"error": str(e)})
                time.sleep(60)


if __name__ == "__main__":
    watchdog = Watchdog()
    watchdog.run()
