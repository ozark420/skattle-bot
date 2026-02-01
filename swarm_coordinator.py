#!/usr/bin/env python3
"""
Swarm Coordinator
Master orchestrator that runs and coordinates all agents in the trading swarm.
"""

import subprocess
import sys
import time
import json
import threading
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - COORDINATOR - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / "coordinator_log.txt"),
        logging.StreamHandler()
    ]
)

SWARM_DIR = Path(__file__).parent
STATUS_FILE = SWARM_DIR / "swarm_status.json"

class SwarmCoordinator:
    def __init__(self):
        self.agents = {}
        self.processes = {}
        self.start_time = datetime.now()
        self.running = True
        
    def update_status(self, status: str, details: dict = None):
        """Update swarm status file"""
        data = {
            'status': status,
            'updated_at': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'agents': {
                name: {
                    'running': proc.poll() is None if proc else False,
                    'pid': proc.pid if proc and proc.poll() is None else None
                }
                for name, proc in self.processes.items()
            },
            **(details or {})
        }
        STATUS_FILE.write_text(json.dumps(data, indent=2))
    
    def start_agent(self, name: str, script: str, delay: int = 0):
        """Start an agent subprocess"""
        if delay:
            logging.info(f"Starting {name} in {delay}s...")
            time.sleep(delay)
        
        script_path = SWARM_DIR / script
        if not script_path.exists():
            logging.error(f"Agent script not found: {script_path}")
            return None
        
        logging.info(f"Starting agent: {name}")
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=SWARM_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        self.processes[name] = proc
        
        # Start output reader thread
        def read_output():
            for line in proc.stdout:
                print(f"[{name}] {line}", end='')
        
        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()
        
        return proc
    
    def check_agents(self):
        """Check agent health and restart if needed"""
        for name, proc in list(self.processes.items()):
            if proc.poll() is not None:
                logging.warning(f"Agent {name} died (exit code: {proc.returncode}). Restarting...")
                
                # Map name back to script
                script_map = {
                    'sentiment': 'agents/sentiment_agent.py',
                    'risk': 'agents/risk_manager.py',
                    'trader': 'agent.py'
                }
                
                if name in script_map:
                    time.sleep(5)  # Brief delay before restart
                    self.start_agent(name, script_map[name])
    
    def run(self):
        """Start and coordinate all agents"""
        logging.info("=" * 60)
        logging.info("🎲 TRADING SWARM COORDINATOR STARTING")
        logging.info("=" * 60)
        
        self.update_status('starting')
        
        # Start agents in sequence
        # 1. Sentiment agent first (needs time to gather initial data)
        self.start_agent('sentiment', 'agents/sentiment_agent.py')
        
        # 2. Risk manager
        self.start_agent('risk', 'agents/risk_manager.py', delay=5)
        
        # 3. Main trading agent (wait for sentiment to have initial data)
        self.start_agent('trader', 'agent.py', delay=30)
        
        logging.info("")
        logging.info("All agents started. Swarm is running.")
        logging.info("Dashboard: http://localhost:8420")
        logging.info("")
        
        self.update_status('running')
        
        # Monitor loop
        check_interval = 30
        while self.running:
            try:
                time.sleep(check_interval)
                self.check_agents()
                self.update_status('running')
                
                # Log summary every 5 minutes
                uptime = (datetime.now() - self.start_time).total_seconds()
                if int(uptime) % 300 < check_interval:
                    running = sum(1 for p in self.processes.values() if p.poll() is None)
                    logging.info(f"Swarm health: {running}/{len(self.processes)} agents running")
                    
            except KeyboardInterrupt:
                logging.info("Shutdown requested...")
                self.running = False
            except Exception as e:
                logging.error(f"Coordinator error: {e}")
        
        # Cleanup
        logging.info("Stopping all agents...")
        for name, proc in self.processes.items():
            if proc.poll() is None:
                proc.terminate()
                logging.info(f"Stopped {name}")
        
        self.update_status('stopped')
        logging.info("Swarm coordinator stopped.")


if __name__ == "__main__":
    coordinator = SwarmCoordinator()
    coordinator.run()
