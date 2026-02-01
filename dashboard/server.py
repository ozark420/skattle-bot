#!/usr/bin/env python3
"""
Skattle_Bot Dashboard Server
Serves the dashboard UI and provides API endpoints for status data.
"""

import http.server
import socketserver
import json
import os
import re
import base64
import requests
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime

# Auto-load .env file from parent directory
env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

# Bankr API for position fetching
BANKR_API_KEY = os.environ.get("BANKR_API_KEY", "")
BANKR_API_URL = "https://api.bankr.bot/agent"

# Cache for positions (updated every 60 seconds)
positions_cache = {
    'positions': [],
    'total_collateral': 0,
    'total_pnl': 0,
    'usdc_balance': 0,
    'updated_at': None
}

PORT = 8420
SWARM_DIR = Path(__file__).parent.parent
DASHBOARD_DIR = Path(__file__).parent

# ============================================================
# SECURITY: Dashboard is PUBLIC (view-only) - API keys stay hidden in .env
# ============================================================
AUTH_ENABLED = False  # Public dashboard - everyone can watch Skattle trade
AUTH_USERNAME = "ozark"
AUTH_PASSWORD = "unused"
# ============================================================

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)
    
    def check_auth(self):
        """Check basic authentication"""
        if not AUTH_ENABLED:
            return True
        
        auth_header = self.headers.get('Authorization')
        if auth_header is None:
            return False
        
        try:
            auth_type, credentials = auth_header.split(' ', 1)
            if auth_type.lower() != 'basic':
                return False
            decoded = base64.b64decode(credentials).decode('utf-8')
            username, password = decoded.split(':', 1)
            return username == AUTH_USERNAME and password == AUTH_PASSWORD
        except:
            return False
    
    def send_auth_required(self):
        """Send 401 Unauthorized response"""
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="Skattle_Bot Dashboard"')
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<h1>401 - Authentication Required</h1>')
    
    def do_GET(self):
        # Check authentication if enabled
        if not self.check_auth():
            self.send_auth_required()
            return
        
        parsed = urlparse(self.path)
        
        if parsed.path == '/api/status':
            self.send_status_response()
        elif parsed.path == '/api/trades':
            self.send_trades_response()
        elif parsed.path == '/api/logs':
            self.send_logs_response()
        elif parsed.path == '/api/social':
            self.send_social_response()
        elif parsed.path == '/api/skills':
            self.send_skills_response()
        else:
            super().do_GET()
    
    def send_json(self, data):
        content = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)
    
    def send_status_response(self):
        """Aggregate status from all sources"""
        risk_state = self.get_risk_state()
        
        # Merge live_positions with risk_state fallback for reliability
        live_pos = dict(positions_cache)
        # If Bankr USDC balance is 0 or missing, use risk_state balance
        if not live_pos.get('usdc_balance') or live_pos['usdc_balance'] == 0:
            live_pos['usdc_balance'] = risk_state.get('balance', 0)
        # Keep last known values - never reset to 0
        if live_pos.get('updated_at') is None:
            live_pos['usdc_balance'] = risk_state.get('balance', 0)
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'swarm_status': self.get_swarm_status(),
            'risk_state': risk_state,
            'signals': self.get_signals(),
            'trades': self.get_recent_trades(),
            'winRate': self.calculate_win_rate(),
            'positions': self.get_active_positions(),
            'live_positions': live_pos
        }
        self.send_json(status)
    
    def get_swarm_status(self):
        """Get swarm coordinator status"""
        status_file = SWARM_DIR / 'swarm_status.json'
        if status_file.exists():
            try:
                return json.loads(status_file.read_text())
            except:
                pass
        return {'status': 'stopped', 'agents': {}}
    
    def get_risk_state(self):
        """Get risk manager state"""
        state_file = SWARM_DIR / 'risk_state.json'
        if state_file.exists():
            try:
                return json.loads(state_file.read_text())
            except:
                pass
        return {'balance': 0, 'daily_pnl': 0, 'total_pnl': 0}
    
    def get_signals(self):
        """Get sentiment signals"""
        signals_file = SWARM_DIR / 'signals.json'
        if signals_file.exists():
            try:
                return json.loads(signals_file.read_text())
            except:
                pass
        return {'signals': {}}
    
    def send_trades_response(self):
        self.send_json({'trades': self.get_recent_trades()})
    
    def send_logs_response(self):
        self.send_json({'logs': self.get_recent_logs()})
    
    def send_social_response(self):
        """Get social posts from 4claw and Moltbook"""
        posts = []
        
        # Load 4claw posts from cache file
        fourclaw_file = SWARM_DIR / 'social_4claw.json'
        if fourclaw_file.exists():
            try:
                data = json.loads(fourclaw_file.read_text(encoding='utf-8'))
                for post in data.get('posts', []):
                    post['platform'] = '4claw'
                    posts.append(post)
            except:
                pass
        
        # Load Moltbook posts from cache file
        moltbook_file = SWARM_DIR / 'social_moltbook.json'
        if moltbook_file.exists():
            try:
                data = json.loads(moltbook_file.read_text(encoding='utf-8'))
                for post in data.get('posts', []):
                    post['platform'] = 'moltbook'
                    posts.append(post)
            except:
                pass
        
        # Sort by date, newest first
        posts.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        self.send_json({'posts': posts})
    
    def send_skills_response(self):
        """Get skills log"""
        skills_file = SWARM_DIR / 'skills_log.json'
        if skills_file.exists():
            try:
                data = json.loads(skills_file.read_text(encoding='utf-8'))
                today = datetime.now().strftime('%Y-%m-%d')
                today_skills = [s for s in data.get('skills', []) 
                               if s.get('learnedAt', '').startswith(today)]
                active_skills = [s for s in data.get('skills', []) 
                                if s.get('status') == 'active']
                
                self.send_json({
                    'skills': data.get('skills', []),
                    'today': today_skills,
                    'total': len(data.get('skills', [])),
                    'todayCount': len(today_skills),
                    'activeCount': len(active_skills)
                })
                return
            except:
                pass
        
        self.send_json({'skills': [], 'today': [], 'total': 0, 'todayCount': 0, 'activeCount': 0})
    
    def get_watchdog_status(self):
        status_file = SWARM_DIR / 'watchdog_status.json'
        if status_file.exists():
            try:
                return json.loads(status_file.read_text())
            except:
                pass
        return {'status': 'not_running', 'uptime_seconds': 0, 'total_restarts': 0}
    
    def get_recent_trades(self):
        """Parse trades from agent log with full details"""
        trades = []
        log_file = SWARM_DIR / 'agent_log.txt'
        if not log_file.exists():
            return trades
        
        try:
            raw_lines = log_file.read_text().splitlines()[-1000:]
            
            # Join multi-line log entries (lines not starting with timestamp belong to previous)
            lines = []
            current_line = ""
            for raw_line in raw_lines:
                if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', raw_line):
                    if current_line:
                        lines.append(current_line)
                    current_line = raw_line
                else:
                    current_line += " " + raw_line.strip()
            if current_line:
                lines.append(current_line)
            
            for i, line in enumerate(lines):
                lower = line.lower()
                
                # Parse Bankrbot execution confirmations (most detailed)
                if 'bankrbot response' in lower and ('opened' in lower):
                    # Extract pair from "for PAIR:" 
                    pair_match = re.search(r'for (\w+):', line)
                    pair = pair_match.group(1).upper() if pair_match else 'UNKNOWN'
                    
                    # Extract direction
                    direction = 'long' if 'long' in lower else 'short' if 'short' in lower else 'unknown'
                    
                    # Extract collateral size (multiple formats)
                    size = 0
                    for pattern in [r'([\d.]+)\s*usdc\s*collateral', r'collateral[:\s]+\$?([\d.]+)', r'size[:\s]+\$?([\d.]+)\s*usdc']:
                        size_match = re.search(pattern, lower)
                        if size_match:
                            size = float(size_match.group(1))
                            break
                    
                    # Extract leverage
                    leverage = 0
                    lev_match = re.search(r'(\d+)x\s', lower)
                    if lev_match:
                        leverage = int(lev_match.group(1))
                    
                    # Extract entry price (multiple formats)
                    entry_price = 0
                    for pattern in [r'entry[:\s]+\$?([\d,]+\.?\d*)', r'entry price[:\s]+\$?([\d,]+\.?\d*)']:
                        entry_match = re.search(pattern, line, re.I)
                        if entry_match:
                            entry_price = float(entry_match.group(1).replace(',', ''))
                            break
                    
                    # Extract stop loss (multiple formats)
                    stop_loss = 0
                    for pattern in [r'stop loss[:\s]+\$?([\d,]+\.?\d*)', r'sl[:\s]+\$?([\d,]+\.?\d*)']:
                        sl_match = re.search(pattern, line, re.I)
                        if sl_match:
                            stop_loss = float(sl_match.group(1).replace(',', ''))
                            break
                    
                    # Extract take profit (multiple formats)
                    take_profit = 0
                    for pattern in [r'take profit[:\s]+\$?([\d,]+\.?\d*)', r'tp[:\s]+\$?([\d,]+\.?\d*)']:
                        tp_match = re.search(pattern, line, re.I)
                        if tp_match:
                            take_profit = float(tp_match.group(1).replace(',', ''))
                            break
                    
                    # Extract tx hash
                    tx_match = re.search(r'(0x[a-fA-F0-9]{10,})', line)
                    tx_hash = tx_match.group(1) if tx_match else ''
                    
                    trades.append({
                        'pair': pair,
                        'direction': direction,
                        'action': 'OPEN',
                        'entry_price': entry_price,
                        'size': size,
                        'leverage': leverage,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'tx_hash': tx_hash,
                        'pnl': 0,
                        'status': 'LIVE',
                        'time': self.extract_time(line)
                    })
                
                # Skip PENDING, REJECTED, FAILED - only show executed trades
                        
        except Exception as e:
            print(f"Error parsing trades: {e}")
        
        # Show all executed trades (LIVE status)
        executed = [t for t in trades if t.get('status') == 'LIVE']
        
        # Return most recent 30, newest first
        return list(reversed(executed[-30:]))
    
    def extract_time(self, line):
        """Extract timestamp from log line"""
        match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if match:
            return match.group(1)
        return datetime.now().strftime('%H:%M:%S')
    
    def get_current_balance(self):
        """Try to parse current balance from logs"""
        log_file = SWARM_DIR / 'agent_log.txt'
        if not log_file.exists():
            return 50.00
        
        try:
            lines = log_file.read_text().splitlines()[-100:]
            for line in reversed(lines):
                match = re.search(r'balance to \$?([\d.]+)', line.lower())
                if match:
                    return float(match.group(1))
        except:
            pass
        return 50.00
    
    def calculate_win_rate(self):
        """Calculate win rate from closed trades"""
        trades = self.get_recent_trades()
        closed = [t for t in trades if t.get('action') == 'close']
        if not closed:
            return 0
        wins = sum(1 for t in closed if t.get('pnl', 0) > 0)
        return round((wins / len(closed)) * 100, 1)
    
    def get_active_positions(self):
        """Get currently active positions"""
        # This would ideally come from a state file the agent maintains
        return {
            'BTC': {'active': False, 'direction': None},
            'ETH': {'active': False, 'direction': None},
            'SOL': {'active': False, 'direction': None},
            'GOLD': {'active': False, 'direction': None}
        }
    
    def get_recent_logs(self):
        """Get recent log entries"""
        logs = []
        for log_file in ['agent_log.txt', 'watchdog_log.txt']:
            path = SWARM_DIR / log_file
            if path.exists():
                try:
                    lines = path.read_text().splitlines()[-50:]
                    for line in lines:
                        level = 'info'
                        if 'error' in line.lower():
                            level = 'error'
                        elif 'warning' in line.lower():
                            level = 'warn'
                        logs.append({
                            'source': log_file,
                            'level': level,
                            'message': line[:200],
                            'time': self.extract_time(line)
                        })
                except:
                    pass
        
        # Sort by time and return last 100
        return sorted(logs, key=lambda x: x.get('time', ''), reverse=True)[:100]


def fetch_usdc_balance():
    """Fetch USDC balance from Bankr"""
    try:
        headers = {"Content-Type": "application/json", "X-API-Key": BANKR_API_KEY}
        body = {"prompt": "What is my USDC balance on Base?"}
        res = requests.post(f"{BANKR_API_URL}/prompt", headers=headers, json=body, timeout=15)
        job_id = res.json().get('jobId')
        if not job_id:
            return 0
        
        for _ in range(15):
            time.sleep(2)
            res = requests.get(f"{BANKR_API_URL}/job/{job_id}", headers=headers, timeout=15)
            data = res.json()
            if data.get('status') == 'completed':
                response = data.get('response', '')
                # Parse "USDC - 13.219507 $13.22" or "13.22 USDC"
                match = re.search(r'usdc\s*[-:]\s*([\d.]+)|([\d.]+)\s*usdc', response.lower())
                if match:
                    bal = match.group(1) or match.group(2)
                    return float(bal)
                break
            elif data.get('status') == 'failed':
                break
    except Exception as e:
        print(f"Error fetching USDC balance: {e}")
    return 0


def fetch_positions_from_bankr():
    """Fetch current positions and USDC balance from Bankr API"""
    global positions_cache
    try:
        headers = {"Content-Type": "application/json", "X-API-Key": BANKR_API_KEY}
        body = {"prompt": "What are my current open positions on Avantis? Show each position with collateral amount and PnL."}
        
        # Submit job
        res = requests.post(f"{BANKR_API_URL}/prompt", headers=headers, json=body, timeout=15)
        job_id = res.json().get('jobId')
        if not job_id:
            return
        
        # Wait for result
        for _ in range(30):
            time.sleep(2)
            res = requests.get(f"{BANKR_API_URL}/job/{job_id}", headers=headers, timeout=15)
            data = res.json()
            if data.get('status') == 'completed':
                response = data.get('response', '')
                positions = parse_positions_response(response)
                positions_cache['positions'] = positions
                positions_cache['total_collateral'] = sum(p.get('collateral', 0) for p in positions)
                positions_cache['total_pnl'] = sum(p.get('pnl', 0) for p in positions)
                
                # Extract USDC balance (multiple formats)
                usdc_match = re.search(r'usdc[:\s\-]*([\d.]+)|have ([\d.]+) usdc|([\d.]+)\s*usdc|\$([\d.]+)', response.lower())
                if usdc_match:
                    bal = usdc_match.group(1) or usdc_match.group(2) or usdc_match.group(3) or usdc_match.group(4)
                    if bal:
                        positions_cache['usdc_balance'] = float(bal)
                
                positions_cache['updated_at'] = datetime.now().isoformat()
                print(f"Updated: {len(positions)} positions, ${positions_cache['total_collateral']:.2f} collateral, ${positions_cache['usdc_balance']:.2f} USDC")
                break
            elif data.get('status') == 'failed':
                break
    except Exception as e:
        print(f"Error fetching positions: {e}")


def parse_positions_response(response):
    """Parse Bankr positions response"""
    positions = []
    current_pos = {}
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # New position starts with pair name (e.g., "BTC/USD")
        if '/USD' in line and 'side' not in line.lower():
            if current_pos:
                positions.append(current_pos)
            current_pos = {'pair': line.replace('/USD', '').strip()}
        elif 'side:' in line.lower():
            current_pos['direction'] = 'long' if 'long' in line.lower() else 'short'
        elif 'collateral:' in line.lower():
            match = re.search(r'([\d.]+)', line)
            if match:
                current_pos['collateral'] = float(match.group(1))
        elif 'leverage:' in line.lower():
            match = re.search(r'([\d.]+)', line)
            if match:
                current_pos['leverage'] = float(match.group(1))
        elif 'entry price:' in line.lower():
            match = re.search(r'\$?([\d,]+\.?\d*)', line)
            if match:
                current_pos['entry_price'] = float(match.group(1).replace(',', ''))
        elif 'current price:' in line.lower():
            match = re.search(r'\$?([\d,]+\.?\d*)', line)
            if match:
                current_pos['current_price'] = float(match.group(1).replace(',', ''))
        elif 'pnl:' in line.lower():
            match = re.search(r'([+-]?[\d.]+)\s*USDC.*\(([+-]?[\d.]+)%\)', line)
            if match:
                current_pos['pnl'] = float(match.group(1))
                current_pos['pnl_pct'] = float(match.group(2))
    
    if current_pos:
        positions.append(current_pos)
    
    return positions


def positions_updater():
    """Background thread to update positions every 60 seconds"""
    cache_file = SWARM_DIR / 'balance_cache.json'
    while True:
        # Fetch positions
        fetch_positions_from_bankr()
        # Fetch USDC balance separately for accuracy
        usdc = fetch_usdc_balance()
        if usdc > 0:
            positions_cache['usdc_balance'] = usdc
            positions_cache['updated_at'] = datetime.now().isoformat()
            # Save to cache file for fast page loads
            try:
                cache_file.write_text(json.dumps(positions_cache, default=str))
            except:
                pass
            print(f"USDC balance: ${usdc:.2f}")
        time.sleep(60)


def run_server():
    global positions_cache
    
    # Load cached balance from file (for instant display on page load)
    cache_file = SWARM_DIR / 'balance_cache.json'
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            positions_cache.update(cached)
            print(f"Loaded cached balance: ${cached.get('usdc_balance', 0):.2f}")
        except:
            pass
    
    # Fetch balance immediately on startup
    print("Fetching initial balance from Bankr...")
    usdc = fetch_usdc_balance()
    if usdc > 0:
        positions_cache['usdc_balance'] = usdc
        positions_cache['updated_at'] = datetime.now().isoformat()
        # Save to cache file
        cache_file.write_text(json.dumps(positions_cache, default=str))
        print(f"Initial USDC balance: ${usdc:.2f}")
    
    # Start background positions updater
    updater_thread = threading.Thread(target=positions_updater, daemon=True)
    updater_thread.start()
    print("Started positions updater (60s interval)")
    
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"[DASHBOARD] Trading Swarm Dashboard")
        print(f"   http://localhost:{PORT}")
        print(f"   Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")


if __name__ == "__main__":
    run_server()
