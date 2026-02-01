#!/usr/bin/env python3
"""
TradingSwarmCore Status Dashboard
Reads agent logs and watchdog status to report current state.
"""

import json
from datetime import datetime
from pathlib import Path

SWARM_DIR = Path(__file__).parent
WATCHDOG_STATUS = SWARM_DIR / "watchdog_status.json"
AGENT_LOG = SWARM_DIR / "agent_log.txt"
WATCHDOG_LOG = SWARM_DIR / "watchdog_log.txt"


def get_watchdog_status() -> dict:
    if not WATCHDOG_STATUS.exists():
        return {"status": "not_running", "message": "Watchdog has never been started"}
    try:
        return json.loads(WATCHDOG_STATUS.read_text())
    except:
        return {"status": "unknown", "message": "Could not read watchdog status"}


def get_recent_trades(n: int = 10) -> list:
    """Parse recent trades from agent log"""
    trades = []
    if not AGENT_LOG.exists():
        return trades
    
    try:
        lines = AGENT_LOG.read_text().splitlines()[-500:]  # Last 500 lines
        for line in lines:
            if any(x in line.lower() for x in ['opened', 'closed', 'bought', 'sold']):
                trades.append(line)
        return trades[-n:]
    except:
        return []


def get_recent_errors(n: int = 5) -> list:
    """Parse recent errors from logs"""
    errors = []
    for log_file in [AGENT_LOG, WATCHDOG_LOG]:
        if not log_file.exists():
            continue
        try:
            lines = log_file.read_text().splitlines()[-200:]
            for line in lines:
                if 'error' in line.lower() or 'exception' in line.lower():
                    errors.append(line)
        except:
            pass
    return errors[-n:]


def format_uptime(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m {int(seconds%60)}s"
    elif seconds < 86400:
        return f"{int(seconds/3600)}h {int((seconds%3600)/60)}m"
    else:
        return f"{int(seconds/86400)}d {int((seconds%86400)/3600)}h"


def get_full_status() -> dict:
    """Get comprehensive status report"""
    watchdog = get_watchdog_status()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "watchdog": watchdog,
        "uptime": format_uptime(watchdog.get("uptime_seconds", 0)) if watchdog.get("uptime_seconds") else "N/A",
        "restarts_total": watchdog.get("total_restarts", 0),
        "restarts_last_hour": watchdog.get("restarts_last_hour", 0),
        "pid": watchdog.get("pid"),
        "recent_trades": get_recent_trades(10),
        "recent_errors": get_recent_errors(5),
        "agent_log_exists": AGENT_LOG.exists(),
        "agent_log_size_kb": round(AGENT_LOG.stat().st_size / 1024, 1) if AGENT_LOG.exists() else 0
    }


def print_status():
    """Print formatted status to console"""
    status = get_full_status()
    
    print("\n" + "="*50)
    print("🎲 TRADING SWARM CORE STATUS")
    print("="*50)
    print(f"Status:     {status['watchdog'].get('status', 'unknown').upper()}")
    print(f"Uptime:     {status['uptime']}")
    print(f"PID:        {status['pid'] or 'N/A'}")
    print(f"Restarts:   {status['restarts_total']} total, {status['restarts_last_hour']} last hour")
    print(f"Log size:   {status['agent_log_size_kb']} KB")
    
    if status['recent_trades']:
        print("\n📈 Recent Trades:")
        for trade in status['recent_trades'][-5:]:
            print(f"  {trade[:80]}...")
    else:
        print("\n📈 No trades logged yet")
    
    if status['recent_errors']:
        print("\n⚠️ Recent Errors:")
        for err in status['recent_errors']:
            print(f"  {err[:80]}...")
    else:
        print("\n✅ No recent errors")
    
    print("="*50 + "\n")
    
    return status


if __name__ == "__main__":
    print_status()
