#!/usr/bin/env python3
"""
Risk Manager Agent
Monitors portfolio risk, enforces limits, and can emergency-close positions.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RISK - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "risk_log.txt"),
        logging.StreamHandler()
    ]
)

SWARM_DIR = Path(__file__).parent.parent
CONFIG_FILE = SWARM_DIR / "config.json"
RISK_STATE_FILE = SWARM_DIR / "risk_state.json"
ALERTS_FILE = SWARM_DIR / "alerts.json"

class RiskManager:
    def __init__(self):
        self.config = self.load_config()
        self.state = self.load_state()
        self.alerts = deque(maxlen=100)
        self.daily_pnl = 0
        self.daily_start_balance = self.state.get('balance', 30.0)
        self.last_daily_reset = datetime.now().date()
        
    def load_config(self) -> dict:
        try:
            if CONFIG_FILE.exists():
                return json.loads(CONFIG_FILE.read_text())
        except:
            pass
        return {
            'risk': {
                'max_position_pct': 0.25,
                'max_daily_drawdown_pct': 0.15,
                'max_consecutive_losses': 4
            }
        }
    
    def load_state(self) -> dict:
        try:
            if RISK_STATE_FILE.exists():
                return json.loads(RISK_STATE_FILE.read_text())
        except:
            pass
        return {
            'balance': 30.0,
            'positions': [],
            'consecutive_losses': 0,
            'daily_trades': 0,
            'daily_pnl': 0,
            'total_pnl': 0,
            'paused': False,
            'pause_reason': None
        }
    
    def save_state(self):
        self.state['updated_at'] = datetime.now().isoformat()
        RISK_STATE_FILE.write_text(json.dumps(self.state, indent=2))
    
    def add_alert(self, level: str, message: str):
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.alerts.append(alert)
        logging.warning(f"ALERT [{level}]: {message}")
        
        # Save alerts
        try:
            alerts_list = list(self.alerts)
            ALERTS_FILE.write_text(json.dumps(alerts_list, indent=2))
        except:
            pass
    
    def check_daily_reset(self):
        """Reset daily counters at midnight"""
        today = datetime.now().date()
        if today > self.last_daily_reset:
            logging.info("Daily reset triggered")
            self.daily_pnl = 0
            self.state['daily_trades'] = 0
            self.state['daily_pnl'] = 0
            self.daily_start_balance = self.state.get('balance', 30.0)
            self.last_daily_reset = today
            
            # Clear pause if it was due to daily limits
            if self.state.get('pause_reason') == 'daily_drawdown':
                self.state['paused'] = False
                self.state['pause_reason'] = None
                logging.info("Daily pause cleared")
    
    def update_balance(self, new_balance: float, trade_pnl: float = 0):
        """Update balance and check risk limits"""
        old_balance = self.state.get('balance', 30.0)
        self.state['balance'] = new_balance
        
        if trade_pnl != 0:
            self.state['daily_pnl'] = self.state.get('daily_pnl', 0) + trade_pnl
            self.state['total_pnl'] = self.state.get('total_pnl', 0) + trade_pnl
            self.state['daily_trades'] = self.state.get('daily_trades', 0) + 1
            
            if trade_pnl < 0:
                self.state['consecutive_losses'] = self.state.get('consecutive_losses', 0) + 1
            else:
                self.state['consecutive_losses'] = 0
        
        self.check_risk_limits()
        self.save_state()
    
    def check_risk_limits(self) -> bool:
        """Check all risk limits, return True if trading should pause"""
        risk_config = self.config.get('risk', {})
        
        # Check daily drawdown
        daily_drawdown = (self.daily_start_balance - self.state['balance']) / self.daily_start_balance
        max_drawdown = risk_config.get('max_daily_drawdown_pct', 0.15)
        
        if daily_drawdown >= max_drawdown:
            self.state['paused'] = True
            self.state['pause_reason'] = 'daily_drawdown'
            self.add_alert('CRITICAL', f'Daily drawdown limit hit ({daily_drawdown:.1%}). Trading paused.')
            return True
        
        # Check consecutive losses
        max_losses = risk_config.get('max_consecutive_losses', 4)
        if self.state.get('consecutive_losses', 0) >= max_losses:
            self.state['paused'] = True
            self.state['pause_reason'] = 'consecutive_losses'
            self.add_alert('WARNING', f'Max consecutive losses ({max_losses}). Trading paused for cooldown.')
            return True
        
        # Warning at 50% of limits
        if daily_drawdown >= max_drawdown * 0.5:
            self.add_alert('WARNING', f'Daily drawdown at {daily_drawdown:.1%} (limit: {max_drawdown:.1%})')
        
        if self.state.get('consecutive_losses', 0) >= max_losses - 1:
            self.add_alert('WARNING', f'Consecutive losses: {self.state["consecutive_losses"]} (max: {max_losses})')
        
        return False
    
    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed"""
        self.check_daily_reset()
        
        if self.state.get('paused'):
            return False, self.state.get('pause_reason', 'unknown')
        
        if self.state.get('balance', 0) < 1.0:
            return False, 'insufficient_balance'
        
        return True, 'ok'
    
    def get_position_size(self, base_size: float) -> float:
        """Adjust position size based on risk state"""
        risk_config = self.config.get('risk', {})
        
        # Reduce size after consecutive losses
        losses = self.state.get('consecutive_losses', 0)
        if losses > 0:
            reducer = risk_config.get('loss_streak_reducer', 0.7)
            base_size *= (reducer ** losses)
        
        # Cap at max position
        max_pct = risk_config.get('max_position_pct', 0.25)
        max_size = self.state.get('balance', 30.0) * max_pct
        
        return min(base_size, max_size)
    
    def get_status(self) -> dict:
        """Get current risk status"""
        return {
            'balance': self.state.get('balance', 30.0),
            'daily_pnl': self.state.get('daily_pnl', 0),
            'daily_pnl_pct': self.state.get('daily_pnl', 0) / max(self.daily_start_balance, 1),
            'total_pnl': self.state.get('total_pnl', 0),
            'consecutive_losses': self.state.get('consecutive_losses', 0),
            'daily_trades': self.state.get('daily_trades', 0),
            'paused': self.state.get('paused', False),
            'pause_reason': self.state.get('pause_reason'),
            'can_trade': self.can_trade()[0]
        }
    
    def run(self, interval_seconds: int = 60):
        """Run continuous risk monitoring"""
        logging.info("Risk Manager starting...")
        
        while True:
            try:
                self.check_daily_reset()
                status = self.get_status()
                
                logging.info(
                    f"Balance: ${status['balance']:.2f} | "
                    f"Daily P&L: ${status['daily_pnl']:.2f} ({status['daily_pnl_pct']:.1%}) | "
                    f"Trades: {status['daily_trades']} | "
                    f"Can Trade: {status['can_trade']}"
                )
                
            except Exception as e:
                logging.error(f"Risk monitor error: {e}")
            
            time.sleep(interval_seconds)


if __name__ == "__main__":
    manager = RiskManager()
    manager.run(interval_seconds=60)
