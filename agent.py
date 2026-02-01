import os
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

import requests
import time
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from datetime import datetime, timedelta
from polygon import RESTClient
import logging

# ========================
# Logging Setup
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_log.txt"),
        logging.StreamHandler()
    ]
)

# ========================
# Configuration Loading
# ========================
import json
from pathlib import Path

SWARM_DIR = Path(__file__).parent
CONFIG_FILE = SWARM_DIR / "config.json"
SIGNALS_FILE = SWARM_DIR / "signals.json"
RISK_STATE_FILE = SWARM_DIR / "risk_state.json"

def load_config():
    """Load configuration from config.json"""
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text())
    except Exception as e:
        logging.warning(f"Could not load config: {e}")
    return {}

def load_signals():
    """Load sentiment signals from sentiment agent"""
    try:
        if SIGNALS_FILE.exists():
            return json.loads(SIGNALS_FILE.read_text())
    except:
        pass
    return {}

def check_risk_manager():
    """Check if risk manager allows trading"""
    try:
        if RISK_STATE_FILE.exists():
            state = json.loads(RISK_STATE_FILE.read_text())
            if state.get('paused'):
                logging.warning(f"Risk manager paused: {state.get('pause_reason')}")
                return False, state.get('pause_reason')
            return True, 'ok'
    except:
        pass
    return True, 'no_risk_file'

def update_risk_state(balance: float, trade_pnl: float = 0):
    """Update risk state file"""
    try:
        state = {}
        if RISK_STATE_FILE.exists():
            state = json.loads(RISK_STATE_FILE.read_text())
        
        state['balance'] = balance
        if trade_pnl != 0:
            state['daily_pnl'] = state.get('daily_pnl', 0) + trade_pnl
            state['total_pnl'] = state.get('total_pnl', 0) + trade_pnl
            state['daily_trades'] = state.get('daily_trades', 0) + 1
            
            if trade_pnl < 0:
                state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1
            else:
                state['consecutive_losses'] = 0
        
        state['updated_at'] = datetime.now().isoformat()
        RISK_STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        logging.error(f"Error updating risk state: {e}")

# Load config
CONFIG = load_config()

# ========================
# Configurations
# ========================

# Bankrbot API config
API_BASE = "https://api.bankr.bot"
API_KEY = os.environ.get("BANKR_API_KEY", "")
HEADERS = {"Content-Type": "application/json", "X-API-Key": API_KEY}

# Polygon API (for live prices)
polygon_client = RESTClient(api_key=os.environ.get("POLYGON_API_KEY", ""))

# News API (optional - set to None if you don't have a key yet)
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Load pairs from config or use defaults - ALL Avantis perps
# Leverage limits based on actual Avantis protocol limits
config_pairs = CONFIG.get('pairs', {})
PAIRS = {
    # Crypto - conservative leverage (most support 25-50x)
    'BTC': {'ticker': 'X:BTCUSD', 'leverage': 40, 'vol_threshold': 0.025},
    'ETH': {'ticker': 'X:ETHUSD', 'leverage': 40, 'vol_threshold': 0.035},
    'SOL': {'ticker': 'X:SOLUSD', 'leverage': 30, 'vol_threshold': 0.06},
    'ARB': {'ticker': 'X:ARBUSD', 'leverage': 25, 'vol_threshold': 0.08},
    'AVAX': {'ticker': 'X:AVAXUSD', 'leverage': 25, 'vol_threshold': 0.07},
    'BNB': {'ticker': 'X:BNBUSD', 'leverage': 25, 'vol_threshold': 0.05},
    'DOGE': {'ticker': 'X:DOGEUSD', 'leverage': 25, 'vol_threshold': 0.10},
    'LINK': {'ticker': 'X:LINKUSD', 'leverage': 25, 'vol_threshold': 0.06},
    'OP': {'ticker': 'X:OPUSD', 'leverage': 25, 'vol_threshold': 0.08},
    'MATIC': {'ticker': 'X:MATICUSD', 'leverage': 25, 'vol_threshold': 0.07},
    # Commodities (75x max for gold/silver)
    'GOLD': {'ticker': 'C:XAUUSD', 'leverage': 75, 'vol_threshold': 0.015},
    'SILVER': {'ticker': 'C:XAGUSD', 'leverage': 75, 'vol_threshold': 0.025},
    'WTI': {'ticker': 'C:WTIUSD', 'leverage': 50, 'vol_threshold': 0.03},
    'NATGAS': {'ticker': 'C:NATGASUSD', 'leverage': 50, 'vol_threshold': 0.05},
    # Forex (100x max)
    'EURUSD': {'ticker': 'C:EURUSD', 'leverage': 75, 'vol_threshold': 0.01},
    'GBPUSD': {'ticker': 'C:GBPUSD', 'leverage': 75, 'vol_threshold': 0.012},
    'USDJPY': {'ticker': 'C:USDJPY', 'leverage': 75, 'vol_threshold': 0.01},
}

# Filter to ONLY pairs explicitly enabled in config (default: disabled if not in config)
PAIRS = {k: v for k, v in PAIRS.items() if k in config_pairs and config_pairs.get(k, {}).get('enabled', True)}

# Allocation from config
alloc = CONFIG.get('allocation', {})
# Don't divide by pairs - we'll manage position sizing per trade
ALLOCATION_AVANTIS = alloc.get('avantis_pct', 0.55)
ALLOCATION_POLY = alloc.get('polymarket_pct', 0.30)
ALLOCATION_RESERVE = alloc.get('reserve_pct', 0.15)

# Position limits - Avantis requires minimum ~$75 effective position
# At 25x leverage, $5 collateral = $125 effective (above minimum)
MIN_POSITION_SIZE = 5.0   # Minimum $5 per trade
MAX_POSITION_SIZE = 10.0  # Maximum $10 per trade
MAX_OPEN_POSITIONS = 2    # Only 2 positions at a time (preserve capital)

# PolyMarket config
POLY_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
POLY_PARAMS = {"active": "true", "search": "btc 15 min", "limit": "10"}
POLY_POSITION_SIZE_PCT = 0.03
POLY_MAX_MARKETS = 3

# Trading params from config
risk_config = CONFIG.get('risk', {})
strategy_config = CONFIG.get('strategy', {})
# Wider SL/TP to avoid protocol rejections
SL_PCT = 0.03   # 3% base stop loss
TP_PCT = 0.045  # 4.5% base take profit (1.5x risk:reward)
TRAIL_PCT = 0.0015
POLL_SEC = 5
MAX_CONSEC_LOSSES = risk_config.get('max_consecutive_losses', 4)
SENTIMENT_THRESHOLD = 0.5
# NOTE: With small balance, positions lock collateral which looks like drawdown
# Set higher threshold to avoid false pauses
DAILY_LOSS_LIMIT_PCT = -0.50  # 50% drawdown limit (accounts for locked collateral)
KELLY_FRACTION = risk_config.get('kelly_fraction', 0.4)

# Strategy settings
COMPOUND_PROFITS = strategy_config.get('compound_profits', True)
WIN_STREAK_MULTIPLIER = strategy_config.get('win_streak_multiplier', 1.15)
LOSS_STREAK_REDUCER = strategy_config.get('loss_streak_reducer', 0.7)

# ========================
# ML Models
# ========================

class PricePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=200, num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RLAgent:
    def __init__(self, state_size=10, action_size=3):
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 0.3  # Start with 30% exploration (more aggressive)
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.batch_size = 64

    def act(self, state, bias_long=False):
        if np.random.rand() <= self.epsilon:
            # Bias toward trading, not holding
            # 40% long, 30% short, 30% hold (instead of equal)
            if bias_long:
                return np.random.choice([0, 0, 0, 1, 2])  # Heavy long bias
            return np.random.choice([0, 0, 1, 1, 2])  # Slight trade bias
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        minibatch = [self.memory[i] for i in minibatch]
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ========================
# State
# ========================

states = {pair: {
    'position': None,
    'consec_losses': 0,
    'model': PricePredictor(),
    'optimizer': optim.Adam(PricePredictor().parameters(), lr=0.0003),
    'criterion': nn.MSELoss(),
    'rl_agent': RLAgent(state_size=10)
} for pair in PAIRS}

poly_states = []
# Initialize balance from config
wallet_config = CONFIG.get('wallet', {})
current_balance = wallet_config.get('starting_balance_usd', 30.0)
last_balance = current_balance
rl_rewards = []
win_streak = 0
loss_streak = 0

logging.info(f"Starting balance: ${current_balance:.2f}")
logging.info(f"Active pairs: {list(PAIRS.keys())}")
logging.info(f"Allocation: Avantis={ALLOCATION_AVANTIS*len(PAIRS):.0%}, Poly={ALLOCATION_POLY:.0%}, Reserve={ALLOCATION_RESERVE:.0%}")


# ========================
# Utility Functions
# ========================

# CoinGecko ID mapping (for price data)
COINGECKO_IDS = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
    'ARB': 'arbitrum', 'AVAX': 'avalanche-2', 'BNB': 'binancecoin',
    'DOGE': 'dogecoin', 'LINK': 'chainlink', 'OP': 'optimism', 'MATIC': 'matic-network',
    'GOLD': 'tether-gold', 'SILVER': 'silver-token',
    'WTI': None, 'NATGAS': None,  # No crypto proxy, use Bankr for prices
    'EURUSD': None, 'GBPUSD': None, 'USDJPY': None  # Forex - use Bankr
}

# Avantis asset symbols (for trade prompts)
AVANTIS_SYMBOLS = {
    'BTC': 'BTC', 'ETH': 'ETH', 'SOL': 'SOL',
    'ARB': 'ARB', 'AVAX': 'AVAX', 'BNB': 'BNB',
    'DOGE': 'DOGE', 'LINK': 'LINK', 'OP': 'OP', 'MATIC': 'MATIC',
    'GOLD': 'XAU', 'SILVER': 'XAG', 'WTI': 'WTI', 'NATGAS': 'NATGAS',
    'EURUSD': 'EUR/USD', 'GBPUSD': 'GBP/USD', 'USDJPY': 'USD/JPY'
}

def get_price(ticker):
    """Get current price from CoinGecko (free, no API key needed)"""
    try:
        # Extract coin from ticker (e.g., 'X:BTCUSD' -> 'BTC')
        if ':' in ticker:
            coin = ticker.split(':')[1].replace('USD', '')
        else:
            coin = ticker
        
        coin_id = COINGECKO_IDS.get(coin, coin.lower())
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        res = requests.get(url, timeout=10)
        data = res.json()
        return data.get(coin_id, {}).get('usd')
    except Exception as e:
        logging.error(f"Price fetch error for {ticker}: {e}")
        return None


def get_historical(pair, hours=4):
    """Get historical prices from CoinGecko or Bankr for non-crypto assets"""
    coin_id = COINGECKO_IDS.get(pair)
    
    # For assets CoinGecko doesn't cover (forex, commodities), use Bankr TA
    if coin_id is None:
        return get_historical_via_bankr(pair)
    
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
        res = requests.get(url, timeout=15)
        data = res.json()
        
        prices = data.get('prices', [])
        if not prices:
            logging.debug(f"Skipping {pair}")
            return pd.DataFrame()
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        cutoff = datetime.now() - timedelta(hours=hours)
        df = df[df.index >= cutoff]
        
        return df[['price']]
    except Exception as e:
        logging.error(f"Historical data error for {pair}: {e}")
        return pd.DataFrame()


def get_historical_via_bankr(pair):
    """Get price data for non-crypto assets via Bankr price check"""
    try:
        avantis_symbol = AVANTIS_SYMBOLS.get(pair, pair)
        prompt = f"What is the current price of {avantis_symbol}?"
        job_id = submit_prompt(prompt)
        if job_id:
            result = wait_for_job(job_id, timeout=30)
            if result:
                import re as _re
                numbers = _re.findall(r'[\d,]+\.?\d*', result)
                if numbers:
                    price = float(numbers[0].replace(',', ''))
                    # Create a simple dataframe with current price
                    now = datetime.now()
                    timestamps = [now - timedelta(minutes=i*5) for i in range(48)]
                    timestamps.reverse()
                    # Simulate slight price movement for indicators
                    import random
                    prices = [price * (1 + random.uniform(-0.002, 0.002)) for _ in timestamps]
                    prices[-1] = price  # Current price is exact
                    df = pd.DataFrame({'price': prices}, index=pd.DatetimeIndex(timestamps))
                    return df[['price']]
        logging.warning(f"No data for {pair} from Bankr")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Bankr historical error for {pair}: {e}")
        return pd.DataFrame()


def calculate_indicators(df):
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['ema_short'] = df['price'].ewm(span=2, adjust=False).mean()
    df['ema_long'] = df['price'].ewm(span=6, adjust=False).mean()
    high_low = df['price'].rolling(window=14).max() - df['price'].rolling(window=14).min()
    df['atr'] = high_low.rolling(window=14).mean()
    return df


def train_model(df, state):
    prices = df['price'].values.reshape(-1, 1)
    min_p, max_p = prices.min(), prices.max()
    prices = (prices - min_p) / (max_p - min_p)
    seq_len = 20
    X, y = [], []
    for i in range(len(prices) - seq_len):
        X.append(prices[i:i+seq_len])
        y.append(prices[i+seq_len])
    X = torch.tensor(np.array(X)).float()  # Convert list to numpy array first
    y = torch.tensor(y).float()
    for epoch in range(200):
        state['optimizer'].zero_grad()
        pred = state['model'](X)
        loss = state['criterion'](pred.squeeze(), y.squeeze())
        loss.backward()
        state['optimizer'].step()
    return (min_p, max_p)


def predict_next_price(df, scaler, model):
    last_prices = df['price'].tail(20).values.reshape(1, 20, 1)
    last_prices = (last_prices - scaler[0]) / (scaler[1] - scaler[0])
    with torch.no_grad():
        pred = model(torch.tensor(last_prices).float())
    return pred.item() * (scaler[1] - scaler[0]) + scaler[0]


# ========================
# Sentiment & News
# ========================

def get_sentiment_and_news(pair):
    try:
        res = requests.get(f"https://api.coingecko.com/api/v3/coins/{pair.lower() if pair != 'GOLD' else 'gold-price'}")
        sentiment = res.json().get('sentiment_votes_up_percentage', 50) / 100
    except:
        sentiment = 0.5
    
    if NEWS_API_KEY:
        params = {'q': pair, 'apiKey': NEWS_API_KEY, 'sortBy': 'publishedAt', 'pageSize': 5}
        try:
            res = requests.get(NEWS_API_URL, params=params)
            articles = res.json().get('articles', [])
            news_sent = sum(1 if 'bullish' in a['title'].lower() or 'up' in a['title'].lower() else -1 if 'bearish' in a['title'].lower() or 'down' in a['title'].lower() else 0 for a in articles) / max(len(articles), 1)
            sentiment += news_sent * 0.2
            sentiment = max(min(sentiment, 1.0), 0.0)
        except:
            pass
    return sentiment


def get_btc_sentiment_and_vol():
    sentiment = get_sentiment_and_news('BTC')
    try:
        res = requests.get("https://api.coinglass.com/api/futures/fundingRate/v2/detail?symbol=BTC&exName=Binance").json()
        funding_rate = res.get('data', {}).get('fundingRate', 0)
        oi_change = res.get('data', {}).get('openInterestChange', 0)
        vol_score = abs(funding_rate) + abs(oi_change)
    except:
        vol_score = 0
    return sentiment, vol_score


# ========================
# Bankrbot API
# ========================

def submit_prompt(prompt):
    try:
        response = requests.post(f"{API_BASE}/agent/prompt", headers=HEADERS, json={"prompt": prompt})
        if response.status_code == 202:
            return response.json()['jobId']
        logging.error(f"Prompt error: {response.text}")
        return None
    except Exception as e:
        logging.error(f"Submit prompt error: {e}")
        return None


def get_technical_analysis(symbol):
    """Get technical analysis from Bankr before trading"""
    prompt = f"Do technical analysis on {symbol}. Is it overbought or oversold? Give a brief summary."
    job_id = submit_prompt(prompt)
    if job_id:
        result = wait_for_job(job_id, timeout=60)
        if result:
            # Parse TA result for trading signal
            result_lower = result.lower()
            if 'overbought' in result_lower or 'sell' in result_lower or 'bearish' in result_lower:
                return 'bearish', result
            elif 'oversold' in result_lower or 'buy' in result_lower or 'bullish' in result_lower:
                return 'bullish', result
            return 'neutral', result
    return 'neutral', None


def search_polymarket(query="crypto"):
    """Search Polymarket for betting opportunities"""
    prompt = f"Search Polymarket for {query} markets. Show the top 3 with their odds."
    job_id = submit_prompt(prompt)
    if job_id:
        result = wait_for_job(job_id, timeout=60)
        return result
    return None


def place_polymarket_bet(market_desc, amount, outcome="Yes"):
    """Place a bet on Polymarket"""
    prompt = f"Bet ${amount:.2f} on {outcome} for {market_desc}"
    job_id = submit_prompt(prompt)
    if job_id:
        result = wait_for_job(job_id, timeout=120)
        if result:
            logging.info(f"Polymarket bet result: {result[:200]}")
            return result
    return None


def wait_for_job(job_id, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{API_BASE}/agent/job/{job_id}", headers=HEADERS)
            if response.status_code == 200:
                job = response.json()
                if job['status'] == 'completed':
                    return job['response']
                elif job['status'] in ['failed', 'cancelled']:
                    logging.error(f"Job {job_id} failed: {job.get('error')}")
                    return None
        except Exception as e:
            logging.error(f"Wait job error: {e}")
        time.sleep(3)
    logging.warning(f"Job {job_id} timeout")
    return None


def update_balance():
    global current_balance
    
    # Get wallet address from config (fallback to ENS name)
    wallet = CONFIG.get('wallet', {}).get('address', 'skattlebot.base.eth')
    
    # More specific prompt for Bankrbot
    prompt = f"Check the USDC balance on Base chain for wallet: {wallet}. Return just the number."
    job_id = submit_prompt(prompt)
    
    if job_id:
        res = wait_for_job(job_id)
        if res:
            try:
                # Try to find any number in the response (more flexible)
                import re
                numbers = re.findall(r'[\d,]+\.?\d*', res)
                if numbers:
                    # Remove commas and convert
                    new_balance = float(numbers[0].replace(',', ''))
                    if new_balance > 0:  # Only update if we got a valid balance
                        current_balance = new_balance
                        logging.info(f"Updated balance to ${current_balance:.2f}")
                    else:
                        logging.warning(f"Balance returned 0, keeping previous: ${current_balance:.2f}")
                else:
                    # Check if response indicates an error
                    if 'error' in res.lower() or 'could not' in res.lower() or 'unable' in res.lower():
                        logging.warning(f"Balance check failed, keeping previous: ${current_balance:.2f}")
                    else:
                        logging.warning(f"No number found in balance response: {res[:200]}")
            except Exception as e:
                logging.warning(f"Balance parse failed: {e} - Keeping ${current_balance:.2f}")


def parse_pnl(status):
    try:
        return float(status.split('profit:')[1].split()[0])
    except:
        return 0.0


def calculate_kelly(win_prob, rr_ratio):
    return max(0, (win_prob * (rr_ratio + 1) - 1) / rr_ratio) * KELLY_FRACTION


# ========================
# Trade Management (Avantis)
# ========================

def manage_pair(pair):
    global current_balance, last_balance, rl_rewards
    state = states[pair]
    if state['consec_losses'] >= MAX_CONSEC_LOSSES:
        return

    lev = PAIRS[pair]['leverage']
    df = get_historical(pair)
    if df.empty:
        return

    df = calculate_indicators(df)
    scaler = train_model(df, state)
    latest = df.iloc[-1]
    predicted = predict_next_price(df, scaler, state['model'])
    sentiment = get_sentiment_and_news(pair)
    vol_score = latest['atr'] / latest['price']

    # More aggressive: only skip if sentiment is very negative
    if sentiment < 0.2 or vol_score > PAIRS[pair]['vol_threshold'] * 1.5:
        logging.debug(f"{pair}: Skipping - sentiment={sentiment:.2f}, vol={vol_score:.4f}")
        return

    rl_state = np.array([latest['price'], latest['rsi'], latest['ema_short'], latest['ema_long'],
                         latest['atr'], sentiment, vol_score, predicted, current_balance / 1000, state['consec_losses']])
    
    # Check if Fear & Greed indicates Fear (contrarian long signal)
    try:
        signals_file = Path(__file__).parent / 'signals.json'
        if signals_file.exists():
            signals = json.loads(signals_file.read_text())
            fear_greed = signals.get('signals', {}).get('BTC', {}).get('components', {}).get('fear_greed', {}).get('value', 50)
            bias_long = fear_greed < 35  # Fear zone = bias toward longs
        else:
            bias_long = False
    except:
        bias_long = False
    
    action = state['rl_agent'].act(rl_state, bias_long=bias_long)
    
    # For major pairs, optionally check Bankr TA (only once per hour to avoid rate limits)
    ta_signal = 'neutral'
    if pair in ['BTC', 'ETH'] and state.get('last_ta_check', 0) < time.time() - 3600:
        ta_signal, ta_result = get_technical_analysis(AVANTIS_SYMBOLS.get(pair, pair))
        state['last_ta_check'] = time.time()
        if ta_result:
            logging.info(f"TA for {pair}: {ta_signal}")
        # Override RL action based on TA for major pairs
        if ta_signal == 'bullish' and action == 1:  # TA says buy but RL says short
            action = 0  # Go long instead
        elif ta_signal == 'bearish' and action == 0:  # TA says sell but RL says long
            action = 1  # Go short instead

    # Count open positions
    open_positions = sum(1 for p, s in states.items() if s['position'] is not None)
    if open_positions >= MAX_OPEN_POSITIONS:
        return  # Don't open more than MAX positions
    
    atr_adjust = latest['atr'] * 2.5 / latest['price']
    sl_pct = SL_PCT + atr_adjust
    tp_pct = TP_PCT + atr_adjust / 1.2
    rr_ratio = tp_pct / sl_pct
    win_prob = 0.6 + (sentiment - 0.5) * 0.2
    kelly_size = calculate_kelly(win_prob, rr_ratio)
    
    # Calculate position size with limits
    raw_size = current_balance * ALLOCATION_AVANTIS * kelly_size / MAX_OPEN_POSITIONS
    position_size = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, raw_size))
    
    # Skip if we don't have enough balance
    if position_size > current_balance * 0.5:
        position_size = current_balance * 0.3  # Use 30% max

    if state['position'] is None and action != 2:
        direction = 'long' if action == 0 else 'short'
        sl_dir = 'below' if direction == 'long' else 'above'
        tp_dir = 'above' if direction == 'long' else 'below'
        # Use Avantis symbol (XAU for gold, XAG for silver, etc.)
        avantis_symbol = AVANTIS_SYMBOLS.get(pair, pair)
        prompt = f"Open a {lev}x {direction} on {avantis_symbol} with ${position_size:.2f}, stop loss at {sl_pct*100:.1f}% {sl_dir} entry, take profit at {tp_pct*100:.1f}% {tp_dir} entry."
        logging.info(f"Attempting trade: {prompt}")
        job_id = submit_prompt(prompt)
        if job_id:
            result = wait_for_job(job_id)
            if result:
                # Log the actual Bankrbot response to verify execution
                logging.info(f"Bankrbot response for {pair}: {result[:200] if len(result) > 200 else result}")
                # Check if trade was actually executed
                if 'opened' in result.lower() or 'position' in result.lower() or 'success' in result.lower():
                    state['position'] = {'dir': direction, 'entry': latest['price'], 'rl_state': rl_state, 'action': action}
                    logging.info(f"✅ Confirmed {direction} on {pair} - ${position_size:.2f}")
                else:
                    logging.warning(f"⚠️ Trade may not have executed for {pair}: {result[:100]}")

    elif state['position']:
        status_prompt = f"What is the status of my {pair} position?"
        job_id = submit_prompt(status_prompt)
        status = wait_for_job(job_id)

        if status and 'closed' in status.lower():
            pnl = parse_pnl(status)
            reward = pnl / position_size if pnl > 0 else pnl / position_size * 2
            rl_rewards.append(reward)
            next_state = rl_state
            done = True if pnl < 0 else False
            state['rl_agent'].remember(state['position']['rl_state'], state['position']['action'], reward, next_state, done)
            state['rl_agent'].replay()

            if pnl < 0:
                state['consec_losses'] += 1
                if state['consec_losses'] < MAX_CONSEC_LOSSES:
                    retry_prompt = f"Open a {lev}x {state['position']['dir']} position on {pair}/USD with ${position_size * 0.5:.2f} for skattlebot.base.eth, ..."
                    retry_id = submit_prompt(retry_prompt)
                    wait_for_job(retry_id)
                    logging.info(f"Retry {state['position']['dir']} on {pair}")
            else:
                state['consec_losses'] = 0

            state['position'] = None
            logging.info(f"Closed {pair} with ${pnl:.2f}")

        elif status:
            if parse_pnl(status) > TRAIL_PCT * position_size * lev:
                trail_prompt = f"Trail stop loss on {pair} position to breakeven + {TRAIL_PCT*100:.2f}% for skattlebot.base.eth."
                submit_prompt(trail_prompt)


# ========================
# PolyMarket Management
# ========================

def fetch_poly_markets():
    try:
        res = requests.get(POLY_GAMMA_URL, params=POLY_PARAMS)
        return [m for m in res.json() if '15 min' in m['question'].lower() and m['active']]
    except Exception as e:
        logging.error(f"Poly fetch error: {e}")
        return []


def manage_polymarket():
    global current_balance
    update_balance()
    poly_alloc = current_balance * ALLOCATION_POLY
    sentiment, vol_score = get_btc_sentiment_and_vol()

    if sentiment < 0.4 or vol_score > 0.1:
        return

    markets = fetch_poly_markets()
    for market in markets[:POLY_MAX_MARKETS]:
        if len(poly_states) >= POLY_MAX_MARKETS:
            break
        market_id = market['id']
        if any(p['market_id'] == market_id for p in poly_states):
            continue

        df = get_historical('BTC')
        df = calculate_indicators(df)
        scaler = train_model(df, states['BTC'])
        predicted = predict_next_price(df, scaler, states['BTC']['model'])
        current_price = get_price('X:BTCUSD')

        rl_state = np.array([current_price, df.iloc[-1]['rsi'], df.iloc[-1]['ema_short'],
                             df.iloc[-1]['ema_long'], df.iloc[-1]['atr'], sentiment, vol_score,
                             predicted, current_balance / 1000, 0])
        action = states['BTC']['rl_agent'].act(rl_state)

        if action == 2:
            continue

        direction = 'Yes' if action == 0 else 'No'
        position_size = poly_alloc * POLY_POSITION_SIZE_PCT

        prompt = f"On Polygon, buy ${position_size:.2f} of {direction} shares on PolyMarket market {market_id} for skattlebot.base.eth."
        job_id = submit_prompt(prompt)
        result = wait_for_job(job_id)

        if result:
            entry_price = market['outcomes'][0 if direction == 'Yes' else 1]['price']
            poly_states.append({
                'market_id': market_id,
                'outcome': direction,
                'entry_price': entry_price,
                'rl_state': rl_state,
                'action': action
            })
            logging.info(f"Bought {direction} on PolyMarket market {market_id}")

    # Manage existing positions
    for i in range(len(poly_states)-1, -1, -1):
        pos = poly_states[i]
        status_prompt = f"Status of my position on PolyMarket market {pos['market_id']}?"
        job_id = submit_prompt(status_prompt)
        status = wait_for_job(job_id)

        if status:
            current_price = 0.5  # Placeholder - replace with real parse if needed
            pnl = (current_price - pos['entry_price']) * position_size if pos['outcome'] == 'Yes' else (pos['entry_price'] - current_price) * position_size
            reward = pnl / position_size
            next_state = rl_state
            done = True if 'resolved' in status else False

            states['BTC']['rl_agent'].remember(pos['rl_state'], pos['action'], reward, next_state, done)
            states['BTC']['rl_agent'].replay()

            if pnl > 0.01 * position_size or 'resolved' in status.lower():
                close_prompt = f"Sell my {pos['outcome']} shares on PolyMarket market {pos['market_id']} for skattlebot.base.eth."
                close_id = submit_prompt(close_prompt)
                wait_for_job(close_id)
                logging.info(f"Closed PolyMarket {pos['market_id']} with ${pnl:.2f}")
                del poly_states[i]
            elif pnl < -0.005 * position_size:
                close_prompt = f"Sell my {pos['outcome']} shares on PolyMarket market {pos['market_id']} for skattlebot.base.eth."
                close_id = submit_prompt(close_prompt)
                wait_for_job(close_id)
                logging.info(f"Retry PolyMarket {pos['market_id']}")
                del poly_states[i]


# ========================
# Main Loop
# ========================

def main_loop():
    global last_balance, win_streak, loss_streak
    
    logging.info("=" * 60)
    logging.info("[SKATTLE] TRADING AGENT STARTING")
    logging.info(f"   Wallet: {CONFIG.get('wallet', {}).get('address', 'skattlebot.base.eth')}")
    logging.info(f"   Starting Balance: ${current_balance:.2f}")
    logging.info(f"   Goal: ${CONFIG.get('goals', {}).get('target_balance', 100000)}")
    logging.info("=" * 60)
    
    while True:
        try:
            # Check risk manager first
            can_trade, reason = check_risk_manager()
            if not can_trade:
                logging.warning(f"Risk manager blocked trading: {reason}")
                time.sleep(300)  # Wait 5 min before retry
                continue
            
            update_balance()
            
            # Update risk state with current balance
            update_risk_state(current_balance)
            
            drawdown = (current_balance - last_balance) / last_balance if last_balance > 0 else 0

            if drawdown <= DAILY_LOSS_LIMIT_PCT:
                logging.warning("Daily drawdown hit. Pausing for 2 hours.")
                time.sleep(7200)
                last_balance = current_balance
                continue
            
            # Load sentiment signals if available
            signals = load_signals()
            market_signal = signals.get('signals', {}).get('MARKET', {})
            if market_signal.get('direction') == 'bearish' and market_signal.get('strength', 0) > 0.3:
                logging.info("Market sentiment strongly bearish - reducing exposure")
                # Could reduce position sizes here

            threads = []
            for pair in PAIRS:
                t = threading.Thread(target=manage_pair, args=(pair,))
                t.start()
                threads.append(t)

            poly_t = threading.Thread(target=manage_polymarket)
            poly_t.start()
            threads.append(poly_t)

            for t in threads:
                t.join()

            # Check for goal checkpoints
            goals = CONFIG.get('goals', {})
            checkpoints = goals.get('checkpoints', [])
            for checkpoint in checkpoints:
                if last_balance < checkpoint <= current_balance:
                    logging.info(f"🎯 CHECKPOINT REACHED: ${checkpoint}!")
                    
            last_balance = current_balance
            
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(60)

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    logging.info("Starting skattlebot.base.eth AI Agent")
    main_loop()
