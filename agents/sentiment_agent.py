#!/usr/bin/env python3
"""
Sentiment Agent
Monitors X, news, funding rates, and whale activity to generate trading signals.
Feeds signals to the core trading agent.
"""

import requests
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SENTIMENT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "sentiment_log.txt"),
        logging.StreamHandler()
    ]
)

SIGNALS_FILE = Path(__file__).parent.parent / "signals.json"

class SentimentAgent:
    def __init__(self):
        self.signal_history = deque(maxlen=1000)
        self.current_signals = {}
        
    def get_coingecko_sentiment(self, coin: str) -> dict:
        """Get sentiment from CoinGecko"""
        try:
            coin_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum', 
                'SOL': 'solana'
            }
            coin_id = coin_map.get(coin.upper(), coin.lower())
            
            res = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                timeout=10
            )
            data = res.json()
            
            return {
                'sentiment_up': data.get('sentiment_votes_up_percentage', 50),
                'sentiment_down': data.get('sentiment_votes_down_percentage', 50),
                'community_score': data.get('community_score', 0),
                'developer_score': data.get('developer_score', 0),
                'price_change_24h': data.get('market_data', {}).get('price_change_percentage_24h', 0)
            }
        except Exception as e:
            logging.error(f"CoinGecko error for {coin}: {e}")
            return {}
    
    def get_fear_greed_index(self) -> dict:
        """Get crypto fear & greed index"""
        try:
            res = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=10
            )
            data = res.json()
            if data.get('data'):
                fng = data['data'][0]
                return {
                    'value': int(fng.get('value', 50)),
                    'classification': fng.get('value_classification', 'Neutral')
                }
        except Exception as e:
            logging.error(f"Fear & Greed error: {e}")
        return {'value': 50, 'classification': 'Neutral'}
    
    def get_coinglass_data(self, symbol: str = 'BTC') -> dict:
        """Get funding rates and OI from Coinglass"""
        try:
            # Note: Coinglass API may require auth for some endpoints
            res = requests.get(
                f"https://open-api.coinglass.com/public/v2/funding?symbol={symbol}",
                timeout=10
            )
            data = res.json()
            
            if data.get('data'):
                return {
                    'funding_rate': data['data'].get('fundingRate', 0),
                    'oi_change_24h': data['data'].get('openInterestChange24h', 0),
                    'long_short_ratio': data['data'].get('longShortRatio', 1.0)
                }
        except Exception as e:
            logging.debug(f"Coinglass error: {e}")
        return {}
    
    def analyze_sentiment(self, coin: str) -> dict:
        """Generate aggregated sentiment signal for a coin"""
        cg_data = self.get_coingecko_sentiment(coin)
        fng = self.get_fear_greed_index()
        coinglass = self.get_coinglass_data(coin)
        
        # Calculate composite score (0-100, 50 = neutral)
        scores = []
        
        # CoinGecko sentiment
        if cg_data.get('sentiment_up'):
            scores.append(cg_data['sentiment_up'])
        
        # Fear & Greed (inverted for contrarian signals at extremes)
        fng_value = fng.get('value', 50)
        if fng_value < 20:  # Extreme fear = bullish contrarian
            scores.append(70)
        elif fng_value > 80:  # Extreme greed = bearish contrarian
            scores.append(30)
        else:
            scores.append(fng_value)
        
        # Price momentum
        price_change = cg_data.get('price_change_24h', 0)
        momentum_score = 50 + (price_change * 2)  # Scale price change
        momentum_score = max(0, min(100, momentum_score))
        scores.append(momentum_score)
        
        # Funding rate signal (negative funding = bullish)
        funding = coinglass.get('funding_rate', 0)
        if funding < -0.01:
            scores.append(70)
        elif funding > 0.01:
            scores.append(30)
        else:
            scores.append(50)
        
        composite = sum(scores) / len(scores) if scores else 50
        
        signal = {
            'coin': coin,
            'timestamp': datetime.now().isoformat(),
            'composite_score': round(composite, 2),
            'direction': 'bullish' if composite > 55 else 'bearish' if composite < 45 else 'neutral',
            'strength': abs(composite - 50) / 50,  # 0-1 scale
            'components': {
                'coingecko': cg_data,
                'fear_greed': fng,
                'coinglass': coinglass
            }
        }
        
        self.signal_history.append(signal)
        return signal
    
    def generate_all_signals(self) -> dict:
        """Generate signals for all tracked coins"""
        coins = ['BTC', 'ETH', 'SOL']
        signals = {}
        
        for coin in coins:
            try:
                signals[coin] = self.analyze_sentiment(coin)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logging.error(f"Error generating signal for {coin}: {e}")
                signals[coin] = {
                    'coin': coin,
                    'composite_score': 50,
                    'direction': 'neutral',
                    'strength': 0,
                    'error': str(e)
                }
        
        # Add market-wide signal
        market_avg = sum(s.get('composite_score', 50) for s in signals.values()) / len(signals)
        signals['MARKET'] = {
            'composite_score': round(market_avg, 2),
            'direction': 'bullish' if market_avg > 55 else 'bearish' if market_avg < 45 else 'neutral',
            'strength': abs(market_avg - 50) / 50
        }
        
        self.current_signals = signals
        self.save_signals()
        
        return signals
    
    def save_signals(self):
        """Save current signals to file for other agents"""
        try:
            output = {
                'updated_at': datetime.now().isoformat(),
                'signals': self.current_signals
            }
            SIGNALS_FILE.write_text(json.dumps(output, indent=2))
            signal_summary = [f"{k}={v.get('direction', '?')}" for k,v in self.current_signals.items()]
            logging.info(f"Signals saved: {signal_summary}")
        except Exception as e:
            logging.error(f"Error saving signals: {e}")
    
    def run(self, interval_seconds: int = 300):
        """Run continuous sentiment monitoring"""
        logging.info("Sentiment Agent starting...")
        
        while True:
            try:
                signals = self.generate_all_signals()
                
                # Log summary
                for coin, signal in signals.items():
                    if coin != 'MARKET':
                        logging.info(
                            f"{coin}: {signal.get('direction', 'neutral').upper()} "
                            f"(score={signal.get('composite_score', 50)}, "
                            f"strength={signal.get('strength', 0):.2f})"
                        )
                
            except Exception as e:
                logging.error(f"Sentiment loop error: {e}")
            
            time.sleep(interval_seconds)


if __name__ == "__main__":
    agent = SentimentAgent()
    agent.run(interval_seconds=300)  # Update every 5 minutes
