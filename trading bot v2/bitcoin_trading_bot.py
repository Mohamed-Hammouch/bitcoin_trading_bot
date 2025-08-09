# Green banner + pause before running main code

def show_banner():
    print("\033[92m" + r"""
 __  __       _                              _   _                                 _     
|  \/  |     | |                            | | | |                               | |    
| \  / | ___ | |__   ___  _ __ ___   ___  __| | | |__   __ _ _ __  _ __   ___  _ __| |__  
| |\/| |/ _ \| '_ \ / _ \| '_ ` _ \ / _ \/ _` | | '_ \ / _` | '_ \| '_ \ / _ \| '__| '_ \ 
| |  | | (_) | | | | (_) | | | | | |  __/ (_| | | | | | (_| | | | | | | | (_) | |  | | | |
|_|  |_|\___/|_| |_|\___/|_| |_| |_|\___|\__,_| |_| |_|\__,_|_| |_|_| |_|\___/|_|  |_| |_|
                                                                                          
                                                                                          
""" + "\033[0m")
    input("Press Enter to start...")

# Show banner first
show_banner()

# --- Your main code below ---
print("Running main program...")


import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Tuple
import logging

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'  # Green for BUY
    RED = '\033[91m'    # Red for SELL
    YELLOW = '\033[93m' # Yellow for HOLD
    BLUE = '\033[94m'   # Blue for info
    CYAN = '\033[96m'   # Cyan for highlights
    WHITE = '\033[97m'  # White for normal text
    BOLD = '\033[1m'    # Bold text
    END = '\033[0m'     # Reset color

class BitcoinTradingBot:
    def __init__(self, api_key: str = None, news_api_key: str = None, offline_mode: bool = False):
        """
        Initialize the Bitcoin trading bot
        
        Args:
            api_key: Exchange API key (for actual trading)
            news_api_key: News API key for sentiment analysis
            offline_mode: Run in offline mode with synthetic data
        """
        self.api_key = api_key
        self.news_api_key = news_api_key
        self.offline_mode = offline_mode
        self.position = 0  # Current position: 1 = long, -1 = short, 0 = neutral
        self.balance = 10000  # Starting balance in USD
        self.btc_holdings = 0
        self.trades = []
        
        # Offline mode data
        self.current_btc_price = 45000.0  # Starting price for offline mode
        self.price_history = []
        
        # Technical analysis parameters
        self.rsi_period = 14
        self.ma_short = 10
        self.ma_long = 50
        
        # Trading parameters
        self.max_position_size = 0.1  # 10% of portfolio per trade
        self.stop_loss = 0.02  # 2% stop loss
        self.take_profit = 0.04  # 4% take profit
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.offline_mode:
            self.logger.info("Running in OFFLINE MODE with synthetic data")

    def get_bitcoin_price(self) -> Dict:
        """Fetch current Bitcoin price and basic market data"""
        if self.offline_mode:
            # Simulate price movement in offline mode
            return self._simulate_current_price()
        
        try:
            # Using CoinGecko API (free, no key required)
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': 'bitcoin',
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'bitcoin' in data:
                    return {
                        'price': data['bitcoin']['usd'],
                        'change_24h': data['bitcoin'].get('usd_24h_change', 0),
                        'volume_24h': data['bitcoin'].get('usd_24h_vol', 0),
                        'timestamp': datetime.now()
                    }
            
            # Fallback to offline mode if API fails
            self.logger.warning("API failed, switching to offline mode")
            self.offline_mode = True
            return self._simulate_current_price()
            
        except Exception as e:
            self.logger.error(f"Error fetching price: {e}")
            self.logger.info("Switching to offline mode with synthetic data")
            self.offline_mode = True
            return self._simulate_current_price()

    def _simulate_current_price(self) -> Dict:
        """Simulate current Bitcoin price for offline mode"""
        # Add some realistic price movement
        change_percent = np.random.normal(0, 0.02)  # 2% volatility
        self.current_btc_price *= (1 + change_percent)
        
        # Keep price in reasonable range
        if self.current_btc_price < 20000:
            self.current_btc_price = 20000
        elif self.current_btc_price > 100000:
            self.current_btc_price = 100000
            
        daily_change = np.random.normal(0, 3)  # Random daily change
        
        return {
            'price': self.current_btc_price,
            'change_24h': daily_change,
            'volume_24h': 15000000000,  # Typical volume
            'timestamp': datetime.now()
        }

    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Fetch historical Bitcoin price data"""
        if self.offline_mode:
            return self._generate_synthetic_data(days)
            
        try:
            # First try CoinGecko API
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if days <= 7 else 'daily'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, dict) and 'prices' in data and data['prices']:
                    prices = data['prices']
                    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    self.logger.info(f"Successfully fetched {len(df)} data points")
                    return df
            
            # Switch to offline mode if API fails
            self.logger.warning("API failed, switching to offline mode")
            self.offline_mode = True
            return self._generate_synthetic_data(days)
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data, using synthetic data: {e}")
            self.offline_mode = True
            return self._generate_synthetic_data(days)

    def _generate_synthetic_data(self, days: int) -> pd.DataFrame:
        """Generate synthetic Bitcoin price data for demo purposes"""
        try:
            # Get current price as starting point
            current_data = self.get_bitcoin_price()
            if current_data:
                start_price = current_data['price']
            else:
                start_price = 45000  # Fallback price
            
            # Generate realistic price movement
            dates = pd.date_range(end=datetime.now(), periods=days * 24, freq='H')
            
            # Random walk with Bitcoin-like volatility
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0, 0.02, len(dates))  # 2% hourly volatility
            
            # Add some trend and cycles
            trend = np.linspace(-0.1, 0.1, len(dates))  # Slight upward trend
            cycle = 0.05 * np.sin(np.linspace(0, 4 * np.pi, len(dates)))  # Some cyclical movement
            
            returns = returns + trend / len(dates) + cycle / len(dates)
            
            # Calculate prices using cumulative returns
            prices = [start_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            df = pd.DataFrame({
                'price': prices
            }, index=dates)
            
            self.logger.info(f"Generated {len(df)} synthetic data points starting at ${start_price:,.2f}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic data: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_moving_averages(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate short and long moving averages"""
        ma_short = prices.rolling(window=self.ma_short).mean()
        ma_long = prices.rolling(window=self.ma_long).mean()
        return ma_short, ma_long

    def get_news_sentiment(self) -> float:
        """
        Fetch Bitcoin-related news and analyze sentiment
        Returns sentiment score: -1 (very negative) to 1 (very positive)
        """
        if not self.news_api_key:
            self.logger.warning("No news API key provided, using neutral sentiment")
            return 0.0
        
        try:
            # Using NewsAPI for demonstration
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'bitcoin OR cryptocurrency OR BTC',
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.now() - timedelta(hours=24)).isoformat(),
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            articles = response.json().get('articles', [])
            
            # Simple sentiment analysis based on keywords
            positive_words = ['surge', 'bull', 'rally', 'gain', 'rise', 'pump', 'moon', 'adoption', 'breakthrough']
            negative_words = ['crash', 'bear', 'dump', 'fall', 'decline', 'regulation', 'ban', 'hack', 'scam']
            
            sentiment_scores = []
            for article in articles[:20]:  # Analyze latest 20 articles
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                content = f"{title} {description}"
                
                positive_count = sum(1 for word in positive_words if word in content)
                negative_count = sum(1 for word in negative_words if word in content)
                
                if positive_count + negative_count > 0:
                    score = (positive_count - negative_count) / (positive_count + negative_count)
                    sentiment_scores.append(score)
            
            return np.mean(sentiment_scores) if sentiment_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error fetching news sentiment: {e}")
            return 0.0

    def analyze_market_conditions(self) -> Dict:
        """Analyze current market conditions using technical indicators"""
        # Get historical data
        df = self.get_historical_data(days=60)
        if df.empty:
            return {'signal': 'HOLD', 'strength': 0, 'reason': 'No data available'}
        
        current_price = df['price'].iloc[-1]
        
        # Calculate technical indicators
        rsi = self.calculate_rsi(df['price'])
        ma_short, ma_long = self.calculate_moving_averages(df['price'])
        
        current_rsi = rsi.iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        
        # Calculate price volatility
        volatility = df['price'].pct_change().std() * 100
        
        # Analyze trends
        signals = []
        
        # RSI signals
        if current_rsi < 30:
            signals.append(('BUY', 0.7, 'RSI oversold'))
        elif current_rsi > 70:
            signals.append(('SELL', 0.7, 'RSI overbought'))
        
        # Moving average crossover
        if current_ma_short > current_ma_long:
            ma_signal_strength = min((current_ma_short - current_ma_long) / current_ma_long, 0.5)
            signals.append(('BUY', ma_signal_strength, 'MA bullish crossover'))
        else:
            ma_signal_strength = min((current_ma_long - current_ma_short) / current_ma_short, 0.5)
            signals.append(('SELL', ma_signal_strength, 'MA bearish crossover'))
        
        # Volume analysis (simplified)
        recent_volume = df['price'].rolling(7).std()
        if len(recent_volume) > 14:
            volume_trend = recent_volume.iloc[-7:].mean() / recent_volume.iloc[-14:-7].mean()
            if volume_trend > 1.2:
                signals.append(('BUY', 0.3, 'Increasing volatility'))
        
        return {
            'current_price': current_price,
            'rsi': current_rsi,
            'ma_short': current_ma_short,
            'ma_long': current_ma_long,
            'volatility': volatility,
            'signals': signals
        }

    def make_trading_decision(self) -> Dict:
        """Make trading decision based on market analysis and news sentiment"""
        # Get market analysis
        market_analysis = self.analyze_market_conditions()
        if not market_analysis.get('signals'):
            return {'action': 'HOLD', 'amount': 0, 'reason': 'Insufficient market data'}
        
        # Get timing estimates
        timing = self.estimate_trade_timing(market_analysis)
        
        # Get news sentiment
        news_sentiment = self.get_news_sentiment()
        
        # Combine signals
        buy_score = 0
        sell_score = 0
        reasons = []
        
        for signal, strength, reason in market_analysis['signals']:
            if signal == 'BUY':
                buy_score += strength
                reasons.append(f"BUY: {reason} (strength: {strength:.2f})")
            else:
                sell_score += strength
                reasons.append(f"SELL: {reason} (strength: {strength:.2f})")
        
        # Factor in news sentiment
        if news_sentiment > 0.2:
            buy_score += 0.3
            reasons.append(f"Positive news sentiment: {news_sentiment:.2f}")
        elif news_sentiment < -0.2:
            sell_score += 0.3
            reasons.append(f"Negative news sentiment: {news_sentiment:.2f}")
        
        # Risk management - reduce position size in high volatility
        volatility_factor = max(0.5, 1 - (market_analysis['volatility'] - 2) / 10)
        
        # Make decision
        if buy_score > sell_score and buy_score > 0.5:
            action = 'BUY'
            confidence = min(buy_score, 1.0)
            amount = self.max_position_size * confidence * volatility_factor
        elif sell_score > buy_score and sell_score > 0.5:
            action = 'SELL'
            confidence = min(sell_score, 1.0)
            amount = self.max_position_size * confidence * volatility_factor
        else:
            action = 'HOLD'
            amount = 0
            confidence = 0
        
        return {
            'action': action,
            'amount': amount,
            'confidence': confidence,
            'current_price': market_analysis['current_price'],
            'reasons': reasons,
            'market_analysis': market_analysis,
            'news_sentiment': news_sentiment,
            'timing_estimate': timing
        }

    def execute_trade(self, decision: Dict) -> Dict:
        """
        Execute trade based on decision (SIMULATION MODE)
        In real implementation, this would connect to exchange API
        """
        if decision['action'] == 'HOLD':
            return {'status': 'No trade executed', 'action': 'HOLD'}
        
        current_price = decision['current_price']
        amount_usd = decision['amount'] * self.balance
        
        if decision['action'] == 'BUY' and amount_usd <= self.balance:
            btc_to_buy = amount_usd / current_price
            self.btc_holdings += btc_to_buy
            self.balance -= amount_usd
            self.position = 1
            
            trade_record = {
                'timestamp': datetime.now(),
                'action': 'BUY',
                'amount_btc': btc_to_buy,
                'amount_usd': amount_usd,
                'price': current_price,
                'confidence': decision['confidence']
            }
            self.trades.append(trade_record)
            
            return {
                'status': 'BUY executed',
                'btc_bought': btc_to_buy,
                'amount_spent': amount_usd,
                'new_balance': self.balance,
                'new_btc_holdings': self.btc_holdings
            }
        
        elif decision['action'] == 'SELL' and self.btc_holdings > 0:
            btc_to_sell = min(self.btc_holdings, decision['amount'] * self.balance / current_price)
            usd_received = btc_to_sell * current_price
            self.btc_holdings -= btc_to_sell
            self.balance += usd_received
            self.position = -1 if self.btc_holdings == 0 else 0
            
            trade_record = {
                'timestamp': datetime.now(),
                'action': 'SELL',
                'amount_btc': btc_to_sell,
                'amount_usd': usd_received,
                'price': current_price,
                'confidence': decision['confidence']
            }
            self.trades.append(trade_record)
            
            return {
                'status': 'SELL executed',
                'btc_sold': btc_to_sell,
                'amount_received': usd_received,
                'new_balance': self.balance,
                'new_btc_holdings': self.btc_holdings
            }
        
        return {'status': 'Trade conditions not met', 'action': decision['action']}

    def estimate_trade_timing(self, market_analysis: Dict) -> Dict:
        """
        Estimate optimal timing for next trade based on market conditions
        Returns estimated time ranges for BUY/SELL opportunities
        """
        current_rsi = market_analysis.get('rsi', 50)
        volatility = market_analysis.get('volatility', 5)
        
        # Base timing estimates (in hours)
        base_timing = {
            'next_buy_opportunity': None,
            'next_sell_opportunity': None,
            'confidence': 'low'
        }
        
        # RSI-based timing predictions
        if current_rsi > 50:  # Moving toward overbought
            hours_to_overbought = max(1, (70 - current_rsi) / 2)  # Rough estimate
            base_timing['next_sell_opportunity'] = hours_to_overbought
        
        if current_rsi < 50:  # Moving toward oversold
            hours_to_oversold = max(1, (current_rsi - 30) / 2)  # Rough estimate
            base_timing['next_buy_opportunity'] = hours_to_oversold
        
        # Volatility adjustments
        if volatility > 8:  # High volatility = faster moves
            multiplier = 0.5
            base_timing['confidence'] = 'high'
        elif volatility < 3:  # Low volatility = slower moves
            multiplier = 2.0
            base_timing['confidence'] = 'low'
        else:
            multiplier = 1.0
            base_timing['confidence'] = 'medium'
        
        # Apply volatility adjustment
        if base_timing['next_buy_opportunity']:
            base_timing['next_buy_opportunity'] *= multiplier
        if base_timing['next_sell_opportunity']:
            base_timing['next_sell_opportunity'] *= multiplier
        
        # Convert to time estimates
        timing_result = {
            'next_buy_opportunity': None,
            'next_sell_opportunity': None,
            'confidence': base_timing['confidence']
        }
        
        if base_timing['next_buy_opportunity']:
            hours = base_timing['next_buy_opportunity']
            if hours < 1:
                timing_result['next_buy_opportunity'] = f"{int(hours * 60)} minutes"
            elif hours < 24:
                timing_result['next_buy_opportunity'] = f"{hours:.1f} hours"
            else:
                timing_result['next_buy_opportunity'] = f"{hours/24:.1f} days"
        
        if base_timing['next_sell_opportunity']:
            hours = base_timing['next_sell_opportunity']
            if hours < 1:
                timing_result['next_sell_opportunity'] = f"{int(hours * 60)} minutes"
            elif hours < 24:
                timing_result['next_sell_opportunity'] = f"{hours:.1f} hours"
            else:
                timing_result['next_sell_opportunity'] = f"{hours/24:.1f} days"
        
        return timing_result
        """Check if stop loss or take profit should be triggered"""
        if not self.trades or self.btc_holdings <= 0:
            return {'action': 'HOLD'}
        
        last_trade = self.trades[-1]
        if last_trade['action'] != 'BUY':
            return {'action': 'HOLD'}
        
        current_price_data = self.get_bitcoin_price()
        if not current_price_data:
            return {'action': 'HOLD'}
        
        current_price = current_price_data['price']
        entry_price = last_trade['price']
        
        price_change = (current_price - entry_price) / entry_price
        
        if price_change <= -self.stop_loss:
            return {
                'action': 'SELL',
                'reason': f'Stop loss triggered: {price_change:.2%}',
                'amount': 1.0  # Sell all
            }
        elif price_change >= self.take_profit:
            return {
                'action': 'SELL',
                'reason': f'Take profit triggered: {price_change:.2%}',
                'amount': 1.0  # Sell all
            }
        
        return {'action': 'HOLD'}

    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        current_price_data = self.get_bitcoin_price()
        if current_price_data:
            btc_value = self.btc_holdings * current_price_data['price']
            return self.balance + btc_value
        return self.balance

    def run_trading_cycle(self) -> Dict:
        """Run one complete trading cycle"""
        self.logger.info("Starting trading cycle...")
        
        # Check stop loss/take profit first
        sl_tp_decision = self.check_stop_loss_take_profit()
        if sl_tp_decision['action'] != 'HOLD':
            decision = {
                'action': sl_tp_decision['action'],
                'amount': sl_tp_decision['amount'],
                'current_price': self.get_bitcoin_price()['price'],
                'confidence': 1.0,
                'reasons': [sl_tp_decision['reason']]
            }
            result = self.execute_trade(decision)
            result['trigger'] = 'Stop Loss / Take Profit'
            return result
        
        # Make normal trading decision
        decision = self.make_trading_decision()
        result = self.execute_trade(decision)
        
        # Log results
        portfolio_value = self.get_portfolio_value()
        self.logger.info(f"Decision: {decision['action']}")
        self.logger.info(f"Confidence: {decision['confidence']:.2f}")
        self.logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
        
        return {
            'decision': decision,
            'execution': result,
            'portfolio_value': portfolio_value,
            'total_return': (portfolio_value - 10000) / 10000
        }

    def backtest_strategy(self, days: int = 30) -> Dict:
        """Backtest the trading strategy on historical data"""
        df = self.get_historical_data(days)
        if df.empty:
            return {'error': 'No historical data available'}
        
        # Reset for backtest
        original_balance = self.balance
        original_holdings = self.btc_holdings
        self.balance = 10000
        self.btc_holdings = 0
        self.trades = []
        
        results = []
        
        for i in range(len(df) - 1):
            current_data = df.iloc[:i+1]
            if len(current_data) < max(self.ma_long, self.rsi_period):
                continue
            
            # Simulate market analysis with historical data
            current_price = current_data['price'].iloc[-1]
            rsi = self.calculate_rsi(current_data['price']).iloc[-1]
            ma_short, ma_long = self.calculate_moving_averages(current_data['price'])
            
            # Simple strategy for backtest
            if rsi < 30 and ma_short.iloc[-1] > ma_long.iloc[-1]:
                action = 'BUY'
            elif rsi > 70 or (self.btc_holdings > 0 and ma_short.iloc[-1] < ma_long.iloc[-1]):
                action = 'SELL'
            else:
                action = 'HOLD'
            
            if action == 'BUY' and self.balance > 100:
                amount_to_invest = self.balance * 0.1
                btc_bought = amount_to_invest / current_price
                self.btc_holdings += btc_bought
                self.balance -= amount_to_invest
                
            elif action == 'SELL' and self.btc_holdings > 0:
                usd_received = self.btc_holdings * current_price
                self.balance += usd_received
                self.btc_holdings = 0
            
            portfolio_value = self.balance + (self.btc_holdings * current_price)
            results.append({
                'date': current_data.index[-1],
                'price': current_price,
                'action': action,
                'portfolio_value': portfolio_value,
                'rsi': rsi
            })
        
        final_value = self.balance + (self.btc_holdings * df['price'].iloc[-1])
        total_return = (final_value - 10000) / 10000
        
        # Restore original state
        self.balance = original_balance
        self.btc_holdings = original_holdings
        
        return {
            'initial_value': 10000,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len([r for r in results if r['action'] != 'HOLD']),
            'results': results
        }

    def display_status(self):
        """Display current bot status with colors"""
        current_price_data = self.get_bitcoin_price()
        portfolio_value = self.get_portfolio_value()
        
        print("\n" + "="*50)
        print(f"{Colors.BOLD}{Colors.CYAN}BITCOIN TRADING BOT STATUS{Colors.END}")
        print("="*50)
        print(f"{Colors.WHITE}Current BTC Price: {Colors.BOLD}${current_price_data['price']:,.2f}{Colors.END}")
        
        # Color code the 24h change
        change = current_price_data['change_24h']
        if change > 0:
            print(f"{Colors.GREEN}24h Change: +{change:.2f}%{Colors.END}")
        else:
            print(f"{Colors.RED}24h Change: {change:.2f}%{Colors.END}")
            
        print(f"{Colors.WHITE}Cash Balance: ${self.balance:,.2f}{Colors.END}")
        print(f"{Colors.WHITE}BTC Holdings: {self.btc_holdings:.6f} BTC{Colors.END}")
        
        # Color code portfolio performance
        total_return = ((portfolio_value - 10000) / 10000) * 100
        if total_return > 0:
            print(f"{Colors.WHITE}Portfolio Value: {Colors.GREEN}${portfolio_value:,.2f}{Colors.END}")
            print(f"{Colors.GREEN}Total Return: +{total_return:.2f}%{Colors.END}")
        else:
            print(f"{Colors.WHITE}Portfolio Value: {Colors.RED}${portfolio_value:,.2f}{Colors.END}")
            print(f"{Colors.RED}Total Return: {total_return:.2f}%{Colors.END}")
            
        # Color code position
        if self.position == 1:
            print(f"{Colors.GREEN}Current Position: LONG{Colors.END}")
        elif self.position == -1:
            print(f"{Colors.RED}Current Position: SHORT{Colors.END}")
        else:
            print(f"{Colors.YELLOW}Current Position: NEUTRAL{Colors.END}")
            
        print(f"{Colors.WHITE}Total Trades: {len(self.trades)}{Colors.END}")
        print("="*50)


# Example usage and demo
def demo_trading_bot():
    """Demonstrate the trading bot functionality"""
    print("Bitcoin Short-Term Trading Bot Demo")
    print("===================================")
    print("Note: Running in OFFLINE MODE due to network restrictions")
    print("Using realistic synthetic data for demonstration")
    print()
    
    # Initialize bot in offline mode
    bot = BitcoinTradingBot(offline_mode=True)
    
    # Display initial status
    bot.display_status()
    
    # Run a trading cycle
    print("\nRunning trading analysis...")
    
    try:
        result = bot.run_trading_cycle()
        
        print(f"\nTrading Decision: {result['decision']['action']}")
        print(f"Confidence: {result['decision']['confidence']:.2f}")
        print(f"Reasons: {', '.join(result['decision']['reasons'])}")
        
        if result['execution']['status']:
            print(f"Execution: {result['execution']['status']}")
        
        # Display updated status
        bot.display_status()
    except Exception as e:
        print(f"Error in trading cycle: {e}")
    
    # Run backtest
    print("\nRunning 30-day backtest with synthetic data...")
    try:
        backtest = bot.backtest_strategy(30)
        
        if 'error' not in backtest:
            print(f"Backtest Results:")
            print(f"  Initial Value: ${backtest['initial_value']:,.2f}")
            print(f"  Final Value: ${backtest['final_value']:,.2f}")
            print(f"  Total Return: {backtest['total_return']*100:+.2f}%")
            print(f"  Total Trades: {backtest['total_trades']}")
            
            # Show some trade details
            if backtest['results']:
                winning_trades = [r for r in backtest['results'] if r['action'] != 'HOLD']
                if winning_trades:
                    print(f"  Sample trades: {len(winning_trades)} total")
                    
        else:
            print(f"Backtest error: {backtest.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error in backtest: {e}")
        
    print("\n" + "="*50)
    print("OFFLINE DEMO COMPLETE")
    print("="*50)
    print("This demonstrates the trading strategy using synthetic data.")
    print("Key features shown:")
    print("✓ Technical analysis (RSI, Moving Averages)")
    print("✓ Risk management (stop-loss, position sizing)")
    print("✓ Portfolio tracking and performance metrics")
    print("✓ Backtesting capabilities")
    print()
    print("In a real environment with internet access:")
    print("• Uses live Bitcoin price data from exchanges")
    print("• Incorporates real-time news sentiment")
    print("• Can execute actual trades through exchange APIs")
    print("="*50)

if __name__ == "__main__":
    # Run the demo
    demo_trading_bot()
    
    # Example of continuous trading (commented out for safety)
    """
    bot = BitcoinTradingBot()
    
    while True:
        try:
            result = bot.run_trading_cycle()
            print(f"Cycle completed at {datetime.now()}")
            time.sleep(300)  # Wait 5 minutes between cycles
        except KeyboardInterrupt:
            print("Trading bot stopped by user")
            break
        except Exception as e:
            print(f"Error in trading cycle: {e}")
            time.sleep(60)  # Wait 1 minute before retry
    """
    input("Press Enter to exit...")
