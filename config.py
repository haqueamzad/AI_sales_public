# /Users/amzadhaque/AI_sales/src/config.py
# config.py (sanitized)
# Configuration file for the AI Sales Trading Pipeline

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('/Users/amzadhaque/AI_sales/src/.env')

# Define function to calculate date range based on BACKTEST_DAYS
def calculate_date_range(end_date: str = None, backtest_days: int = None) -> tuple[str, str]:
    """Calculate start and end dates for backtesting.
    
    Args:
        end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
        backtest_days (int, optional): Number of days to look back. Defaults to BACKTEST_DAYS.
    
    Returns:
        tuple[str, str]: (start_date, end_date) in YYYY-MM-DD format.
    """
    backtest_days = int(os.getenv('BACKTEST_DAYS', backtest_days or 90))
    end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
    start = end - timedelta(days=backtest_days)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

# Configuration variables from .env
TICKERS = ['NVDA', 'AAPL', 'AVGO', 'MSFT', 'AMZN', 'META', 'LLY', 'JPM', 'TSLA', 'UBER']
TECHNICAL_INDICATORS = {
    'rsi_window': int(os.getenv('rsi_window', 14)),
    'macd_fast': int(os.getenv('macd_fast', 12)),
    'macd_slow': int(os.getenv('macd_slow', 26)),
    'macd_signal': int(os.getenv('macd_signal', 9)),
    'bb_window': int(os.getenv('bb_window', 20)),
    'bb_std': float(os.getenv('bb_std', 2.0)),
    'atr_window': int(os.getenv('atr_window', 14)),
    'adx_window': int(os.getenv('adx_window', 14)),
    'volatility_window': int(os.getenv('volatility_window', 10)),
    'momentum_window': int(os.getenv('momentum_window', 10)),
    'volume_window': int(os.getenv('volume_window', 20)),
    'sma_5_window': int(os.getenv('sma_5_window', 5)),
    'sma_20_window': int(os.getenv('sma_20_window', 20))
}
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
POLYGON_API_RATE_LIMIT = int(os.getenv('POLYGON_API_RATE_LIMIT', 5))
SENTIMENT_WEIGHT_X = float(os.getenv('SENTIMENT_WEIGHT_X', 0.5))
SENTIMENT_WEIGHT_NEWS = float(os.getenv('SENTIMENT_WEIGHT_NEWS', 0.3))
VIX_THRESHOLD = float(os.getenv('VIX_THRESHOLD', 30.0))
TRADE_COOLDOWN = int(os.getenv('TRADE_COOLDOWN', 5))
ATR_MULTIPLIER_STOP_LOSS = float(os.getenv('ATR_MULTIPLIER_STOP_LOSS', 1.5))
ADAPTIVE_STOP_LOSS = os.getenv('ADAPTIVE_STOP_LOSS', 'True')
ATR_MULTIPLIER_TAKE_PROFIT = float(os.getenv('ATR_MULTIPLIER_TAKE_PROFIT', 1.5))
ADAPTIVE_TAKE_PROFIT = os.getenv('ADAPTIVE_TAKE_PROFIT', 'True')
DYNAMIC_POSITION_SIZING = os.getenv('DYNAMIC_POSITION_SIZING', 'True')  # Added: Enable dynamic position sizing
RL_MODEL_SAVE_PATH = os.getenv('RL_MODEL_SAVE_PATH', '/Users/amzadhaque/AI_sales/models/sac_trading_model')
RL_TRAINING_STEPS = int(os.getenv('RL_TRAINING_STEPS', 200000))
RL_LEARNING_RATE = float(os.getenv('RL_LEARNING_RATE', 0.00005))
RL_GAMMA = float(os.getenv('RL_GAMMA', 0.99))
RL_ENT_COEF = float(os.getenv('RL_ENT_COEF', 0.01))
RL_BATCH_SIZE = int(os.getenv('RL_BATCH_SIZE', 256))
BACKTEST_DAYS = int(os.getenv('BACKTEST_DAYS', 90))
N_TRIALS = int(os.getenv('N_TRIALS', 100))
TIMESTEPS_PER_TRIAL = int(os.getenv('TIMESTEPS_PER_TRIAL', 10000))
DEVICE = os.getenv('DEVICE', 'auto')
HYPERPARAMETER_TUNING_ENABLED = os.getenv('HYPERPARAMETER_TUNING_ENABLED', 'True')
SENTIMENT_ANALYSIS = os.getenv('ENABLE_SENTIMENT_ANALYSIS', 'True')
MODEL_INTERPRETABILITY_ENABLED = os.getenv('MODEL_INTERPRETABILITY_ENABLED', 'True')
MODEL_INTERPRETABILITY_TOOL = os.getenv('MODEL_INTERPRETABILITY_TOOL', 'GREEN')
SYNTHETIC_DATA_ENABLED = os.getenv('SYNTHETIC_DATA_ENABLED', 'True')
GAN_AUGMENTATION_ENABLED = os.getenv('GAN_AUGMENTATION_ENABLED', 'False')
ADVANCED_GAN_ENABLED = os.getenv('ADVANCED_GAN_ENABLED', 'False')
PORTFOLIO_OPTIMIZATION_ENABLED = os.getenv('PORTFOLIO_OPTIMIZATION_ENABLED', 'True')
DATABASE_ENABLED = os.getenv('DATABASE_ENABLED', 'False')
DATABASE_PATH = os.getenv('DATABASE_PATH', '/Users/amzadhaque/AI_sales/src/trading.db')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', '/Users/amzadhaque/AI_sales/src/trading.log')
CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'False')
CACHE_DIR = os.getenv('CACHE_DIR', '/Users/amzadhaque/AI_sales/src/cache')
EMAIL_SETTINGS = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', 587)),
    'sender_email': os.getenv('SENDER_EMAIL', ''),
    'smtp_password': os.getenv('SMTP_PASSWORD', ''),
    'recipient_email': os.getenv('RECIPIENT_EMAIL', '')
}
LIVE_TRADING_ENABLED = os.getenv('LIVE_TRADING_ENABLED', 'False')
LIVE_DATA_MODE = os.getenv('LIVE_DATA_MODE', 'False')
PAPER_TRADING = os.getenv('PAPER_TRADING', 'True')

# Trading Environment Parameters
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 100000.0))  # Initial portfolio balance in USD
MAX_TRADES_PER_EPISODE = int(os.getenv('MAX_TRADES_PER_EPISODE', 100))  # Max trades per episode
MAX_TOTAL_TRADES = int(os.getenv('MAX_TOTAL_TRADES', 1000))  # Max total trades across all episodes
MAX_TRADES_PER_STEP = int(os.getenv('MAX_TRADES_PER_STEP', 10))  # Max trades per step
SLIPPAGE_BASE = float(os.getenv('SLIPPAGE_BASE', 0.001))  # Base slippage for trades
MIN_SHARES = float(os.getenv('MIN_SHARES', 1.0))  # Minimum shares per trade
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', 0.05))  # Stop loss percentage
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', 0.10))  # Take profit percentage
TRAILING_STOP_PCT = float(os.getenv('TRAILING_STOP_PCT', 0.03))  # Trailing stop percentage
MAX_POSITION = float(os.getenv('MAX_POSITION', 1000.0))  # Max shares per position
MAX_POSITION_EXPOSURE = float(os.getenv('MAX_POSITION_EXPOSURE', 0.1))  # Max exposure per position
MAX_PORTFOLIO_EXPOSURE = float(os.getenv('MAX_PORTFOLIO_EXPOSURE', 0.5))  # Max portfolio exposure
MIN_LIQUIDITY_THRESHOLD = float(os.getenv('MIN_LIQUIDITY_THRESHOLD', 10000.0))  # Min liquidity for trading
MAX_CONSECUTIVE_BUYS = int(os.getenv('MAX_CONSECUTIVE_BUYS', 5))  # Max consecutive buy orders
MAX_DRAWDOWN_PCT = float(os.getenv('MAX_DRAWDOWN_PCT', 0.2))  # Max portfolio drawdown
MAX_FILL_PCT = float(os.getenv('MAX_FILL_PCT', 0.1))  # Max fill percentage for orders
TRANSACTION_COST = float(os.getenv('TRANSACTION_COST', 0.001))  # Transaction cost percentage
IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')  # Interactive Brokers host
IBKR_PORT = int(os.getenv('IBKR_PORT', 7497))  # Interactive Brokers port
IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', 1))  # Interactive Brokers client ID
MAX_DAILY_TRADES_PER_TICKER = int(os.getenv('MAX_DAILY_TRADES_PER_TICKER', 5))  # Max daily trades per ticker
