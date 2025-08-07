import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import time
import secrets
import asyncio
import sys
import glob
import json
from collections import deque
import pytz
from ratelimit import limits, sleep_and_retry
import pickle
import shutil
from typing import Dict, List, Tuple, Optional, Any
import pandas_market_calendars as mcal
import multiprocessing
from scipy.stats import pearsonr
import stat
import getpass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import sqlite3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import Dataset, DataLoader
from gymnasium import Env
from gymnasium.spaces import Box
from polygon import RESTClient
from ib_insync import IB, Stock, util
from config import *
import nest_asyncio
from functools import lru_cache

nest_asyncio.apply()

logger = logging.getLogger('ai_sales')

# Log library versions for diagnostics
package_import_mapping = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'stable_baselines3': 'stable_baselines3',
    'torch': 'torch',
    'optuna': 'optuna',
    'polygon': 'polygon',
    'ib_insync': 'ib_insync',
    'pandas_market_calendars': 'pandas_market_calendars',
    'ratelimit': 'ratelimit',
    'scipy': 'scipy',
    'requests': 'requests',
    'vaderSentiment': 'vaderSentiment',
    'aiohttp': 'aiohttp',
    'sqlite3': 'sqlite3'
}
missing_packages = []
for pkg, import_name in package_import_mapping.items():
    try:
        module = __import__(import_name)
        logger.info(f"Imported {pkg} version: {getattr(module, '__version__', 'unknown')}")
    except ImportError:
        missing_packages.append(pkg)
if missing_packages:
    logger.error(f"Missing required packages: {missing_packages}. Please install them using 'pip install {' '.join(missing_packages)}'")
    sys.exit(1)

# Validate configuration
required_configs = [
    ('POLYGON_API_KEY', POLYGON_API_KEY),
    ('EMAIL_SETTINGS', {
        'smtp_server': EMAIL_SETTINGS.get('smtp_server'),
        'sender_email': EMAIL_SETTINGS.get('sender_email'),
        'smtp_password': EMAIL_SETTINGS.get('smtp_password'),
        'recipient_email': EMAIL_SETTINGS.get('recipient_email')
    })
]
for config_name, config_value in required_configs:
    if not config_value:
        logger.error(f"{config_name} not found or incomplete in environment variables. Please set it and try again.")
        sys.exit(1)

# Initialize SQLite database
if DATABASE_ENABLED:
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    date TEXT,
                    ticker TEXT,
                    action TEXT,
                    shares REAL,
                    price REAL,
                    proceeds REAL,
                    cost REAL,
                    fees REAL,
                    pnl REAL,
                    stop_loss BOOLEAN,
                    take_profit BOOLEAN,
                    trailing_stop BOOLEAN,
                    forced BOOLEAN
                )
            ''')
            conn.commit()
        logger.info("Initialized SQLite database for trades")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize SQLite database: {e}")
        sys.exit(1)

NYSE_CAL = mcal.get_calendar('NYSE')

class CustomCallback(BaseCallback):
    def __init__(self, checkpoint_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq
        self.total_timesteps = 0

    def set_total_timesteps(self, total_timesteps: int) -> None:
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_freq == 0:
            model_path = f"{RL_MODEL_SAVE_PATH}_checkpoint_{self.model.__class__.__name__.lower()}_{self.n_calls}"
            self.model.save(model_path)
            logger.info(f"Saved checkpoint at {model_path}")
        return True

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, nhead: int = 8, num_layers: int = 3):
        super().__init__(observation_space, features_dim=128)
        d_model = self.features_dim
        self.d_model = d_model
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        input_dim = observation_space.shape[0]
        self.input_fc = nn.Linear(input_dim, d_model)
        self.output_fc = nn.Linear(d_model, d_model)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        try:
            if len(observations.shape) == 1:
                observations = observations.unsqueeze(0)
            batch_size = observations.shape[0]
            x = self.input_fc(observations)
            x = x.unsqueeze(0)
            x = self.transformer(x, x)
            x = x.permute(1, 0, 2)
            x = x.squeeze(1)
            x = self.output_fc(x)
            return x
        except Exception as e:
            logger.error(f"Error in TransformerFeaturesExtractor forward: {e}")
            return torch.zeros((batch_size, self.features_dim), device=DEVICE)

class TradingEnv(Env):
    """A Gymnasium environment for trading with reinforcement learning."""
    def __init__(
        self,
        df: pd.DataFrame,
        top_stocks_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        sentiment_dfs: Dict[str, pd.DataFrame],
        backtest_mode: bool = True,
        training_mode: bool = True,
        dynamic_position_sizing: bool = DYNAMIC_POSITION_SIZING,
        market_open: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        super().__init__()
        # Validate input DataFrames
        if df.empty and backtest_mode:
            logger.error("Input DataFrame 'df' is empty in backtest mode. Cannot initialize TradingEnv.")
            raise ValueError("Input DataFrame 'df' is empty in backtest mode.")
        required_columns = ['ticker', 'date', 'close', 'volume', 'high', 'low', 'open', 'adjusted_close']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Input DataFrame 'df' missing required columns: {missing_cols}. Expected: {required_columns}, Got: {df.columns.tolist()}")
            raise ValueError(f"Input DataFrame 'df' missing required columns: {missing_cols}")
        if not top_stocks_df.empty and not all(col in top_stocks_df.columns for col in ['ticker', 'sentiment', 'final_score']):
            logger.error(f"top_stocks_df missing required columns. Expected: ['ticker', 'sentiment', 'final_score'], Got: {top_stocks_df.columns.tolist()}")
            raise ValueError("top_stocks_df missing required columns.")
        if not vix_df.empty and not all(col in vix_df.columns for col in ['date', 'vix', 'sp500_return', 'sp500_momentum']):
            logger.error(f"vix_df missing required columns. Expected: ['date', 'vix', 'sp500_return', 'sp500_momentum'], Got: {vix_df.columns.tolist()}")
            raise ValueError("vix_df missing required columns.")
        # Validate sentiment_dfs
        validated_sentiment_dfs = {}
        for ticker, sdf in sentiment_dfs.items():
            if not isinstance(sdf, pd.DataFrame):
                logger.warning(f"sentiment_dfs[{ticker}] is not a DataFrame (type: {type(sdf)}). Creating fallback.")
                dates = pd.date_range(start=start_date or datetime.now() - timedelta(days=90), end=end_date or datetime.now(), freq='B')
                validated_sentiment_dfs[ticker] = pd.DataFrame({"date": dates, "sentiment": [0.0] * len(dates)})
            elif sdf.empty or not all(col in sdf.columns for col in ['date', 'sentiment']):
                logger.warning(f"sentiment_dfs[{ticker}] is empty or missing 'date', 'sentiment' columns. Columns: {sdf.columns.tolist() if isinstance(sdf, pd.DataFrame) else 'N/A'}. Creating fallback.")
                dates = pd.date_range(start=start_date or datetime.now() - timedelta(days=90), end=end_date or datetime.now(), freq='B')
                validated_sentiment_dfs[ticker] = pd.DataFrame({"date": dates, "sentiment": [0.0] * len(dates)})
            else:
                validated_sentiment_dfs[ticker] = sdf.copy()
        self.sentiment_dfs = validated_sentiment_dfs
        # Validate date range
        try:
            start_date = pd.to_datetime(start_date) if start_date else pd.to_datetime(df['date'].min())
            end_date = pd.to_datetime(end_date) if end_date else pd.to_datetime(df['date'].max())
            if end_date < start_date:
                logger.error(f"end_date ({end_date}) is before start_date ({start_date})")
                raise ValueError("end_date cannot be before start_date")
        except Exception as e:
            logger.error(f"Invalid date format: start_date={start_date}, end_date={end_date}, error={e}")
            raise ValueError(f"Invalid date format: {e}")
        self.df = df
        self.top_stocks_df = top_stocks_df
        self.vix_df = vix_df
        self.backtest_mode = backtest_mode
        self.training_mode = training_mode
        self.dynamic_position_sizing = dynamic_position_sizing
        self.market_open = market_open
        self.start_date = start_date
        self.end_date = end_date
        # Initialize tickers and dates
        self.tickers = sorted(df['ticker'].unique()) if not df.empty else []
        self.num_tickers = len(self.tickers)
        self.dates = pd.Series(pd.to_datetime(df['date'].unique())).sort_values().tolist() if not df.empty else []
        self.max_steps = len(self.dates) if backtest_mode else float('inf')
        self.current_step = 0
        # Initialize data arrays efficiently
        try:
            pivot_columns = ['close', 'volume', 'avg_volume', 'rsi', 'macd', 'bb_width', 'atr', 'adx', 'volatility', 'momentum', 'volume_trend', 'sma_5', 'sma_20']
            self.data_arrays = {}
            expected_shape = (self.max_steps if backtest_mode else 1, self.num_tickers)
            for col in pivot_columns:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' missing in df. Initializing with zeros.")
                    array = np.zeros(expected_shape, dtype=np.float32)
                else:
                    pivot_df = df.pivot(index='date', columns='ticker', values=col).reindex(columns=self.tickers)
                    array = pivot_df.fillna(0.0).to_numpy(dtype=np.float32)
                    if array.shape != expected_shape:
                        logger.warning(f"Array df_{col} has incorrect shape: expected {expected_shape}, got {array.shape}. Initializing with zeros.")
                        array = np.zeros(expected_shape, dtype=np.float32)
                self.data_arrays[col] = array
                setattr(self, f'df_{col}', array)
        except Exception as e:
            logger.error(f"Error initializing data arrays: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize data arrays: {e}")
        # Initialize sentiment data
        self.df_news_sentiment = np.zeros((len(self.dates) if self.dates else 1, len(self.tickers)), dtype=np.float32)
        dates_to_use = pd.to_datetime(self.dates) if self.dates else [datetime.now()]
        for idx, ticker in enumerate(self.tickers):
            sentiment_df = self.sentiment_dfs.get(ticker)
            logger.debug(f"Processing sentiment data for ticker: {ticker}")
            if sentiment_df is None or not isinstance(sentiment_df, pd.DataFrame):
                logger.warning(f"Sentiment data for ticker {ticker} is None or not a DataFrame. Creating fallback.")
                dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B') if self.start_date and self.end_date else [datetime.now()]
                sentiment_df = pd.DataFrame({"date": dates, "sentiment": [0.0] * len(dates)})
            elif 'date' not in sentiment_df.columns:
                logger.warning(f"Sentiment DataFrame for ticker {ticker} lacks 'date' column. Columns: {sentiment_df.columns.tolist()}. Creating fallback.")
                dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B') if self.start_date and self.end_date else [datetime.now()]
                sentiment_df = pd.DataFrame({"date": dates, "sentiment": [0.0] * len(dates)})
            elif 'sentiment' not in sentiment_df.columns:
                logger.warning(f"Sentiment DataFrame for ticker {ticker} lacks 'sentiment' column. Columns: {sentiment_df.columns.tolist()}. Adding default sentiment.")
                sentiment_df['sentiment'] = 0.0
            try:
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                sentiment_df.set_index('date', inplace=True)
                sentiment_df = sentiment_df.reindex(dates_to_use).ffill().bfill()
                self.df_news_sentiment[:, idx] = sentiment_df['sentiment'].to_numpy()
                logger.debug(f"Successfully processed sentiment data for ticker {ticker}")
            except Exception as e:
                logger.error(f"Failed to process sentiment data for {ticker}: {e}")
                self.df_news_sentiment[:, idx] = 0.0
        if np.any(np.isnan(self.df_news_sentiment)):
            logger.warning(f"NaN values found in df_news_sentiment: {np.sum(np.isnan(self.df_news_sentiment))} NaNs")
            self.df_news_sentiment = np.nan_to_num(self.df_news_sentiment, 0.0)
        # Initialize VIX and S&P 500 data
        self.df_vix = np.full(len(self.dates) if self.dates else 1, 20.0, dtype=np.float32)
        self.df_sp500_return = np.zeros(len(self.dates) if self.dates else 1, dtype=np.float32)
        self.df_sp500_momentum = np.zeros(len(self.dates) if self.dates else 1, dtype=np.float32)
        if self.vix_df is not None and not self.vix_df.empty:
            vix_df_dates = pd.to_datetime(self.vix_df['date']).dt.tz_localize(None)
            dates_to_use = pd.to_datetime(self.dates) if self.dates else [datetime.now()]
            for i, date in enumerate(dates_to_use):
                mask = vix_df_dates == date
                if mask.any():
                    self.df_vix[i] = self.vix_df.loc[mask, 'vix'].iloc[0]
                    self.df_sp500_return[i] = self.vix_df.loc[mask, 'sp500_return'].iloc[0]
                    self.df_sp500_momentum[i] = self.vix_df.loc[mask, 'sp500_momentum'].iloc[0]
        logger.info(f"Initialized df_vix with shape: {self.df_vix.shape}")
        logger.info(f"Initialized df_sp500_return with shape: {self.df_sp500_return.shape}")
        logger.info(f"Initialized df_sp500_momentum with shape: {self.df_sp500_momentum.shape}")
        # Initialize trading parameters
        self.initial_balance = float(INITIAL_BALANCE)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = np.zeros(self.num_tickers, dtype=np.float32)
        self.positions = [{"total_cost": 0.0, "total_shares": 0.0, "avg_entry_price": 0.0} for _ in range(self.num_tickers)]
        self.trade_count = 0
        self.step_trades = 0
        self.max_trades_per_episode = int(MAX_TRADES_PER_EPISODE)
        self.max_total_trades = int(MAX_TOTAL_TRADES)
        self.max_trades_per_step = int(MAX_TRADES_PER_STEP)
        self.trade_cooldown = int(TRADE_COOLDOWN)
        self.slippage_base = float(SLIPPAGE_BASE)
        self.min_shares = float(MIN_SHARES)
        self.stop_loss_pct = float(STOP_LOSS_PCT)
        self.take_profit_pct = float(TAKE_PROFIT_PCT)
        self.trailing_stop_pct = float(TRAILING_STOP_PCT)
        self.max_position = float(MAX_POSITION)
        self.max_position_exposure = float(MAX_POSITION_EXPOSURE)
        self.max_portfolio_exposure = float(MAX_PORTFOLIO_EXPOSURE)
        self.ticker_trade_limits = {ticker: int(MAX_DAILY_TRADES_PER_TICKER) for ticker in self.tickers}
        self.ticker_buy_counts = {ticker: 0 for ticker in self.tickers}
        self.ticker_sell_counts = {ticker: 0 for ticker in self.tickers}
        self.ticker_consecutive_buys = {ticker: 0 for ticker in self.tickers}
        self.last_trade_step = np.full(self.num_tickers, -self.trade_cooldown - 1, dtype=np.int32)
        self.steps_since_last_trade = np.zeros(self.num_tickers, dtype=np.int32)
        self.holding_steps = np.zeros(self.num_tickers, dtype=np.int32)
        self.trades = []
        self.recent_profits = deque(maxlen=5)
        self.daily_pnl = np.zeros(self.max_steps + 1, dtype=np.float32) if backtest_mode else []
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.trade_fraction_history = {ticker: [] for ticker in self.tickers}
        self.min_liquidity_threshold = float(MIN_LIQUIDITY_THRESHOLD)
        logger.info(f"TradingEnv initialized with {self.num_tickers} tickers: {self.tickers}")
        logger.info(f"Initial balance: ${self.initial_balance:.2f}")
        # Define action and observation spaces
        self.action_space = Box(low=-1, high=1, shape=(self.num_tickers,), dtype=np.float32)
        obs_shape = (
            2 +  # balance_ratio, net_worth_ratio
            self.num_tickers * 2 +  # shares_held_ratio, steps_since_trade_ratio
            self.num_tickers * 11 +  # close, rsi, macd, bb_width, atr, adx, volatility, momentum, volume_trend, sma_5, sma_20
            self.num_tickers * 6 +  # norm_prices, norm_volumes, norm_volatilities, norm_momentums, norm_volume_trends, norm_news_sentiments
            self.num_tickers * 2 +  # holding_steps_ratio, recent_profits_ratio
            2 +  # vix_value, sp500_return
            2  # sp500_momentum, market_open
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        try:
            if seed is not None:
                np.random.seed(seed)
            self.current_step = 0
            self.trade_count = 0
            self.step_trades = 0
            self.last_trade_step = np.full(self.num_tickers, -self.trade_cooldown - 1, dtype=np.int32)
            self.steps_since_last_trade = np.zeros(self.num_tickers, dtype=np.int32)
            self.holding_steps = np.zeros(self.num_tickers, dtype=np.int32)
            self.trades = []
            self.recent_profits = deque(maxlen=5)
            self.daily_pnl = np.zeros(self.max_steps + 1, dtype=np.float32) if self.backtest_mode else []
            if self.backtest_mode:
                self.daily_pnl[0] = 0.0
                logger.debug(f"Reset - max_steps: {self.max_steps}, daily_pnl shape: {self.daily_pnl.shape}")
            self.realized_pnl = 0.0
            self.unrealized_pnl = 0.0
            self.trade_fraction_history = {ticker: [] for ticker in self.tickers}
            self.ticker_buy_counts = {ticker: 0 for ticker in self.tickers}
            self.ticker_sell_counts = {ticker: 0 for ticker in self.tickers}
            self.ticker_consecutive_buys = {ticker: 0 for ticker in self.tickers}
            self.balance = self.initial_balance
            self.net_worth = self.initial_balance
            self.shares_held = np.zeros(self.num_tickers, dtype=np.float32)
            self.positions = [{"total_cost": 0.0, "total_shares": 0.0, "avg_entry_price": 0.0} for _ in range(self.num_tickers)]
            logger.debug(f"Reset trade counters: buy={self.ticker_buy_counts}, sell={self.ticker_sell_counts}, consecutive_buys={self.ticker_consecutive_buys}")
            obs = self._get_obs()
            info = {
                "reset_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "step": self.current_step,
                "balance": self.balance,
                "net_worth": self.net_worth,
                "tickers": self.tickers,
                "max_steps": self.max_steps
            }
            logger.debug(f"Reset successful, observation shape: {obs.shape}, info: {info}")
            return obs, info
        except Exception as e:
            logger.error(f"Error in reset method: {e}", exc_info=True)
            logger.debug(f"Input data - df shape: {self.df.shape if not self.df.empty else 'empty'}, "
                        f"top_stocks_df columns: {self.top_stocks_df.columns.tolist() if not self.top_stocks_df.empty else 'empty'}, "
                        f"vix_df shape: {self.vix_df.shape if not self.vix_df.empty else 'empty'}, "
                        f"sentiment_dfs keys: {list(self.sentiment_dfs.keys()) if self.sentiment_dfs else 'empty'}")
            default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            default_info = {
                "error": str(e),
                "reset_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "step": 0,
                "balance": self.initial_balance,
                "net_worth": self.initial_balance,
                "tickers": self.tickers,
                "max_steps": self.max_steps
            }
            return default_obs, default_info
    def _get_obs(self) -> np.ndarray:
        """Generate observation for the current step."""
        try:
            step_idx = self.current_step if self.current_step < self.max_steps else -1
            prices = self.data_arrays['close'][step_idx]
            volumes = self.data_arrays['volume'][step_idx]
            volatilities = self.data_arrays['volatility'][step_idx]
            momentums = self.data_arrays['momentum'][step_idx]
            volume_trends = self.data_arrays['volume_trend'][step_idx]
            news_sentiments = self.df_news_sentiment[step_idx]
            vix_value = self.df_vix[step_idx] if step_idx < len(self.df_vix) else self.df_vix[-1]
            sp500_return = self.df_sp500_return[step_idx] if step_idx < len(self.df_sp500_return) else self.df_sp500_return[-1]
            sp500_momentum = self.df_sp500_momentum[step_idx] if step_idx < len(self.df_sp500_momentum) else self.df_sp500_momentum[-1]
            # Normalize arrays efficiently
            max_prices = np.max(prices)
            norm_prices = prices / max_prices if max_prices != 0 else prices
            max_volumes = np.max(volumes)
            norm_volumes = volumes / max_volumes if max_volumes != 0 else volumes
            max_volatilities = np.max(volatilities)
            norm_volatilities = volatilities / max_volatilities if max_volatilities != 0 else volatilities
            max_abs_momentums = np.max(np.abs(momentums))
            norm_momentums = momentums / max_abs_momentums if max_abs_momentums != 0 else momentums
            max_abs_volume_trends = np.max(np.abs(volume_trends))
            norm_volume_trends = volume_trends / max_abs_volume_trends if max_abs_volume_trends != 0 else volume_trends
            max_abs_news_sentiments = np.max(np.abs(news_sentiments))
            norm_news_sentiments = news_sentiments / max_abs_news_sentiments if max_abs_news_sentiments != 0 else news_sentiments
            # Handle recent profits
            recent_profits_array = np.array(list(self.recent_profits), dtype=np.float32)
            recent_profits_mean = np.mean(recent_profits_array) if recent_profits_array.size > 0 else 0.0
            recent_profits_per_ticker = np.full(self.num_tickers, recent_profits_mean, dtype=np.float32)
            # Calculate ratios
            balance_ratio = self.balance / self.initial_balance if self.initial_balance != 0 else 0.0
            net_worth_ratio = self.net_worth / self.initial_balance if self.initial_balance != 0 else 0.0
            shares_held_ratio = self.shares_held / self.max_position if self.max_position != 0 else self.shares_held
            steps_since_trade_ratio = self.steps_since_last_trade / (self.max_steps if self.backtest_mode else 120) if (self.max_steps if self.backtest_mode else 120) != 0 else self.steps_since_last_trade
            holding_steps_ratio = self.holding_steps / (self.max_steps if self.backtest_mode else 120) if (self.max_steps if self.backtest_mode else 120) != 0 else self.holding_steps
            recent_profits_ratio = recent_profits_per_ticker / (self.initial_balance if self.initial_balance != 0 else 1)
            balance_ratio = 0.0 if np.isnan(balance_ratio) else balance_ratio
            net_worth_ratio = 0.0 if np.isnan(net_worth_ratio) else net_worth_ratio
            shares_held_ratio = np.nan_to_num(shares_held_ratio, 0.0)
            steps_since_trade_ratio = np.nan_to_num(steps_since_trade_ratio, 0.0)
            holding_steps_ratio = np.nan_to_num(holding_steps_ratio, 0.0)
            recent_profits_ratio = np.nan_to_num(recent_profits_ratio, 0.0)
            # Fetch data from data_arrays
            data_arrays_step = {col: self.data_arrays[col][step_idx] for col in self.data_arrays}
            obs = np.concatenate([
                np.array([balance_ratio, net_worth_ratio], dtype=np.float32),
                shares_held_ratio,
                steps_since_trade_ratio,
                data_arrays_step['close'],
                data_arrays_step['rsi'],
                data_arrays_step['macd'],
                data_arrays_step['bb_width'],
                data_arrays_step['atr'],
                data_arrays_step['adx'],
                data_arrays_step['volatility'],
                data_arrays_step['momentum'],
                data_arrays_step['volume_trend'],
                data_arrays_step['sma_5'],
                data_arrays_step['sma_20'],
                norm_prices,
                norm_volumes,
                norm_volatilities,
                norm_momentums,
                norm_volume_trends,
                norm_news_sentiments,
                np.array([vix_value / 100.0, sp500_return], dtype=np.float32),
                np.array([sp500_momentum, float(self.market_open)], dtype=np.float32),
                holding_steps_ratio,
                recent_profits_ratio
            ])
            obs = np.nan_to_num(obs, 0.0)
            logger.debug(f"Observation shape: {obs.shape}, step_idx: {step_idx}")
            return obs
        except Exception as e:
            logger.error(f"Error in _get_obs method: {e}", exc_info=True)
            logger.debug(f"_get_obs failure details - step_idx: {step_idx}, "
             f"data_arrays keys: {list(self.data_arrays.keys())}, "
             f"df_news_sentiment shape: {self.df_news_sentiment.shape}, "
             f"df_vix shape: {self.df_vix.shape}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    def _calc_fees(self, shares: float, price: float, action: str) -> float:
        try:
            trade_value = shares * price
            fees = trade_value * float(TRANSACTION_COST)
            logger.debug(f"Calculated fees for {action}: shares={shares}, price={price}, trade_value={trade_value:.2f}, fees={fees:.2f}")
            return fees
        except Exception as e:
            logger.error(f"Error calculating fees: {e}")
            return 0.0
    def _save_trade_to_db(self, trade: Dict) -> None:
        if DATABASE_ENABLED:
            try:
                with sqlite3.connect(DATABASE_PATH) as conn:
                    trade_df = pd.DataFrame([trade])
                    trade_df.to_sql('trades', conn, if_exists='append', index=False)
                    conn.commit()
                logger.debug(f"Saved trade to database: {trade}")
            except Exception as e:
                logger.error(f"Failed to save trade to database: {e}")
    async def _async_live_trade(self, ticker: str, shares: float, action: str) -> None:
        try:
            ib = IB()
            await ib.connectAsync(IBKR_HOST, int(IBKR_PORT), clientId=int(IBKR_CLIENT_ID) + secrets.randbelow(1000))
            contract = Stock(ticker, 'SMART', 'USD')
            order = util.order.LimitOrder(action, shares, self.data_arrays['close'][self.current_step][self.tickers.index(ticker)])
            trade = ib.placeOrder(contract, order)
            await asyncio.sleep(1)
            logger.info(f"Live {action} trade executed for {ticker}: {shares} shares")
        except Exception as e:
            logger.error(f"Failed to execute live trade for {ticker}: {e}")
        finally:
            ib.disconnect()
    def _execute_live_trade(self, ticker_idx: int, shares: float, action: str) -> None:
        if not LIVE_TRADING_ENABLED or not self.market_open:
            logger.debug(f"Live trading skipped: LIVE_TRADING_ENABLED={LIVE_TRADING_ENABLED}, market_open={self.market_open}")
            return
        ticker = self.tickers[ticker_idx]
        try:
            asyncio.run(self._async_live_trade(ticker, shares, action))
        except RuntimeError as e:
            logger.error(f"Event loop error for {ticker}: {e}. Scheduling task.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._async_live_trade(ticker, shares, action))
            finally:
                loop.close()
    def _execute_trade(
        self,
        ticker_idx: int,
        shares: float,
        action: str,
        confidence: float = 1.0,
        forced: bool = False,
        stop_loss: bool = False,
        take_profit: bool = False,
        trailing_stop: bool = False
    ) -> None:
        try:
            price = self.data_arrays['close'][self.current_step][ticker_idx] if self.current_step < len(self.data_arrays['close']) else self.data_arrays['close'][-1][ticker_idx]
            slippage = np.random.uniform(-self.slippage_base, self.slippage_base)
            adjusted_price = price * (1 + slippage)
            ticker = self.tickers[ticker_idx]
            if self.dynamic_position_sizing:
                confidence_factor = min(max(confidence, 0.1), 1.0)
                shares = shares * confidence_factor
                shares = int(max(self.min_shares, shares))
            if action == "BUY":
                fees = self._calc_fees(shares, adjusted_price, "BUY")
                cost = shares * adjusted_price + fees
                portfolio_exposure = (self.net_worth - cost) * self.max_portfolio_exposure
                current_exposure = sum([self.shares_held[i] * self.data_arrays['close'][self.current_step][i] for i in range(self.num_tickers)])
                if current_exposure + cost > portfolio_exposure:
                    shares = (portfolio_exposure - current_exposure) / (adjusted_price + fees / shares) if (adjusted_price + fees / shares) > 0 else 0
                    shares = int(max(0, shares))
                    cost = shares * adjusted_price + fees
                if shares <= 0 or cost > self.balance:
                    logger.debug(f"Skipped BUY for {ticker}: shares={shares}, cost={cost:.2f}, balance={self.balance:.2f}")
                    return
                self.balance -= cost
                self.shares_held[ticker_idx] += shares
                self.positions[ticker_idx]["total_cost"] += cost
                self.positions[ticker_idx]["total_shares"] += shares
                if self.positions[ticker_idx]["total_shares"] == shares:
                    self.positions[ticker_idx]["avg_entry_price"] = adjusted_price
                else:
                    self.positions[ticker_idx]["avg_entry_price"] = self.positions[ticker_idx]["total_cost"] / self.positions[ticker_idx]["total_shares"]
                self.trade_count += 1
                self.step_trades += 1
                self.ticker_buy_counts[ticker] += 1
                self.ticker_consecutive_buys[ticker] += 1
                self.last_trade_step[ticker_idx] = self.current_step
                self.steps_since_last_trade[ticker_idx] = 0
                trade = {
                    "ticker": ticker,
                    "action": "BUY",
                    "shares": float(shares),
                    "price": float(adjusted_price),
                    "cost": float(cost),
                    "fees": float(fees),
                    "date": str(pd.Timestamp(self.dates[self.current_step])) if self.current_step < len(self.dates) else datetime.now().strftime('%Y-%m-%d'),
                    "stop_loss": False,
                    "take_profit": False,
                    "trailing_stop": False,
                    "forced": forced
                }
                self.trades.append(trade)
                self._save_trade_to_db(trade)
                self.trade_fraction_history[ticker].append(shares / self.max_position if self.max_position > 0 else 0.0)
                logger.info(f"BUY of {shares:.2f} shares of {ticker} at ${adjusted_price:.2f} (Market open: {self.market_open})")
                if LIVE_TRADING_ENABLED:
                    self._execute_live_trade(ticker_idx, shares, "BUY")
            else:
                if self.shares_held[ticker_idx] < shares:
                    shares = self.shares_held[ticker_idx]
                if shares <= 0:
                    logger.debug(f"Skipped SELL for {ticker}: insufficient shares ({self.shares_held[ticker_idx]})")
                    return
                fees = self._calc_fees(shares, adjusted_price, "SELL")
                proceeds = shares * adjusted_price - fees
                avg_entry_price = self.positions[ticker_idx]["avg_entry_price"]
                cost_basis = shares * avg_entry_price
                trade_pnl = proceeds - cost_basis
                self.recent_profits.append(trade_pnl)
                self.realized_pnl += trade_pnl
                self.balance += proceeds
                self.shares_held[ticker_idx] -= shares
                self.positions[ticker_idx]["total_shares"] -= shares
                self.positions[ticker_idx]["total_cost"] -= cost_basis
                if self.positions[ticker_idx]["total_shares"] <= 0:
                    self.positions[ticker_idx]["total_shares"] = 0.0
                    self.positions[ticker_idx]["total_cost"] = 0.0
                    self.positions[ticker_idx]["avg_entry_price"] = 0.0
                else:
                    self.positions[ticker_idx]["avg_entry_price"] = self.positions[ticker_idx]["total_cost"] / self.positions[ticker_idx]["total_shares"]
                self.trade_count += 1
                self.step_trades += 1
                self.ticker_sell_counts[ticker] += 1
                self.ticker_consecutive_buys[ticker] = 0
                self.last_trade_step[ticker_idx] = self.current_step
                self.steps_since_last_trade[ticker_idx] = 0
                trade = {
                    "ticker": ticker,
                    "action": "SELL",
                    "shares": float(shares),
                    "price": float(adjusted_price),
                    "proceeds": float(proceeds),
                    "pnl": float(trade_pnl),
                    "fees": float(fees),
                    "date": str(pd.Timestamp(self.dates[self.current_step])) if self.current_step < len(self.dates) else datetime.now().strftime('%Y-%m-%d'),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "trailing_stop": trailing_stop,
                    "forced": forced
                }
                self.trades.append(trade)
                self._save_trade_to_db(trade)
                self.trade_fraction_history[ticker].append(-shares / self.max_position if self.max_position > 0 else 0.0)
                logger.info(f"SELL of {shares:.2f} shares of {ticker} at ${adjusted_price:.2f}, PnL: {trade_pnl:.2f}")
                if LIVE_TRADING_ENABLED:
                    self._execute_live_trade(ticker_idx, shares, "SELL")
        except Exception as e:
            logger.error(f"Error executing trade for {ticker}: {e}")

    def _apply_rule_based_filter(self, action, confidence, sentiments, vix_value, volatilities, news_sentiments, momentums):
        filtered_action = action.copy()
        for i in range(self.num_tickers):
            ticker = self.tickers[i]
            if momentums[i] < -0.03:
                filtered_action[i] = min(filtered_action[i], 0.0)
                logger.debug(f"[{ticker}] Filtered action due to strong downtrend: momentum={momentums[i]:.2f}")
            if vix_value > float(VIX_THRESHOLD) and sentiments[i] + news_sentiments[i] < 0.2:
                filtered_action[i] = 0.0
                logger.debug(f"[{ticker}] Filtered action to 0: High VIX ({vix_value:.2f}) and low combined sentiment ({sentiments[i] + news_sentiments[i]:.2f})")
            if volatilities[i] > 0.8 and confidence[i] < 0.5:
                filtered_action[i] = 0.0
                logger.debug(f"[{ticker}] Filtered action to 0: High volatility ({volatilities[i]:.2f}) and low confidence ({confidence[i]:.2f})")
            if filtered_action[i] > 0 and (sentiments[i] + news_sentiments[i]) < -0.5:
                filtered_action[i] = 0.0
                logger.debug(f"[{ticker}] Filtered action to 0: Positive action but low combined sentiment ({sentiments[i] + news_sentiments[i]:.2f})")
        return filtered_action
    def step(self, action):
        """Take a step in the environment based on action."""
        try:
            self.step_trades = 0
            self.trades = []
            terminated = False
            truncated = False
            total_proceeds = 0.0
            daily_pnl_list = []
            logger.debug(f"Step {self.current_step}: balance={self.balance:.2f}, net_worth={self.net_worth:.2f}, shares_held={self.shares_held}")
            if self.trade_count >= self.max_total_trades:
                logger.info("Reached maximum total trades. Terminating episode early.")
                terminated = True
            old_net_worth = self.net_worth
            self.current_step = min(self.current_step + 1, self.max_steps - 1) if self.backtest_mode else self.current_step + 1
            step_idx = self.current_step if self.backtest_mode else 0
            terminated = terminated or (self.backtest_mode and (self.current_step >= self.max_steps)) or (self.trade_count >= self.max_trades_per_episode)
            truncated = False
            self.steps_since_last_trade = self.current_step - self.last_trade_step
            self.holding_steps[self.shares_held > 0] += 1
            self.holding_steps[self.shares_held == 0] = 0
            logger.debug(f"Step {self.current_step}: Trade counts - Buy: {self.ticker_buy_counts}, Sell: {self.ticker_sell_counts}")
            if terminated:
                total_proceeds = 0
                prices = self.data_arrays['close'][step_idx] if self.backtest_mode else self.data_arrays['close'][-1]
                for j in range(self.num_tickers):
                    if self.trade_count >= self.max_total_trades:
                        break
                    if self.shares_held[j] >= self.min_shares:
                        shares_filled = int(self.shares_held[j] / self.min_shares) * self.min_shares
                        avg_volume = self.data_arrays['avg_volume'][step_idx][j] if self.backtest_mode else self.data_arrays['avg_volume'][-1][j]
                        slippage_adjustment = max(self.slippage_base * 0.5, 0.005 / np.sqrt(avg_volume / 1000000))
                        slip_price = prices[j] * (1 - slippage_adjustment)
                        fees = self._calc_fees(shares_filled, prices[j], "SELL")
                        proceeds = shares_filled * slip_price - fees
                        self.realized_pnl += proceeds - (shares_filled * self.positions[j]['avg_entry_price'])
                        total_proceeds += proceeds
                        self.shares_held[j] -= shares_filled
                        self._execute_live_trade(j, shares_filled, "SELL")
                        trade = {
                            "date": str(pd.Timestamp(self.dates[step_idx])) if self.backtest_mode else datetime.now().strftime('%Y-%m-%d'),
                            "ticker": self.tickers[j],
                            "action": "SELL",
                            "shares": float(shares_filled),
                            "price": float(prices[j]),
                            "proceeds": float(proceeds),
                            "fees": float(fees),
                            "pnl": float(proceeds - (shares_filled * self.positions[j]['avg_entry_price'])),
                            "stop_loss": False,
                            "take_profit": False,
                            "trailing_stop": False,
                            "forced": True
                        }
                        self.trades.append(trade)
                        self._save_trade_to_db(trade)
                        self.trade_count += 1
                        self.last_trade_step[j] = self.current_step
                        self.steps_since_last_trade[j] = 0
                        self.positions[j] = {"total_cost": 0.0, "total_shares": 0.0, "avg_entry_price": 0.0}
                        self.ticker_sell_counts[self.tickers[j]] += 1
                        self.ticker_consecutive_buys[self.tickers[j]] = 0
                        logger.debug(f"Final SELL sweep for {self.tickers[j]}: shares_filled = {shares_filled}, price={prices[j]:.2f}, proceeds={proceeds:.2f}")
                self.balance += total_proceeds
                self.net_worth = self.balance + np.sum(self.shares_held * prices)
                reward = (self.net_worth - self.initial_balance) * 0.01 - (self.trade_count / self.max_trades_per_episode) * 10
                logger.info(f"Episode ended at step {self.current_step}: net_worth={self.net_worth:.2f}, balance={self.balance:.2f}, trades={self.trade_count}")
                daily_pnl_list = self.daily_pnl[:self.current_step + 1].tolist() if self.backtest_mode else self.daily_pnl
                if not daily_pnl_list:
                    daily_pnl_list = [0.0]
                return self._get_obs(), reward, terminated, truncated, {"trades": self.trades, "daily_pnl": daily_pnl_list}
            if not terminated:
                confidence = np.abs(action)
                sentiments = self.df_news_sentiment[step_idx]
                volatilities = self.data_arrays['volatility'][step_idx]
                vix_value = self.df_vix[step_idx] if step_idx < len(self.df_vix) else self.df_vix[-1]
                news_sentiments = self.df_news_sentiment[step_idx]
                momentums = self.data_arrays['momentum'][step_idx]
                filtered_action = self._apply_rule_based_filter(action, confidence, sentiments, vix_value, volatilities, news_sentiments, momentums)
                prices = self.data_arrays['close'][step_idx]
                volumes = self.data_arrays['volume'][step_idx]
                avg_volumes = self.data_arrays['avg_volume'][step_idx]
                momentums = self.data_arrays['momentum'][step_idx]
                atrs = self.data_arrays['atr'][step_idx]
                adxs = self.data_arrays['adx'][step_idx]
                sp500_return = self.df_sp500_return[step_idx] if step_idx < len(self.df_sp500_return) else self.df_sp500_return[-1]
                sp500_momentum = self.df_sp500_momentum[step_idx] if step_idx < len(self.df_sp500_momentum) else self.df_sp500_momentum[-1]
                valid_mask = (
                    (prices > 0) &
                    (volumes > 0) &
                    (avg_volumes >= self.min_liquidity_threshold) &
                    (np.abs(filtered_action) >= 0.000001) &
                    (self.steps_since_last_trade >= self.trade_cooldown)
                )
                trade_candidates = np.zeros((self.num_tickers, 3), dtype=np.float32)
                portfolio_value = self.balance + np.sum(self.shares_held * prices)
                portfolio_volatility = np.mean(volatilities) if len(volatilities) > 0 else 0.0
                risk_factor = 1 / (1 + portfolio_volatility * 1.5) if portfolio_volatility != 0 else 1.0
                final_scores = np.zeros(self.num_tickers, dtype=np.float32)
                for i in range(self.num_tickers):
                    final_score = 0.0
                    if valid_mask[i]:
                        sentiment_score = sentiments[i] * float(SENTIMENT_WEIGHT_X) * 20
                        news_sentiment_score = news_sentiments[i] * float(SENTIMENT_WEIGHT_NEWS) * 20
                        ratings_score = self.top_stocks_df[self.top_stocks_df['ticker'] == self.tickers[i]]['final_score'].iloc[0] if not self.top_stocks_df.empty else 0.0
                        final_score = sentiment_score + news_sentiment_score + ratings_score
                    final_scores[i] = final_score
                    trade_candidates[i] = [i, filtered_action[i], final_score]
                sentiments_array = np.zeros(self.num_tickers, dtype=np.float32)
                for i, ticker in enumerate(self.tickers):
                    if ticker in self.top_stocks_df['ticker'].values:
                        sentiments_array[i] = self.top_stocks_df.loc[self.top_stocks_df['ticker'] == ticker, 'sentiment'].iloc[0]
                confidence = np.abs(action) * (1 + momentums) * (1 + np.abs(sentiments_array) + 2 * final_scores + np.abs(news_sentiments))
                trade_candidates[:, 0] = np.arange(self.num_tickers)
                trade_candidates[:, 1] = filtered_action
                trade_candidates[:, 2] = filtered_action * confidence * risk_factor / (1 + volatilities * 0.03)
                for i in range(self.num_tickers):
                    ticker = self.tickers[i]
                    ticker_trade_count = len([t for t in self.trades if t['ticker'] == ticker])
                    trade_candidates[i, 2] *= (1 - ticker_trade_count / 10)
                for ticker_idx, ticker in enumerate(self.tickers):
                    total_trades_for_ticker = self.ticker_buy_counts[ticker] + self.ticker_sell_counts[ticker]
                    if total_trades_for_ticker >= self.ticker_trade_limits[ticker]:
                        trade_candidates[ticker_idx, 2] = -np.inf
                peak_prices = np.max(self.data_arrays['close'][:self.current_step + 1], axis=0) if self.backtest_mode else self.data_arrays['close']
                avg_entry_prices = np.array([pos['avg_entry_price'] for pos in self.positions])
                price_diff = prices - avg_entry_prices
                valid_entries = avg_entry_prices > 0
                unrealized_gains = np.zeros_like(prices)
                unrealized_gains[valid_entries] = price_diff[valid_entries] / avg_entry_prices[valid_entries]
                sentiment_adjustment = np.clip(sentiments_array + news_sentiments, -0.5, 0.5)
                dynamic_stop_loss = self.stop_loss_pct * (1 + atrs / prices * float(ATR_MULTIPLIER_STOP_LOSS)) if bool(ADAPTIVE_STOP_LOSS) else self.stop_loss_pct
                dynamic_take_profit = self.take_profit_pct * (1 + np.abs(momentums) + sentiment_adjustment * float(ATR_MULTIPLIER_TAKE_PROFIT)) if bool(ADAPTIVE_TAKE_PROFIT) else self.take_profit_pct
                stop_mask = (self.shares_held > 0) & (prices < peak_prices * (1 - self.trailing_stop_pct - atrs / prices))
                profit_mask = (self.shares_held > 0) & (valid_entries) & (unrealized_gains >= dynamic_take_profit)
                loss_mask = (self.shares_held > 0) & (valid_entries) & (unrealized_gains <= -dynamic_stop_loss)
                position_limit_mask = (self.shares_held >= self.max_position * 0.9)
                holding_too_long_mask = (self.shares_held > 0) & (self.holding_steps >= 20)
                trade_candidates[stop_mask | profit_mask | loss_mask | position_limit_mask | holding_too_long_mask, 1] = -1.0
                for i in range(self.num_tickers):
                    if self.ticker_consecutive_buys[self.tickers[i]] >= int(MAX_CONSECUTIVE_BUYS):
                        logger.info(f"[{self.tickers[i]}] Forcing a SELL trade due to {MAX_CONSECUTIVE_BUYS} consecutive BUYs...")
                        shares_to_sell = self.shares_held[i]
                        if shares_to_sell > 0:
                            self._execute_trade(i, shares_to_sell, "SELL", forced=True)
                            self.ticker_consecutive_buys[self.tickers[i]] = 0
                for i in range(self.num_tickers):
                    if position_limit_mask[i] and self.shares_held[i] >= self.min_shares:
                        ticker = self.tickers[i]
                        total_trades_for_ticker = self.ticker_buy_counts[ticker] + self.ticker_sell_counts[ticker]
                        if total_trades_for_ticker >= self.ticker_trade_limits[ticker]:
                            logger.info(f"[{ticker}] Skipping forced SELL due to position limit: Reached total trade limit of {self.ticker_trade_limits[ticker]} trades")
                            continue
                        price = prices[i]
                        volume = volumes[i]
                        avg_volume = avg_volumes[i]
                        shares_to_sell = self.shares_held[i]
                        shares_filled = min(shares_to_sell, volume * float(MAX_FILL_PCT))
                        slippage_adjustment = max(self.slippage_base * (volatilities[i] / np.mean(volatilities)) * 0.5, 0.005 / np.sqrt(avg_volume / 1000000))
                        fees = self._calc_fees(shares_filled, price, "SELL")
                        proceeds = shares_filled * price * (1 - slippage_adjustment) - fees
                        self.trade_count += 1
                        self.realized_pnl += proceeds - (shares_filled * self.positions[i]['avg_entry_price'])
                        self.shares_held[i] -= shares_filled
                        self.positions[i]['total_cost'] -= shares_filled * self.positions[i]['avg_entry_price']
                        self.positions[i]['total_shares'] -= shares_filled
                        self.positions[i]['avg_entry_price'] = 0.0 if self.positions[i]['total_shares'] <= 0 else self.positions[i]['total_cost'] / self.positions[i]['total_shares']
                        self.ticker_sell_counts[ticker] += 1
                        self.ticker_consecutive_buys[ticker] = 0
                        trade = {
                            "ticker": ticker,
                            "action": "SELL",
                            "shares": float(shares_filled),
                            "price": float(price),
                            "proceeds": float(proceeds),
                            "pnl": float(proceeds - (shares_filled * self.positions[i]['avg_entry_price'])),
                            "fees": float(fees),
                            "date": str(pd.Timestamp(self.dates[step_idx])) if self.backtest_mode else datetime.now().strftime('%Y-%m-%d'),
                            "forced": True,
                            "stop_loss": False,
                            "take_profit": False,
                            "trailing_stop": False
                        }
                        self.trades.append(trade)
                        self._save_trade_to_db(trade)
                        logger.debug(f"Forced SELL for {ticker}: shares_filled={shares_filled}, price={price:.2f}, proceeds={proceeds:.2f}")
                valid_candidates = trade_candidates[valid_mask | stop_mask | profit_mask | loss_mask | position_limit_mask | holding_too_long_mask]
                if valid_candidates.size == 0:
                    pass
                else:
                    if len(valid_candidates.shape) == 1:
                        valid_candidates = valid_candidates.reshape(1, -1)
                    sorted_indices = np.argsort(-np.abs(valid_candidates[:, 2]))
                    valid_candidates = valid_candidates[sorted_indices]
                for candidate in valid_candidates:
                    i = int(candidate[0])
                    action_value = candidate[1]
                    ticker = self.tickers[i]
                    if action_value > 0:
                        position_value = self.net_worth * self.max_position_exposure
                        shares_to_buy = min(self.max_position, position_value / prices[i]) if prices[i] > 0 else 0
                        shares_to_buy *= action_value
                        shares_to_buy = max(self.min_shares, shares_to_buy)
                        volume_limit = volumes[i] * float(MAX_FILL_PCT)
                        shares_filled = min(shares_to_buy, volume_limit)
                        shares_filled = int(shares_filled)
                        if shares_filled > 0 and self.step_trades < self.max_trades_per_step and self.trade_count < self.max_trades_per_episode and self.ticker_buy_counts[ticker] < self.ticker_trade_limits[ticker]:
                            self._execute_trade(i, shares_filled, "BUY", confidence=confidence[i])
                            self.ticker_consecutive_buys[ticker] += 1
                    elif action_value < 0:
                        shares_to_sell = self.shares_held[i] * abs(action_value)
                        shares_to_sell = max(self.min_shares, shares_to_sell)
                        volume_limit = volumes[i] * float(MAX_FILL_PCT)
                        shares_filled = min(shares_to_sell, volume_limit)
                        shares_filled = int(shares_filled)
                        if shares_filled > 0 and self.step_trades < self.max_trades_per_step and self.trade_count < self.max_trades_per_episode and self.ticker_sell_counts[ticker] < self.ticker_trade_limits[ticker]:
                            if stop_mask[i]:
                                self._execute_trade(i, shares_filled, "SELL", confidence=confidence[i], stop_loss=True)
                            elif profit_mask[i]:
                                self._execute_trade(i, shares_filled, "SELL", confidence=confidence[i], take_profit=True)
                            elif loss_mask[i]:
                                self._execute_trade(i, shares_filled, "SELL", confidence=confidence[i], stop_loss=True)
                            elif holding_too_long_mask[i]:
                                self._execute_trade(i, shares_filled, "SELL", confidence=confidence[i], forced=True)
                            else:
                                self._execute_trade(i, shares_filled, "SELL", confidence=confidence[i])
                            self.ticker_consecutive_buys[ticker] = 0
                total_proceeds = sum(trade["proceeds"] for trade in self.trades if trade["action"] == "SELL")
                portfolio_value = 0.0
                self.unrealized_pnl = 0.0
                for i in range(self.num_tickers):
                    if self.shares_held[i] > 0:
                        current_value = self.shares_held[i] * prices[i]
                        portfolio_value += current_value
                        cost_basis = self.positions[i]["avg_entry_price"] * self.shares_held[i]
                        position_unrealized_pnl = current_value - cost_basis
                        self.unrealized_pnl += position_unrealized_pnl
                self.net_worth = self.balance + portfolio_value
                drawdown = (self.initial_balance - self.net_worth) / self.initial_balance if self.initial_balance != 0 else 0.0
                if drawdown > float(MAX_DRAWDOWN_PCT) and not self.training_mode:
                    logger.warning(f"Max drawdown exceeded ({drawdown*100:.2f}% > {MAX_DRAWDOWN_PCT*100:.2f}%). Terminating episode.")
                    terminated = True
                base_reward = (self.net_worth - self.initial_balance) / self.initial_balance if self.initial_balance != 0 else 0.0
                holding_penalty = -0.0005 * np.sum(self.holding_steps) / (self.max_steps if self.backtest_mode else 120) if np.sum(self.holding_steps) > 0 else 0.0
                losing_position_penalty = -0.005 * (self.unrealized_pnl / self.initial_balance) if self.unrealized_pnl < 0 else 0.0
                sentiment_bonus = np.mean(sentiments) * 0.5
                unrealized_reward = 0.1 * (self.unrealized_pnl / self.initial_balance) if self.unrealized_pnl > 0 else 0.0
                trade_frequency_penalty = -0.001 * self.step_trades if self.step_trades > self.max_trades_per_step / 2 else 0.0
                trade_bonus = 0.1 * len(self.trades) if len(self.trades) > 0 else 0.0
                diversification_bonus = 0.02 * len(np.unique([trade["ticker"] for trade in self.trades])) if self.trades else 0.0
                volatility_penalty = -0.005 * np.mean(volatilities) if np.mean(volatilities) > 0.3 else 0.0
                vix_exposure_penalty = -0.002 * vix_value / 100 if vix_value > float(VIX_THRESHOLD) else 0.0
                concentration_penalty = -0.01 * max(len([t for t in self.trades if t['ticker'] == ticker]) / len(self.trades) for ticker in self.tickers) if self.trades else 0.0
                total_reward = (
                    base_reward * 3.0 +
                    holding_penalty +
                    losing_position_penalty +
                    sentiment_bonus +
                    unrealized_reward +
                    trade_frequency_penalty +
                    trade_bonus +
                    diversification_bonus +
                    volatility_penalty +
                    vix_exposure_penalty +
                    concentration_penalty
                )
                daily_pnl_value = self.net_worth - old_net_worth
                if self.backtest_mode:
                    self.daily_pnl[step_idx] = daily_pnl_value
                    daily_pnl_list = self.daily_pnl[:step_idx + 1].tolist()
                else:
                    self.daily_pnl.append(daily_pnl_value)
                    daily_pnl_list = self.daily_pnl
                for i in range(self.num_tickers):
                    self.holding_steps[i] += 1 if self.shares_held[i] > 0 else 0
                    self.steps_since_last_trade[i] += 1
                logger.debug(f"End of step {self.current_step}: balance={self.balance:.2f}, net_worth={self.net_worth:.2f}, reward={total_reward:.4f}")
                return self._get_obs(), total_reward, terminated, truncated, {"trades": self.trades, "daily_pnl": daily_pnl_list}
        except Exception as e:
            logger.error(f"Error in step method: {e}", exc_info=True)
            return self._get_obs(), 0.0, True, False, {"trades": [], "daily_pnl": []}
    def get_trade_counts(self) -> Dict[str, Dict[str, int]]:
        return {
            "ticker_buy_counts": self.ticker_buy_counts.copy(),
            "ticker_sell_counts": self.ticker_sell_counts.copy(),
            "ticker_consecutive_buys": self.ticker_consecutive_buys.copy()
        }
    def set_trade_counts(self, trade_counts: Dict[str, Dict[str, int]]) -> None:
        self.ticker_buy_counts = trade_counts["ticker_buy_counts"].copy()
        self.ticker_sell_counts = trade_counts["ticker_sell_counts"].copy()
        self.ticker_consecutive_buys = trade_counts["ticker_consecutive_buys"].copy()
    def render(self, mode: str = 'human') -> None:
        logger.info(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, Net Worth: ${self.net_worth:.2f}")
        logger.info(f"Shares Held: {self.shares_held}")
        logger.info(f"Recent Trades: {self.trades[-5:]}")
    def close(self) -> None:
        if DATABASE_ENABLED:
            try:
                with sqlite3.connect(DATABASE_PATH) as conn:
                    conn.commit()
                logger.info("Database connection closed.")
            except sqlite3.Error as e:
                logger.error(f"Failed to close database connection: {e}")
@lru_cache(maxsize=128)
async def fetch_market_data(tickers: Tuple[str, ...], start_date: str, end_date: str, api_key: str = POLYGON_API_KEY, use_alpha_vantage: bool = False) -> pd.DataFrame:
    """Fetch market data for specified tickers and date range."""
    all_data = []
    calls_per_minute = int(POLYGON_API_RATE_LIMIT) if 'POLYGON_API_RATE_LIMIT' in globals() else 5
    period = 60
    current_date = datetime.now().strftime('%Y-%m-%d')
    if start_date > current_date or end_date > current_date:
        logger.warning(f"Future date range detected ({start_date} to {end_date}). Returning simulated data.")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        simulated_data = []
        for ticker in tickers:
            for date in dates:
                simulated_data.append({
                    'ticker': ticker,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': 100.0 + np.random.uniform(-5, 5),
                    'high': 105.0 + np.random.uniform(-5, 5),
                    'low': 95.0 + np.random.uniform(-5, 5),
                    'close': 100.0 + np.random.uniform(-5, 5),
                    'volume': 1000.0 + np.random.uniform(0, 500),
                    'adjusted_close': 100.0 + np.random.uniform(-5, 5)
                })
        df = pd.DataFrame(simulated_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
        logger.info(f"Generated simulated market data for {len(tickers)} tickers from {start_date} to {end_date}")
        return df
    if use_alpha_vantage:
        logger.info("Using Alpha Vantage API for market data")
        @sleep_and_retry
        @limits(calls=calls_per_minute, period=period)
        async def fetch_ticker_data(session, ticker):
            try:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={api_key}"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Alpha Vantage: Failed to fetch data for {ticker}: HTTP {response.status}")
                        return []
                    data = await response.json()
                    if 'Time Series (Daily)' not in data:
                        logger.warning(f"Alpha Vantage: No data for {ticker}: {data.get('Note', 'No data returned')}")
                        return []
                    daily_data = data['Time Series (Daily)']
                    return [
                        {
                            'ticker': ticker,
                            'date': date,
                            'close': float(item['4. close']),
                            'volume': float(item['6. volume']),
                            'high': float(item['2. high']),
                            'low': float(item['3. low']),
                            'open': float(item['1. open']),
                            'adjusted_close': float(item['5. adjusted close'])
                        } for date, item in daily_data.items() if pd.to_datetime(date) >= pd.to_datetime(start_date) and pd.to_datetime(date) <= pd.to_datetime(end_date)
                    ]
            except Exception as e:
                logger.error(f"Alpha Vantage: Error fetching data for {ticker}: {e}")
                return []
    else:
        logger.info("Using Polygon API for market data")
        @sleep_and_retry
        @limits(calls=calls_per_minute, period=period)
        async def fetch_ticker_data(session, ticker):
            try:
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={api_key}"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Polygon: Failed to fetch data for {ticker}: HTTP {response.status}")
                        return []
                    data = await response.json()
                    if 'results' not in data:
                        logger.warning(f"Polygon: No data for {ticker}: {data.get('message', 'No data returned')}")
                        return []
                    return [
                        {
                            'ticker': ticker,
                            'date': datetime.fromtimestamp(item['t'] / 1000).strftime('%Y-%m-%d'),
                            'close': float(item['c']),
                            'volume': float(item['v']),
                            'high': float(item['h']),
                            'low': float(item['l']),
                            'open': float(item['o']),
                            'adjusted_close': float(item['c'])
                        } for item in data['results']
                    ]
            except Exception as e:
                logger.error(f"Polygon: Error fetching data for {ticker}: {e}")
                return []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_ticker_data(session, ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, list) and result:
            all_data.extend(result)
        elif isinstance(result, Exception):
            logger.error(f"Error in fetch_market_data: {result}")
    if not all_data:
        logger.error("No market data fetched for any ticker.")
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    if df.empty:
        logger.error("Market data DataFrame is empty after processing.")
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    logger.info(f"Fetched market data for {len(tickers)} tickers from {start_date} to {end_date}")
    return df
async def fetch_news_sentiment(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Fetch news sentiment data for specified tickers and date range."""
    sentiment_dfs = {}
    analyzer = SentimentIntensityAnalyzer()
    async def fetch_news_for_ticker(ticker: str) -> pd.DataFrame:
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            sentiments = []
            for date in dates:
                news_text = f"Sample news for {ticker} on {date}"
                sentiment_scores = analyzer.polarity_scores(news_text)
                sentiments.append(sentiment_scores['compound'])
            df = pd.DataFrame({'date': dates, 'sentiment': sentiments})
            logger.debug(f"Fetched sentiment data for {ticker}: shape={df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {ticker}: {e}")
            return pd.DataFrame({'date': dates, 'sentiment': [0.0] * len(dates)})
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, lambda t=ticker: asyncio.run(fetch_news_for_ticker(t))) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for ticker, result in zip(tickers, results):
        if isinstance(result, pd.DataFrame) and not result.empty:
            sentiment_dfs[ticker] = result
        else:
            logger.warning(f"No sentiment data for {ticker}. Using default.")
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            sentiment_dfs[ticker] = pd.DataFrame({'date': dates, 'sentiment': [0.0] * len(dates)})
    return sentiment_dfs
async def fetch_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch VIX and S&P 500 data for the specified date range."""
    try:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        vix_data = {
            'date': dates,
            'vix': [20.0 + secrets.randbelow(10) - 5 for _ in dates],
            'sp500_return': [secrets.randbelow(40) / 1000 - 0.02 for _ in dates],
            'sp500_momentum': [secrets.randbelow(200) / 1000 - 0.1 for _ in dates]
        }
        df = pd.DataFrame(vix_data)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Fetched VIX data from {start_date} to {end_date}, shape={df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error fetching VIX data: {e}")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame({
            'date': dates,
            'vix': [20.0] * len(dates),
            'sp500_return': [0.0] * len(dates),
            'sp500_momentum': [0.0] * len(dates)
        })
async def fetch_top_stocks(tickers: List[str]) -> pd.DataFrame:
    """Fetch top stocks data for specified tickers."""
    try:
        top_stocks_data = {
            'ticker': tickers,
            'sentiment': [secrets.randbelow(1000) / 1000 - 0.5 for _ in tickers],
            'final_score': [secrets.randbelow(1000) / 1000 for _ in tickers]
        }
        df = pd.DataFrame(top_stocks_data)
        logger.info(f"Fetched top stocks data for {len(tickers)} tickers, shape={df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error fetching top stocks data: {e}")
        return pd.DataFrame({
            'ticker': tickers,
            'sentiment': [0.0] * len(tickers),
            'final_score': [0.0] * len(tickers)
        })
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the input DataFrame."""
    try:
        df = df.copy()
        grouped = df.groupby('ticker')
        df_list = []
        for ticker, group in grouped:
            group = group.sort_values('date')
            group['sma_5'] = group['close'].rolling(window=5).mean().astype(np.float32)
            group['sma_20'] = group['close'].rolling(window=20).mean().astype(np.float32)
            delta = group['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            group['rsi'] = 100 - (100 / (1 + rs))
            exp1 = group['close'].ewm(span=12, adjust=False).mean()
            exp2 = group['close'].ewm(span=26, adjust=False).mean()
            group['macd'] = exp1 - exp2
            sma20 = group['close'].rolling(window=20).mean()
            std20 = group['close'].rolling(window=20).std()
            group['bb_width'] = (group['close'] - sma20) / std20
            high_low = group['high'] - group['low']
            high_close = np.abs(group['high'] - group['close'].shift())
            low_close = np.abs(group['low'] - group['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            group['atr'] = tr.rolling(window=14).mean()
            dx = 100 * np.abs(group['high'].diff() - group['low'].diff()) / (group['high'].diff() + group['low'].diff())
            group['adx'] = dx.rolling(window=14).mean()
            group['volatility'] = group['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            group['momentum'] = group['close'].pct_change(periods=5)
            group['volume_trend'] = group['volume'].pct_change(periods=5)
            df_list.append(group)
        result_df = pd.concat(df_list, ignore_index=True)
        result_df = result_df.fillna(0.0)
        logger.info("Calculated technical indicators for all tickers")
        return result_df
    except Exception as e:
        logger.error(f"Error in calculate_technical_indicators: {e}", exc_info=True)
        return df

def objective(trial: optuna.Trial, df: pd.DataFrame, top_stocks_df: pd.DataFrame, vix_df: pd.DataFrame, sentiment_dfs: Dict[str, pd.DataFrame], tickers: List[str]) -> float:
    """Objective function for hyperparameter optimization."""
    try:
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
        gamma = trial.suggest_float('gamma', 0.9, 0.999)
        ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
        n_epochs = trial.suggest_int('n_epochs', 5, 20)
        env = TradingEnv(
            df=df,
            top_stocks_df=top_stocks_df,
            vix_df=vix_df,
            sentiment_dfs=sentiment_dfs,
            backtest_mode=True,
            training_mode=True,
            start_date=df['date'].min(),
            end_date=df['date'].max()
        )
        env = DummyVecEnv([lambda: env])
        check_env(env, warn=True)
        policy_kwargs = {
            'features_extractor_class': TransformerFeaturesExtractor,
            'features_extractor_kwargs': {'nhead': 8, 'num_layers': 3},
            'net_arch': [128, 64]
        }
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            ent_coef=ent_coef,
            n_epochs=n_epochs,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=DEVICE
        )
        callback = CustomCallback(checkpoint_freq=CHECKPOINT_FREQ)
        callback.set_total_timesteps(TOTAL_TIMESTEPS)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        total_reward = 0.0
        obs, _ = env.reset()
        for _ in range(env.envs[0].max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        logger.info(f"Trial {trial.number}: Total reward = {total_reward:.2f}, Params = {trial.params}")
        return total_reward
    except Exception as e:
        logger.error(f"Error in objective function: {e}")
        return -np.inf

def main():
    """Main function to run the trading environment."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_env.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger.info("Starting main function")
        print("Starting main function...")

        # Define date range and tickers
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        logger.info(f"Processing tickers: {tickers} from {start_date} to {end_date}")
        print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}")

        # Fetch data
        df = asyncio.run(fetch_market_data(tuple(tickers), start_date, end_date))
        if df.empty:
            logger.error("Failed to fetch market data. Exiting.")
            print("Error: No market data fetched. Check logs for details.")
            sys.exit(1)
        logger.info(f"Fetched market data: {df.shape}")
        print(f"Fetched market data: {df.shape[0]} rows, {df.shape[1]} columns")

        df = calculate_technical_indicators(df)
        if df.empty:
            logger.error("Failed to calculate technical indicators. Exiting.")
            print("Error: Technical indicators calculation failed.")
            sys.exit(1)
        logger.info(f"Calculated technical indicators: {df.shape}")
        print(f"Calculated technical indicators for {len(df['ticker'].unique())} tickers")

        top_stocks_df = asyncio.run(fetch_top_stocks(tickers))
        logger.info(f"Fetched top stocks data: {top_stocks_df.shape}")
        print(f"Fetched top stocks data: {top_stocks_df.shape[0]} rows")

        vix_df = asyncio.run(fetch_vix_data(start_date, end_date))
        logger.info(f"Fetched VIX data: {vix_df.shape}")
        print(f"Fetched VIX data: {vix_df.shape[0]} rows")

        sentiment_dfs = asyncio.run(fetch_news_sentiment(tickers, start_date, end_date))
        logger.info(f"Fetched sentiment data for {len(sentiment_dfs)} tickers")
        print(f"Fetched sentiment data for {len(sentiment_dfs)} tickers")

        # Initialize environment
        env = TradingEnv(
            df=df,
            top_stocks_df=top_stocks_df,
            vix_df=vix_df,
            sentiment_dfs=sentiment_dfs,
            backtest_mode=True,
            training_mode=True,
            start_date=start_date,
            end_date=end_date
        )
        logger.info("Trading environment initialized")
        print("Trading environment initialized")

        # Check environment
        check_env(env, warn=True)
        env = DummyVecEnv([lambda: env])
        logger.info("Environment check passed")
        print("Environment check passed")

        # Hyperparameter optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, df, top_stocks_df, vix_df, sentiment_dfs, tickers),
            n_trials=OPTUNA_TRIALS,
            n_jobs=1
        )
        logger.info(f"Best trial: {study.best_trial.number}, Value: {study.best_trial.value}, Params: {study.best_trial.params}")
        print(f"Best trial: {study.best_trial.number}, Value: {study.best_trial.value}, Params: {study.best_trial.params}")

        # Train model with best parameters
        best_params = study.best_trial.params
        policy_kwargs = {
            'features_extractor_class': TransformerFeaturesExtractor,
            'features_extractor_kwargs': {'nhead': 8, 'num_layers': 3},
            'net_arch': [128, 64]
        }
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=best_params['learning_rate'],
            batch_size=best_params['batch_size'],
            gamma=best_params['gamma'],
            ent_coef=best_params['ent_coef'],
            n_epochs=best_params['n_epochs'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=DEVICE
        )
        logger.info("Starting model training with best parameters")
        print("Starting model training...")

        callback = CustomCallback(checkpoint_freq=CHECKPOINT_FREQ)
        callback.set_total_timesteps(TOTAL_TIMESTEPS)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        logger.info("Model training completed")
        print("Model training completed")

        model.save(RL_MODEL_SAVE_PATH)
        logger.info(f"Model saved to {RL_MODEL_SAVE_PATH}")
        print(f"Model saved to {RL_MODEL_SAVE_PATH}")

        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(env.envs[0].max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        logger.info(f"Evaluation completed. Total reward: {total_reward:.2f}")
        print(f"Evaluation completed. Total reward: {total_reward:.2f}")

        subject = "Trading Environment Run Completed"
        body = (
            f"Trading environment run completed.\n"
            f"Best trial: {study.best_trial.number}\n"
            f"Best value: {study.best_trial.value:.2f}\n"
            f"Best params: {study.best_trial.params}\n"
            f"Total reward: {total_reward:.2f}"
        )
        send_email(subject, body)
        logger.info("Main function completed successfully")
        print("Main function completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        print(f"Error in main function: {e}")
        subject = "Trading Environment Error"
        body = f"An error occurred in the trading environment: {str(e)}"
        send_email(subject, body)
        sys.exit(1)

if __name__ == '__main__':
    main()