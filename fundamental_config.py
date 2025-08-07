#!/usr/bin/env python3
# AI_sales/src/fundamental_config.py - Fundamental Analysis for Trading Pipeline
# Configured for MacBook Pro M4 Max (14 cores, 36 GB RAM)

"""
Fundamental configuration script (April 27, 2025):
- Aligns with universe_builder.py logic.
- Uses only Polygon.io for data fetching (removed Alpha Vantage dependency).
- Introduces simulated price data for missing market data, aligned with universe_builder.py.
- Enhances sentiment scoring with a heuristic fallback, using universe_builder.py's sentiment_score when available.
- Fixes uniform momentum, volatility, and sentiment_score issues.
- Aligns key ticker enforcement ranges with universe_builder.py (NVDA: 1, AAPL: 2, AVGO: 3).
- Uses simulated data for non-key tickers to bypass API failures.
- Fixes KeyError for NVDA by using ESTIMATED_2024 for key tickers.
- Adds debug logging to diagnose KeyError issue.
- Adds LATEST_RANKINGS global variable for enhanced_trading_pipeline_v214_new.py.
- Modified to save rankings to a fixed file to prevent accumulation.
- Incorporates technical indicators from current_universe.csv into scoring.
- Fixes key ticker enforcement to ensure NVDA: 1, AAPL: 2, AVGO: 3.
- Caps revenue_growth to prevent outliers (e.g., GOOG's 350.0).
- Adjusts scoring weights for better alignment with universe_builder.py.
- Fixes SyntaxError in data dictionary by removing erroneous 'perfect: true'.
- Selects top 10 stocks from the top 20 provided by universe_builder.py.
- Fixes key ticker enforcement order to ensure NVDA: 1, AAPL: 2, AVGO: 3.
- Fixes SettingWithCopyWarning by using .copy() for df_final.
- Improves sector diversity by introducing a sector cap (max 3 Technology stocks) and adjusting weights.
- Fixes sector cap logic to strictly enforce the max Technology stock limit.
- Fixes SyntaxError in simulate_price_data function (p_e asd > 0 -> p_e > 0).
- Fixes KeyError: ' ticker' by correcting the column name to 'ticker'.
- Ensures exactly TOP_N tickers are returned by relaxing the sector cap if necessary.
- Adjusts SECTOR_WEIGHTS to boost non-Technology stocks for better sector diversity.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import time
import ratelimit
import pickle
import shutil
from typing import Dict, List, Any, Optional
from config import MAX_WORKERS, logger, RETRY_DELAY, SOURCE_CREDIBILITY

# Global variable to store the latest rankings
LATEST_RANKINGS = None

# ----------------------------------------------------------------------
# Define BASE_DIR and Constants
# ----------------------------------------------------------------------
BASE_DIR = Path("/Users/amzadhaque/AI_sales/src")
UNIVERSE_FILE = BASE_DIR / "current_universe.csv"
CACHE_DIR = BASE_DIR / "cache"
POLYGON_AGG_URL = "https://api.polygon.io/v2/aggs/ticker"
POLYGON_FINANCIALS_URL = "https://api.polygon.io/vX/reference/financials"
POLYGON_SENTIMENT_URL = "https://api.polygon.io/v2/reference/news"

# Load environment variables
load_dotenv(BASE_DIR / ".env")
env_file = BASE_DIR / "list.env"
if env_file.exists():
    load_dotenv(env_file, override=False)
    logger.info(f"Loaded {env_file}")
else:
    logger.warning(f"No {env_file} found, proceeding without it")

# Configuration from .env
START_DATE = "2024-01-01"
END_DATE = "2024-04-13"
SENTIMENT_END_DATE = END_DATE
SENTIMENT_START_DATE = (datetime.strptime(SENTIMENT_END_DATE, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "0.5"))
VOLATILITY_WINDOW = int(os.getenv("VOLATILITY_WINDOW", "21"))
TOP_N = 10  # Select top 10 stocks from the top 20 provided by universe_builder.py
MAX_TECH_STOCKS = 3  # Sector cap for Technology stocks in top 10

# Hardcoded data for key tickers (aligned with universe_builder.py)
ESTIMATED_2024 = {
    "NVDA": {
        "market_cap": 2.706692e+12,
        "avg_dollar_vol": 5e9,
        "momentum_6m": 0.7736,
        "momentum_1m": 0.1,
        "price_growth": 0.7736,
        "eps_val": 50.0,
        "revenue_growth": 47.9,
        "cash_flow_growth": 10.0,
        "net_margin": 25.0,
        "debt_to_equity": 0.5,
        "sentiment_score": 75.0,
        "close": 100.0 * (1 + 0.7736),
        "rsi": 45.0,
        "macd_signal": 1,
        "macd_histogram": 0.5,
        "bbands_width": 0.05,
        "sma_trend": 1,
        "stoch_k": 70.0,
        "atr": 5.0,
        "vwap": 105.0
    },
    "AAPL": {
        "market_cap": 2.976629e+12,
        "avg_dollar_vol": 6e9,
        "momentum_6m": 0.35,
        "momentum_1m": 0.05,
        "price_growth": 0.35,
        "eps_val": 15.0,
        "revenue_growth": 8.0,
        "cash_flow_growth": 10.0,
        "net_margin": 25.0,
        "debt_to_equity": 0.5,
        "sentiment_score": 70.0,
        "close": 100.0 * (1 + 0.35),
        "rsi": 50.0,
        "macd_signal": 1,
        "macd_histogram": 0.3,
        "bbands_width": 0.04,
        "sma_trend": 1,
        "stoch_k": 65.0,
        "atr": 4.0,
        "vwap": 102.0
    },
    "AVGO": {
        "market_cap": 8.554728e+11,
        "avg_dollar_vol": 3e9,
        "momentum_6m": 0.7086,
        "momentum_1m": 0.08,
        "price_growth": 0.7086,
        "eps_val": 50.0,
        "revenue_growth": 20.0,
        "cash_flow_growth": 10.0,
        "net_margin": 25.0,
        "debt_to_equity": 0.5,
        "sentiment_score": 65.0,
        "close": 100.0 * (1 + 0.7086),
        "rsi": 40.0,
        "macd_signal": 1,
        "macd_histogram": 0.7,
        "bbands_width": 0.06,
        "sma_trend": 1,
        "stoch_k": 60.0,
        "atr": 6.0,
        "vwap": 108.0
    }
}

# Simulated data for non-key tickers (aligned with universe_builder.py)
SIMULATED_2024 = {
    "NFLX": {
        "price_performance": 32.9413,
        "eps_growth": 20.28,
        "revenue_growth": 15.0,
        "cash_flow_growth": 10.0,
        "net_margin": 15.0,
        "debt_to_equity": 1.0,
        "p_e": 46.28,
        "market_cap": 3.928050e+11,
        "sentiment_score": 57.0,
        "portfolio_return": 0.3294,
        "rsi": 50.0,
        "macd_signal": 0,
        "macd_histogram": 0.0,
        "bbands_width": 0.05,
        "sma_trend": 0,
        "stoch_k": 50.0,
        "atr": 0.0,
        "vwap": 0.0
    },
    "JPM": {
        "price_performance": 10.9754,
        "eps_growth": 18.02,
        "revenue_growth": 8.0,
        "cash_flow_growth": 5.0,
        "net_margin": 15.0,
        "debt_to_equity": 2.0,
        "p_e": 11.5,
        "market_cap": 6.564234e+11,
        "sentiment_score": 55.5,
        "portfolio_return": 0.1098,
        "rsi": 50.0,
        "macd_signal": 0,
        "macd_histogram": 0.0,
        "bbands_width": 0.05,
        "sma_trend": 0,
        "stoch_k": 50.0,
        "atr": 0.0,
        "vwap": 0.0
    },
    "COST": {
        "price_performance": 23.834,
        "eps_growth": 17.06,
        "revenue_growth": 3.14,
        "cash_flow_growth": 3.14,
        "net_margin": 10.35,
        "debt_to_equity": 0.8,
        "p_e": 56.18,
        "market_cap": 4.274486e+11,
        "sentiment_score": 56.0,
        "portfolio_return": 0.2383,
        "rsi": 50.0,
        "macd_signal": 0,
        "macd_histogram": 0.0,
        "bbands_width": 0.05,
        "sma_trend": 0,
        "stoch_k": 50.0,
        "atr": 0.0,
        "vwap": 0.0
    },
    "META": {
        "price_performance": 23.834,
        "eps_growth": 24.61,
        "revenue_growth": 10.0,
        "cash_flow_growth": 10.0,
        "net_margin": 25.0,
        "debt_to_equity": 0.5,
        "p_e": 23.38,
        "market_cap": 1.377221e+12,
        "sentiment_score": 58.0,
        "portfolio_return": 0.2383,
        "rsi": 50.0,
        "macd_signal": 0,
        "macd_histogram": 0.0,
        "bbands_width": 0.05,
        "sma_trend": 0,
        "stoch_k": 50.0,
        "atr": 0.0,
        "vwap": 0.0
    },
    "MSFT": {
        "price_performance": 23.834,
        "eps_growth": 12.17,
        "revenue_growth": 10.0,
        "cash_flow_growth": 10.0,
        "net_margin": 25.0,
        "debt_to_equity": 0.5,
        "p_e": 31.35,
        "market_cap": 2.887730e+12,
        "sentiment_score": 56.5,
        "portfolio_return": 0.2383,
        "rsi": 50.0,
        "macd_signal": 0,
        "macd_histogram": 0.0,
        "bbands_width": 0.05,
        "sma_trend": 0,
        "stoch_k": 50.0,
        "atr": 0.0,
        "vwap": 0.0
    },
    "AMZN": {
        "price_performance": 23.834,
        "eps_growth": 6.1157,
        "revenue_growth": 5.0,
        "cash_flow_growth": 5.0,
        "net_margin": 15.0,
        "debt_to_equity": 1.0,
        "p_e": 33.43,
        "market_cap": 1.961914e+12,
        "sentiment_score": 56.0,
        "portfolio_return": 0.2383,
        "rsi": 50.0,
        "macd_signal": 0,
        "macd_histogram": 0.0,
        "bbands_width": 0.05,
        "sma_trend": 0,
        "stoch_k": 50.0,
        "atr": 0.0,
        "vwap": 0.0
    },
    "MA": {
        "price_performance": 2.3553,
        "eps_growth": 13.91,
        "revenue_growth": 5.0,
        "cash_flow_growth": 5.0,
        "net_margin": 15.0,
        "debt_to_equity": 2.0,
        "p_e": 35.0,
        "market_cap": 4.647434e+11,
        "sentiment_score": 57.0,
        "portfolio_return": 0.0236,
        "rsi": 50.0,
        "macd_signal": 0,
        "macd_histogram": 0.0,
        "bbands_width": 0.05,
        "sma_trend": 0,
        "stoch_k": 50.0,
        "atr": 0.0,
        "vwap": 0.0
    }
}

# Key tickers with enforced ranks (aligned with universe_builder.py)
KEY_TICKERS = {
    "NVDA": 1,
    "AAPL": 2,
    "AVGO": 3
}

# Sector weights for scoring (adjusted to improve diversity)
SECTOR_WEIGHTS = {
    "Technology": {
        "price_performance": 0.08,  # Reduced to balance with other sectors
        "momentum": 0.08,           # Reduced to balance with other sectors
        "volatility": 0.05,
        "eps_growth": 0.08,         # Reduced to allow non-tech sectors to compete
        "revenue_growth": 0.10,
        "cash_flow_growth": 0.05,
        "dividend_yield": 0.05,
        "net_margin": 0.10,
        "debt_to_equity": 0.05,
        "sentiment_score": 0.10,
        "rsi": 0.05,
        "macd_signal": 0.05,
        "macd_histogram": 0.05,
        "bbands_width": 0.05,
        "sma_trend": 0.05,
        "stoch_k": 0.05,
        "atr": 0.05,
        "vwap": 0.05
    },
    "Financial": {
        "price_performance": 0.15,  # Increased to boost non-Tech
        "momentum": 0.15,           # Increased to boost non-Tech
        "volatility": 0.05,
        "eps_growth": 0.20,
        "revenue_growth": 0.10,
        "cash_flow_growth": 0.05,
        "dividend_yield": 0.075,
        "net_margin": 0.15,
        "debt_to_equity": 0.05,
        "sentiment_score": 0.15,    # Increased to boost non-Tech
        "rsi": 0.05,
        "macd_signal": 0.05,
        "macd_histogram": 0.05,
        "bbands_width": 0.05,
        "sma_trend": 0.05,
        "stoch_k": 0.05,
        "atr": 0.05,
        "vwap": 0.05
    },
    "Consumer Defensive": {
        "price_performance": 0.15,  # Increased to boost non-Tech
        "momentum": 0.15,           # Increased to boost non-Tech
        "volatility": 0.05,
        "eps_growth": 0.20,
        "revenue_growth": 0.10,
        "cash_flow_growth": 0.05,
        "dividend_yield": 0.075,
        "net_margin": 0.15,
        "debt_to_equity": 0.05,
        "sentiment_score": 0.15,    # Increased to boost non-Tech
        "rsi": 0.05,
        "macd_signal": 0.05,
        "macd_histogram": 0.05,
        "bbands_width": 0.05,
        "sma_trend": 0.05,
        "stoch_k": 0.05,
        "atr": 0.05,
        "vwap": 0.05
    },
    "Communication Services": {
        "price_performance": 0.15,  # Increased to boost non-Tech
        "momentum": 0.15,           # Increased to boost non-Tech
        "volatility": 0.05,
        "eps_growth": 0.15,
        "revenue_growth": 0.15,
        "cash_flow_growth": 0.05,
        "dividend_yield": 0.05,
        "net_margin": 0.15,
        "debt_to_equity": 0.05,
        "sentiment_score": 0.15,    # Increased to boost non-Tech
        "rsi": 0.05,
        "macd_signal": 0.05,
        "macd_histogram": 0.05,
        "bbands_width": 0.05,
        "sma_trend": 0.05,
        "stoch_k": 0.05,
        "atr": 0.05,
        "vwap": 0.05
    },
    "Consumer Cyclical": {
        "price_performance": 0.15,  # Increased to boost non-Tech
        "momentum": 0.15,           # Increased to boost non-Tech
        "volatility": 0.05,
        "eps_growth": 0.15,
        "revenue_growth": 0.15,
        "cash_flow_growth": 0.05,
        "dividend_yield": 0.05,
        "net_margin": 0.15,
        "debt_to_equity": 0.05,
        "sentiment_score": 0.15,    # Increased to boost non-Tech
        "rsi": 0.05,
        "macd_signal": 0.05,
        "macd_histogram": 0.05,
        "bbands_width": 0.05,
        "sma_trend": 0.05,
        "stoch_k": 0.05,
        "atr": 0.05,
        "vwap": 0.05
    },
    "Healthcare": {
        "price_performance": 0.15,  # Increased to boost non-Tech
        "momentum": 0.15,           # Increased to boost non-Tech
        "volatility": 0.05,
        "eps_growth": 0.15,
        "revenue_growth": 0.15,
        "cash_flow_growth": 0.05,
        "dividend_yield": 0.05,
        "net_margin": 0.15,
        "debt_to_equity": 0.05,
        "sentiment_score": 0.15,    # Increased to boost non-Tech
        "rsi": 0.05,
        "macd_signal": 0.05,
        "macd_histogram": 0.05,
        "bbands_width": 0.05,
        "sma_trend": 0.05,
        "stoch_k": 0.05,
        "atr": 0.05,
        "vwap": 0.05
    },
    None: {
        "price_performance": 0.10,
        "momentum": 0.10,
        "volatility": 0.05,
        "eps_growth": 0.15,
        "revenue_growth": 0.10,
        "cash_flow_growth": 0.05,
        "dividend_yield": 0.05,
        "net_margin": 0.10,
        "debt_to_equity": 0.05,
        "sentiment_score": 0.10,
        "rsi": 0.05,
        "macd_signal": 0.05,
        "macd_histogram": 0.05,
        "bbands_width": 0.05,
        "sma_trend": 0.05,
        "stoch_k": 0.05,
        "atr": 0.05,
        "vwap": 0.05
    }
}

# Sector averages for fundamentals fallback (aligned with universe_builder.py)
SECTOR_AVG_GROWTH = {
    "Technology": 10.0, "Financial": 5.0, "Consumer Defensive": 3.0, "Communication Services": 5.0, "Consumer Cyclical": 5.0, "Healthcare": 5.0, None: 5.0
}

SECTOR_AVG_NET_MARGIN = {
    "Technology": 25.0, "Financial": 15.0, "Consumer Defensive": 10.35, "Communication Services": 15.0, "Consumer Cyclical": 15.0, "Healthcare": 15.0, None: 15.0
}

SECTOR_AVG_DEBT_TO_EQUITY = {
    "Technology": 0.5, "Financial": 2.0, "Consumer Defensive": 0.8, "Communication Services": 1.0, "Consumer Cyclical": 1.0, "Healthcare": 1.0, None: 1.0
}

# Sector mapping (aligned with universe_builder.py, fixed typo 'C conceit')
SECTOR_MAPPING = {
    "NVDA": "Technology", "AAPL": "Technology", "AVGO": "Technology", "MSFT": "Technology", 
    "CSCO": "Technology", "TSLA": "Consumer Cyclical", "BKNG": "Consumer Cyclical", 
    "V": "Financial", "MA": "Financial", "JPM": "Financial", "WFC": "Financial", 
    "SCHW": "Financial", "BAC": "Financial", "AXP": "Financial", "HSBC": "Financial", 
    "PG": "Consumer Defensive", "MCD": "Consumer Cyclical", "ABBV": "Healthcare", 
    "TMUS": "Communication Services", "T": "Communication Services", "BABA": "Consumer Cyclical", 
    "GE": "Industrials", "PM": "Consumer Defensive", "APP": "Technology", "SAP": "Technology", 
    "LIN": "Basic Materials", "BRK-B": "Financial", "WMT": "Consumer Defensive", 
    "TSM": "Technology", "IBM": "Technology", "KO": "Consumer Defensive", "META": "Technology", 
    "DE": "Industrials", "AMZN": "Consumer Cyclical", "NFLX": "Communication Services", 
    "COST": "Consumer Defensive", "UNH": "Healthcare", "PGR": "Financial", "LLY": "Healthcare", 
    "CVX": "Energy", "GOOG": "Technology", "MS": "Financial", "XOM": "Energy", 
    "ABT": "Healthcare", "CRM": "Technology", "NOW": "Technology"
}

# ----------------------------------------------------------------------
# API SETUP
# ----------------------------------------------------------------------
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
if not POLYGON_API_KEY:
    logger.error("Missing POLYGON_API_KEY")
    raise ValueError("POLYGON_API_KEY not configured")

# Rate limiter for Polygon API (600 requests per minute for paid plan)
@ratelimit.sleep_and_retry
@ratelimit.limits(calls=600, period=60)
def rate_limited_request(url, params, timeout=REQUEST_TIMEOUT):
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp

# Cache setup
CACHE_DIR.mkdir(exist_ok=True)

def clear_cache():
    """Clear the cache directory to ensure fresh data."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        logger.info(f"Cleared cache directory: {CACHE_DIR}")
    CACHE_DIR.mkdir(exist_ok=True)

def load_cache(filename):
    filepath = CACHE_DIR / filename
    if filepath.exists():
        cache_age = datetime.now().timestamp() - filepath.stat().st_mtime
        if cache_age < 24 * 60 * 60:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            logger.info(f"Cache for {filename} is older than 24 hours, refreshing")
    return None

def save_cache(data, filename):
    filepath = CACHE_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

# ----------------------------------------------------------------------
# DATA FETCHING FUNCTIONS
# ----------------------------------------------------------------------
def simulate_price_data(ticker: str, sector: Optional[str] = None, market_cap: Optional[float] = None, p_e: Optional[float] = None) -> pd.DataFrame:
    """Simulate price data for a ticker over the backtest period with ticker-specific variability (aligned with universe_builder.py)."""
    seed_value = sum(ord(c) for c in ticker)
    rng = np.random.RandomState(seed_value)
    
    base_price = 100.0 * (1 + rng.uniform(-0.5, 0.5))
    volatility_multiplier = 1.0
    if sector in SECTOR_MAPPING.values():
        volatility_multiplier = {
            "Technology": 1.5,
            "Financial": 1.0,
            "Consumer Defensive": 0.8,
            "Healthcare": 1.2,
            "Communication Services": 1.1,
            "Consumer Cyclical": 1.3,
            None: 1.0
        }.get(sector, 1.0)
    if market_cap:
        market_cap_factor = 1.0 / (market_cap / 500e9) if market_cap > 0 else 1.0
        volatility_multiplier *= max(0.3, min(2.0, market_cap_factor))
    if p_e:
        pe_factor = (p_e / 30.0) if p_e > 0 else 1.0
        volatility_multiplier *= max(0.5, min(2.0, pe_factor))
    daily_volatility = rng.uniform(0.02, 0.05) * volatility_multiplier

    trend_factor = rng.uniform(0.005, 0.015) if ticker in KEY_TICKERS else rng.uniform(-0.01, 0.01)
    if market_cap:
        trend_factor *= max(0.3, min(2.0, 500e9 / market_cap))
    if p_e:
        trend_factor += (p_e - 30.0) / 30.0 * 0.005

    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    prices = [base_price]
    for _ in range(1, len(date_range)):
        daily_change = rng.normal(trend_factor, daily_volatility)
        prices.append(max(prices[-1] * (1 + daily_change), 1.0))
    
    df = pd.DataFrame({
        "date": date_range,
        "close": prices,
        "volume": rng.randint(100000, 1000000, size=len(date_range))
    })
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(window=VOLATILITY_WINDOW, min_periods=1).std() * np.sqrt(252)
    momentum = (df["close"].iloc[-1] - df["close"].mean()) / df["close"].mean() if df["close"].mean() != 0 else 0
    logger.debug(f"{ticker}: Simulated price data - base_price={base_price:.2f}, daily_volatility={daily_volatility:.4f}, trend_factor={trend_factor:.4f}, momentum={momentum:.4f}, volatility={df['volatility'].iloc[-1]:.4f}")
    return df

def get_price_data(ticker: str, sector: Optional[str] = None, price_performance: Optional[float] = None, market_cap: Optional[float] = None, p_e: Optional[float] = None) -> Dict[str, float]:
    """Fetch or simulate price data for a ticker using Polygon.io."""
    if ticker in ESTIMATED_2024 or ticker in SIMULATED_2024:
        logger.info(f"{ticker}: Using ESTIMATED_2024/SIMULATED_2024 price_performance={price_performance}")
        return {
            "price_performance": price_performance,
            "momentum": price_performance / 100,
            "volatility": 0.376155
        }

    url = f"{POLYGON_AGG_URL}/{ticker}/range/1/day/{START_DATE}/{END_DATE}?adjusted=true&sort=asc&limit=10000"
    params = {"apiKey": POLYGON_API_KEY}
    cache_key = f"price_{ticker}_{START_DATE}_{END_DATE}.pkl"
    cached_data = load_cache(cache_key)
    if cached_data:
        logger.debug(f"Using cached price data for {ticker}")
        return cached_data

    try:
        resp = rate_limited_request(url, params)
        data = resp.json().get("results", [])
        if not data:
            raise ValueError("No price data returned")
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df.rename(columns={"c": "close"}, inplace=True)
        df["returns"] = df["close"].pct_change()
        volatility = df["returns"].rolling(window=VOLATILITY_WINDOW, min_periods=1).std().iloc[-1] * np.sqrt(252)
        momentum = (df["close"].iloc[-1] - df["close"].mean()) / df["close"].mean() if df["close"].mean() != 0 else 0
        price_performance = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
        result = {
            "price_performance": price_performance,
            "momentum": momentum,
            "volatility": volatility
        }
        save_cache(result, cache_key)
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch price data for {ticker}: {e}, using simulated data")
        df = simulate_price_data(ticker, sector, market_cap, p_e)
        volatility = df["volatility"].iloc[-1]
        momentum = (df["close"].iloc[-1] - df["close"].mean()) / df["close"].mean() if df["close"].mean() != 0 else 0
        price_performance = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
        result = {
            "price_performance": price_performance,
            "momentum": momentum,
            "volatility": volatility
        }
        save_cache(result, cache_key)
        return result

def get_fundamentals(ticker: str, period: str = "annual", retries: int = 3) -> pd.DataFrame:
    """Fetch fundamentals with retries using Polygon.io, falling back to sector averages."""
    logger.debug(f"Processing fundamentals for ticker: {ticker}, KEY_TICKERS: {list(KEY_TICKERS.keys())}")
    if ticker in KEY_TICKERS:
        logger.info(f"{ticker}: Using ESTIMATED_2024 fundamentals")
        sector = SECTOR_MAPPING.get(ticker, None)
        return pd.DataFrame([{
            "eps": ESTIMATED_2024[ticker]["eps_val"],
            "revenue": ESTIMATED_2024[ticker].get("revenue_growth", 0),
            "cash_flow": ESTIMATED_2024[ticker].get("cash_flow_growth", 0),
            "net_margin": ESTIMATED_2024[ticker].get("net_margin", SECTOR_AVG_NET_MARGIN.get(sector, 15.0)),
            "debt_to_equity": ESTIMATED_2024[ticker].get("debt_to_equity", SECTOR_AVG_DEBT_TO_EQUITY.get(sector, 1.0))
        }])
    if ticker in SIMULATED_2024:
        logger.info(f"{ticker}: Using SIMULATED_2024 fundamentals")
        return pd.DataFrame([{
            "eps": SIMULATED_2024[ticker]["eps_growth"],
            "revenue": SIMULATED_2024[ticker]["revenue_growth"],
            "cash_flow": SIMULATED_2024[ticker]["cash_flow_growth"],
            "net_margin": SIMULATED_2024[ticker]["net_margin"],
            "debt_to_equity": SIMULATED_2024[ticker]["debt_to_equity"]
        }])

    cache_key = f"fundamentals_{ticker}_{period}.pkl"
    cached_data = load_cache(cache_key)
    if cached_data is not None:
        logger.debug(f"Using cached fundamentals for {ticker}")
        return pd.DataFrame([cached_data])

    sector = SECTOR_MAPPING.get(ticker, None)
    ticker_formatted = ticker.replace(".", "-")
    url = POLYGON_FINANCIALS_URL
    params = {
        "ticker": ticker_formatted,
        "limit": 4,
        "apiKey": POLYGON_API_KEY,
    }
    for attempt in range(retries):
        try:
            resp = rate_limited_request(url, params)
            data = resp.json().get("results", [])
            if not data or len(data) < 2:
                logger.warning(f"No {period} FY data for {ticker}, retrying with quarterly")
                params["period"] = "quarterly"
                resp = rate_limited_request(url, params)
                data = resp.json().get("results", [])
            if not data:
                raise ValueError("No financial data returned")
            data.sort(key=lambda x: x.get("end_date", ""), reverse=True)
            latest = data[0]
            financials = latest.get("financials", {})
            income = financials.get("income_statement", {})
            balance = financials.get("balance_sheet", {})
            eps = income.get("basic_earnings_per_share", {}).get("value", 0) or 0
            revenue = income.get("revenues", {}).get("value", 0) or 0
            cash_flow = financials.get("cash_flow_statement", {}).get("net_cash_flow", {}).get("value", 0) or 0
            net_income = income.get("net_income", {}).get("value", 0) or 0
            net_margin = (net_income / revenue * 100) if revenue != 0 else 0
            debt = balance.get("total_debt", {}).get("value", 0)
            equity = balance.get("total_equity", {}).get("value", 1)
            debt_to_equity = debt / equity if equity != 0 else float('inf')
            result = {
                "eps": eps,
                "revenue": revenue,
                "cash_flow": cash_flow,
                "net_margin": net_margin,
                "debt_to_equity": debt_to_equity
            }
            save_cache(result, cache_key)
            return pd.DataFrame([result])
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{retries} failed for {ticker} (Polygon.io): {e}")
            if attempt + 1 == retries:
                logger.error(f"Failed to fetch fundamentals for {ticker} after {retries} attempts")
                break
            time.sleep(RETRY_DELAY)

    logger.debug(f"No fundamental data for {ticker}, using sector averages")
    result = {
        "eps": SECTOR_AVG_GROWTH.get(sector, 5.0),
        "revenue": SECTOR_AVG_GROWTH.get(sector, 5.0),
        "cash_flow": SECTOR_AVG_GROWTH.get(sector, 5.0),
        "net_margin": SECTOR_AVG_NET_MARGIN.get(sector, 15.0),
        "debt_to_equity": SECTOR_AVG_DEBT_TO_EQUITY.get(sector, 1.0)
    }
    save_cache(result, cache_key)
    return pd.DataFrame([result])

def get_dividends(ticker: str, retries: int = 3) -> float:
    """Return default dividend yield since we only use Polygon.io and it may not provide this data."""
    if ticker in ESTIMATED_2024 or ticker in SIMULATED_2024:
        return 0.0
    logger.debug(f"No dividend data for {ticker}, defaulting to 0")
    return 0.0

def get_growth_metrics(ticker: str, fin_df: pd.DataFrame, sector: Optional[str], eps_val: Optional[float]) -> Dict[str, float]:
    """Extract growth metrics from financial data, with capped revenue_growth."""
    if ticker in KEY_TICKERS:
        logger.info(f"{ticker}: Applied ESTIMATED_2024 values - eps={ESTIMATED_2024[ticker]['eps_val']}, rev={ESTIMATED_2024[ticker].get('revenue_growth', 0)}, cash={ESTIMATED_2024[ticker].get('cash_flow_growth', 0)}")
        return {
            "eps_val": ESTIMATED_2024[ticker]["eps_val"],
            "revenue_growth": ESTIMATED_2024[ticker].get("revenue_growth", 0),
            "cash_flow_growth": ESTIMATED_2024[ticker].get("cash_flow_growth", 0)
        }
    if ticker in SIMULATED_2024:
        logger.info(f"{ticker}: Applied SIMULATED_2024 values - eps={SIMULATED_2024[ticker]['eps_growth']}, rev={SIMULATED_2024[ticker]['revenue_growth']}, cash={SIMULATED_2024[ticker]['cash_flow_growth']}")
        return {
            "eps_val": SIMULATED_2024[ticker]["eps_growth"],
            "revenue_growth": SIMULATED_2024[ticker]["revenue_growth"],
            "cash_flow_growth": SIMULATED_2024[ticker]["cash_flow_growth"]
        }

    fin_data = fin_df.iloc[0]
    eps = fin_data.get("eps", 0)
    revenue = fin_data.get("revenue", 0)
    cash_flow = fin_data.get("cash_flow", 0)

    if eps_val is not None and eps == 0:
        logger.info(f"{ticker}: Used eps_val={eps_val} from current_universe.csv for eps_growth, capped at {eps_val}")
        eps_growth = min(eps_val, eps_val)
    else:
        eps_growth = eps

    revenue_growth = (revenue / 1e9) if revenue != 0 else SECTOR_AVG_GROWTH.get(sector, 5.0)
    revenue_growth = min(revenue_growth, 100.0)  # Cap to prevent outliers
    cash_flow_growth = (cash_flow / 1e9) if cash_flow != 0 else SECTOR_AVG_GROWTH.get(sector, 5.0)

    if revenue_growth == 0:
        logger.info(f"{ticker}: Defaulted revenue_growth to sector average {SECTOR_AVG_GROWTH.get(sector, 5.0)}")
        revenue_growth = SECTOR_AVG_GROWTH.get(sector, 5.0)
    if cash_flow_growth == 0:
        logger.info(f"{ticker}: Defaulted cash_flow_growth to sector average {SECTOR_AVG_GROWTH.get(sector, 5.0)}")
        cash_flow_growth = SECTOR_AVG_GROWTH.get(sector, 5.0)

    return {
        "eps_val": eps_growth,
        "revenue_growth": revenue_growth,
        "cash_flow_growth": cash_flow_growth
    }

def get_valuation_ratios(ticker: str, fin_df: pd.DataFrame, last_close: float, sector: Optional[str]) -> Dict[str, float]:
    """Fetch or estimate valuation ratios using Polygon.io data or sector averages."""
    if ticker in KEY_TICKERS:
        return {
            "p_e": ESTIMATED_2024[ticker].get("p_e", 30.0),
            "net_margin": ESTIMATED_2024[ticker].get("net_margin", SECTOR_AVG_NET_MARGIN.get(sector, 15.0)),
            "debt_to_equity": ESTIMATED_2024[ticker].get("debt_to_equity", SECTOR_AVG_DEBT_TO_EQUITY.get(sector, 1.0)),
            "market_cap": ESTIMATED_2024[ticker]["market_cap"]
        }
    if ticker in SIMULATED_2024:
        return {
            "p_e": SIMULATED_2024[ticker]["p_e"],
            "net_margin": SIMULATED_2024[ticker]["net_margin"],
            "debt_to_equity": SIMULATED_2024[ticker]["debt_to_equity"],
            "market_cap": SIMULATED_2024[ticker]["market_cap"]
        }

    fin_data = fin_df.iloc[0]
    net_margin = fin_data.get("net_margin", SECTOR_AVG_NET_MARGIN.get(sector, 15.0))
    debt_to_equity = fin_data.get("debt_to_equity", SECTOR_AVG_DEBT_TO_EQUITY.get(sector, 1.0))

    if net_margin == 0:
        logger.info(f"{ticker}: Defaulted net_margin to sector average {SECTOR_AVG_NET_MARGIN.get(sector, 15.0)}")
        net_margin = SECTOR_AVG_NET_MARGIN.get(sector, 15.0)
    if debt_to_equity == float('inf'):
        logger.info(f"{ticker}: Defaulted debt_to_equity to sector average {SECTOR_AVG_DEBT_TO_EQUITY.get(sector, 1.0)}")
        debt_to_equity = SECTOR_AVG_DEBT_TO_EQUITY.get(sector, 1.0)

    # Use Polygon.io ticker info for market cap, fallback to estimate
    ticker_formatted = ticker.replace(".", "-")
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker_formatted}"
    params = {"apiKey": POLYGON_API_KEY}
    cache_key = f"valuation_{ticker}.pkl"
    cached_data = load_cache(cache_key)
    if cached_data:
        logger.debug(f"Using cached valuation data for {ticker}")
        return cached_data

    try:
        resp = rate_limited_request(url, params)
        data = resp.json().get("results", {})
        market_cap = data.get("market_cap", last_close * 1e9)
        p_e = 30.0  # Default P/E since Polygon.io may not provide this directly
        result = {
            "p_e": p_e,
            "net_margin": net_margin,
            "debt_to_equity": debt_to_equity,
            "market_cap": market_cap
        }
        save_cache(result, cache_key)
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch valuation data for {ticker}: {e}, using defaults")
        result = {
            "p_e": 30.0,
            "net_margin": net_margin,
            "debt_to_equity": debt_to_equity,
            "market_cap": last_close * 1e9
        }
        save_cache(result, cache_key)
        return result

def get_sentiment_score(ticker: str, sector: Optional[str], last_close: float, sentiment_score_from_universe: Optional[float] = None, sentiment_start_date: str = None, sentiment_end_date: str = None, retries: int = 3) -> float:
    """Fetch sentiment score with heuristic fallback, prioritizing universe_builder.py's sentiment_score."""
    if ticker in KEY_TICKERS:
        return ESTIMATED_2024[ticker]["sentiment_score"]
    if ticker in SIMULATED_2024:
        return SIMULATED_2024[ticker]["sentiment_score"]

    if sentiment_score_from_universe is not None and 0 <= sentiment_score_from_universe <= 100:
        logger.info(f"{ticker}: Using sentiment_score={sentiment_score_from_universe} from current_universe.csv")
        return sentiment_score_from_universe

    cache_key = f"sentiment_{ticker}.pkl"
    cached_data = load_cache(cache_key)
    if cached_data is not None:
        logger.debug(f"Using cached sentiment for {ticker}")
        return cached_data

    params = {
        "ticker": ticker,
        "published_utc.gte": sentiment_start_date,
        "published_utc.lte": sentiment_end_date,
        "limit": 100,
        "apiKey": POLYGON_API_KEY
    }
    try:
        resp = rate_limited_request(POLYGON_SENTIMENT_URL, params)
        data = resp.json()
        if "results" not in data or not data["results"]:
            logger.warning(f"No sentiment data for {ticker} from Polygon.io")
            raise ValueError("No sentiment data")
        sentiment_sum = 0.0
        total_weight = 0.0
        article_count = len(data["results"])
        for r in data["results"]:
            title = r.get("title", "").lower()
            publisher = r.get("publisher", {}).get("name", "Unknown")
            weight = 0.7 * (1 + article_count / 50)
            pos_score = sum(1 for w in ["growth", "profit", "strong", "rise", "beat", "surge"] if w in title)
            neg_score = sum(-1 for w in ["decline", "loss", "weak", "fall", "miss", "drop"] if w in title)
            score = pos_score + neg_score
            sentiment_sum += score * weight
            total_weight += weight
        if total_weight > 0:
            score = (sentiment_sum / total_weight + 1) * 50
            logger.debug(f"{ticker}: Sentiment score={score:.2f} (articles={article_count})")
            score = min(max(score, 0), 100)
            save_cache(score, cache_key)
            return score
        raise ValueError("No sentiment signal")
    except Exception as e:
        logger.warning(f"Failed to fetch sentiment for {ticker}: {e}, using heuristic fallback")
        fin_df = get_fundamentals(ticker, "annual")
        growth = get_growth_metrics(ticker, fin_df, sector, eps_val=None)
        eps_growth = growth.get("eps_val")
        valuation = get_valuation_ratios(ticker, fin_df, last_close, sector)
        p_e = valuation.get("p_e")
        market_cap = valuation.get("market_cap")
        
        if p_e is None:
            p_e = 30.0
            logger.debug(f"{ticker}: p_e was None, defaulted to {p_e}")
        if market_cap is None:
            market_cap = 500e9
            logger.debug(f"{ticker}: market_cap was None, defaulted to {market_cap}")
        if eps_growth is None:
            eps_growth = SECTOR_AVG_GROWTH.get(sector, 5.0)
            logger.debug(f"{ticker}: eps_growth was None, defaulted to sector average {eps_growth}")

        pe_avg = 30.0
        market_cap_avg = 500e9
        eps_growth_avg = 10.0
        pe_factor = (pe_avg - p_e) / pe_avg * 30
        market_cap_factor = (market_cap - market_cap_avg) / market_cap_avg * 25
        eps_growth_factor = (eps_growth - eps_growth_avg) / eps_growth_avg * 20
        adjusted_sentiment = 50.0 + pe_factor + market_cap_factor + eps_growth_factor
        logger.debug(f"{ticker}: Heuristic sentiment score: pe={p_e}, pe_factor={pe_factor:.2f}, market_cap={market_cap}, market_cap_factor={market_cap_factor:.2f}, eps_growth={eps_growth}, eps_growth_factor={eps_growth_factor:.2f}, adjusted={adjusted_sentiment:.2f}")
        score = min(max(adjusted_sentiment, 0), 100)
        save_cache(score, cache_key)
        return score

# ----------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------
def main():
    global LATEST_RANKINGS
    logger.info(f"Running fundamental_config.py standalone on Mac M4 Max")

    if not UNIVERSE_FILE.exists():
        logger.error(f"No universe file found at {UNIVERSE_FILE}")
        return
    universe_df = pd.read_csv(UNIVERSE_FILE)
    logger.info(f"Loaded data for {len(universe_df)} tickers from {UNIVERSE_FILE}")

    if "sector" not in universe_df.columns:
        logger.info(f"'sector' column not found in {UNIVERSE_FILE}, defaulting to None")
        universe_df["sector"] = None

    tickers = universe_df["ticker"].tolist()
    logger.info(f"Loaded {len(tickers)} tickers")

    for ticker in tickers:
        if pd.isna(universe_df.loc[universe_df["ticker"] == ticker, "sector"].iloc[0]):
            sector = SECTOR_MAPPING.get(ticker, None)
            universe_df.loc[universe_df["ticker"] == ticker, "sector"] = sector
            logger.info(f"{ticker}: Inferred sector {sector} from mapping")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for _, row in universe_df.iterrows():
            ticker = row["ticker"]
            sector = row["sector"]
            price_performance = row.get("momentum_6m", 0) * 100
            eps_val = row.get("eps_val", None)
            last_close = row.get("close", 100.0)
            sentiment_score = row.get("sentiment_score", None)
            rsi = row.get("rsi", 50.0)
            macd_signal = row.get("macd_signal", 0)
            macd_histogram = row.get("macd_histogram", 0.0)
            bbands_width = row.get("bbands_width", 0.05)
            sma_trend = row.get("sma_trend", 0)
            stoch_k = row.get("stoch_k", 50.0)
            atr = row.get("atr", 0.0)
            vwap = row.get("vwap", 0.0)
            market_cap_from_universe = row.get("market_cap", None)
            logger.info(f"Analyzing {ticker} (sector={sector})")
            futures.append(executor.submit(
                lambda t=ticker, s=sector, pp=price_performance, ev=eps_val, lc=last_close, ss=sentiment_score, mc=market_cap_from_universe,
                       rsi=rsi, macd_signal=macd_signal, macd_histogram=macd_histogram, bbands_width=bbands_width, sma_trend=sma_trend, stoch_k=stoch_k, atr=atr, vwap=vwap: {
                    "ticker": t,
                    "sector": s,
                    "price_data": get_price_data(t, s, pp, mc, None),
                    "growth_metrics": get_growth_metrics(t, get_fundamentals(t, "annual"), s, ev),
                    "dividend_yield": get_dividends(t),
                    "valuation_ratios": get_valuation_ratios(t, get_fundamentals(t, "annual"), lc, s),
                    "sentiment_score": get_sentiment_score(t, s, lc, ss),
                    "portfolio_return": SIMULATED_2024.get(t, {}).get("portfolio_return", pp / 100),
                    "rsi": rsi,
                    "macd_signal": macd_signal,
                    "macd_histogram": macd_histogram,
                    "bbands_width": bbands_width,
                    "sma_trend": sma_trend,
                    "stoch_k": stoch_k,
                    "atr": atr,
                    "vwap": vwap
                }
            ))
        results = [future.result() for future in futures]

    data = []
    for res in results:
        ticker = res["ticker"]
        sector = res["sector"]
        price_data = res["price_data"]
        growth = res["growth_metrics"]
        dividend_yield = res["dividend_yield"]
        valuation = res["valuation_ratios"]
        sentiment_score = res["sentiment_score"]
        portfolio_return = res["portfolio_return"]
        rsi = res["rsi"]
        macd_signal = res["macd_signal"]
        macd_histogram = res["macd_histogram"]
        bbands_width = res["bbands_width"]
        sma_trend = res["sma_trend"]
        stoch_k = res["stoch_k"]
        atr = res["atr"]
        vwap = res["vwap"]

        if isinstance(sentiment_score, dict):
            logger.warning(f"{ticker}: sentiment_score is a dict {sentiment_score}, extracting 'score' if available")
            sentiment_score = sentiment_score.get('score', 0.0)
        logger.debug(f"{ticker}: sentiment_score={sentiment_score}, type={type(sentiment_score)}")

        weights = SECTOR_WEIGHTS.get(sector, SECTOR_WEIGHTS[None])
        logger.info(f"{ticker}: Applied weights {weights} (sector={sector})")

        raw_score = (
            weights["price_performance"] * price_data["price_performance"] +
            weights["momentum"] * price_data["momentum"] +
            weights["volatility"] * price_data["volatility"] +
            weights["eps_growth"] * growth["eps_val"] +
            weights["revenue_growth"] * growth["revenue_growth"] +
            weights["cash_flow_growth"] * growth["cash_flow_growth"] +
            weights["dividend_yield"] * dividend_yield +
            weights["net_margin"] * valuation["net_margin"] +
            weights["debt_to_equity"] * (1 / (valuation["debt_to_equity"] + 1)) +
            weights["sentiment_score"] * sentiment_score +
            weights["rsi"] * (100 - rsi) +
            weights["macd_signal"] * macd_signal +
            weights["macd_histogram"] * macd_histogram +
            weights["bbands_width"] * bbands_width +
            weights["sma_trend"] * sma_trend +
            weights["stoch_k"] * stoch_k +
            weights["atr"] * atr +
            weights["vwap"] * vwap
        )

        data.append({
            "ticker": ticker,
            "skip": False,
            "sector": sector,
            "price_performance": price_data["price_performance"],
            "momentum": price_data["momentum"],
            "volatility": price_data["volatility"],
            "eps_growth": growth["eps_val"],
            "revenue_growth": growth["revenue_growth"],
            "cash_flow_growth": growth["cash_flow_growth"],
            "dividend_yield": dividend_yield,
            "net_margin": valuation["net_margin"],
            "p_e": valuation["p_e"],
            "debt_to_equity": valuation["debt_to_equity"],
            "market_cap": valuation["market_cap"],
            "sentiment_score": sentiment_score,
            "missing_count": 0.0,
            "portfolio_return": portfolio_return,
            "raw_score": raw_score,
            "rsi": rsi,
            "macd_signal": macd_signal,
            "macd_histogram": macd_histogram,
            "bbands_width": bbands_width,
            "sma_trend": sma_trend,
            "stoch_k": stoch_k,
            "atr": atr,
            "vwap": vwap
        })

    df = pd.DataFrame(data)
    df = df.sort_values("raw_score", ascending=False).reset_index(drop=True)

    # Enforce key ticker ranks by reordering
    key_ticker_df = df[df["ticker"].isin(KEY_TICKERS)].copy()
    non_key_ticker_df = df[~df["ticker"].isin(KEY_TICKERS)].copy()

    # Sort key tickers according to the predefined order
    key_ticker_df = key_ticker_df.set_index("ticker").loc[["NVDA", "AAPL", "AVGO"]].reset_index()
    # Adjust scores to ensure order
    for idx, ticker in enumerate(["NVDA", "AAPL", "AVGO"]):
        target_rank = idx + 1
        current_rank = key_ticker_df.index[key_ticker_df["ticker"] == ticker].tolist()[0] + 1
        if current_rank != target_rank:
            logger.warning(f"{ticker} rank {current_rank} does not match target rank {target_rank}, adjusting score")
            # Assign a score that ensures the order
            target_score = df["raw_score"].max() + (3 - idx) * 10  # Higher rank gets higher score
            key_ticker_df.loc[key_ticker_df["ticker"] == ticker, "raw_score"] = target_score
            logger.info(f"{ticker}: Adjusted score to {target_score:.4f} to achieve rank {target_rank}")

    # Apply sector cap for Technology stocks
    final_df = key_ticker_df.copy()  # Start with key tickers (already 3 Technology stocks)
    tech_count = sum(final_df["sector"] == "Technology")  # Should be 3 (NVDA, AAPL, AVGO)
    remaining_slots = TOP_N - len(final_df)  # 7 slots left

    # Add non-Technology stocks first
    non_tech_df = non_key_ticker_df[non_key_ticker_df["sector"] != "Technology"].copy()
    non_tech_df = non_tech_df.sort_values("raw_score", ascending=False)
    non_tech_to_add = non_tech_df.head(remaining_slots)
    final_df = pd.concat([final_df, non_tech_to_add], ignore_index=True)

    # Update tech count and remaining slots
    tech_count = sum(final_df["sector"] == "Technology")
    remaining_slots = TOP_N - len(final_df)

    # If we still have slots to fill, add remaining stocks while respecting the sector cap
    if remaining_slots > 0:
        remaining_df = non_key_ticker_df[~non_key_ticker_df["ticker"].isin(final_df["ticker"])].copy()
        remaining_df = remaining_df.sort_values("raw_score", ascending=False)
        
        # First, try to add non-Technology stocks
        remaining_non_tech = remaining_df[remaining_df["sector"] != "Technology"].copy()
        remaining_non_tech_to_add = remaining_non_tech.head(remaining_slots)
        final_df = pd.concat([final_df, remaining_non_tech_to_add], ignore_index=True)

        # Update remaining slots
        remaining_slots = TOP_N - len(final_df)

        # If we still have slots and can add more Technology stocks without exceeding the cap
        if remaining_slots > 0 and tech_count < MAX_TECH_STOCKS:
            remaining_tech = remaining_df[remaining_df["sector"] == "Technology"].copy()
            max_tech_to_add = min(MAX_TECH_STOCKS - tech_count, remaining_slots)
            remaining_tech_to_add = remaining_tech.head(max_tech_to_add)
            final_df = pd.concat([final_df, remaining_tech_to_add], ignore_index=True)

        # Update remaining slots again
        remaining_slots = TOP_N - len(final_df)

        # If there are still slots to fill, relax the sector cap and add the highest-scoring remaining stocks
        if remaining_slots > 0:
            remaining_to_add = remaining_df[~remaining_df["ticker"].isin(final_df["ticker"])].head(remaining_slots)
            final_df = pd.concat([final_df, remaining_to_add], ignore_index=True)

    # Ensure we have exactly TOP_N stocks without re-sorting
    df_final = final_df.head(TOP_N).copy()

    # Normalize scores
    min_score = df_final["raw_score"].min()
    max_score = df_final["raw_score"].max()
    if max_score != min_score:
        df_final.loc[:, "final_score"] = (df_final["raw_score"] - min_score) / (max_score - min_score) * 100
    else:
        df_final.loc[:, "final_score"] = 100.0

    logger.info("Scores after normalization:")
    for _, row in df_final.iterrows():
        logger.info(f"{row['ticker']}: raw_score={row['raw_score']:.4f}, final_score={row['final_score']:.4f}")

    logger.info(f"Top {TOP_N} tickers:")
    for idx, row in df_final.iterrows():
        logger.info(f"Rank {idx+1}: {row['ticker']}: raw_score={row['raw_score']:.4f}, final_score={row['final_score']:.4f}, eps={row['eps_growth']:.1f}, rev={row['revenue_growth']:.1f}, margin={row['net_margin']:.1f}, debt_to_equity={row['debt_to_equity']:.1f}, sentiment={row['sentiment_score']:.1f}, return={row['portfolio_return']:.4f}, rsi={row['rsi']:.1f}, macd_signal={row['macd_signal']:.1f}, atr={row['atr']:.1f}, vwap={row['vwap']:.1f}")

    output_file = BASE_DIR / "fundamental_rankings.csv"
    df_final.to_csv(output_file, index=False)
    logger.info(f"Ranking saved => {output_file} at {datetime.now()}")

    LATEST_RANKINGS = df_final

if __name__ == "__main__":
    clear_cache()
    main()