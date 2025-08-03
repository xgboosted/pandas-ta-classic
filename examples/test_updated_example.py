#!/usr/bin/env python3
"""
Test the updated example notebook loading functionality
"""

import datetime as dt
import random as rnd
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import mplfinance as mpf
import pandas_ta_classic as ta
from watchlist import colors

# Load sample data from local CSV file instead of external API
ticker = "SPY"
import os
# Load sample data (go up one directory to find data folder)
data_path = os.path.join('..', 'data', 'SPY_D.csv')
df = pd.read_csv(data_path, index_col='date', parse_dates=True)
# Clean up the data: lowercase columns and drop unnamed columns
df.columns = df.columns.str.lower()
df = df.drop(columns=[col for col in df.columns if 'unnamed' in col.lower()], errors='ignore')
# Set the name attribute for the DataFrame
df.name = ticker

# Test the functions from the notebook
def recent_bars(df, tf: str = "1y"):
    # All Data: 0, Last Four Years: 0.25, Last Two Years: 0.5, This Year: 1, Last Half Year: 2, Last Quarter: 4
    yearly_divisor = {"all": 0, "10y": 0.1, "5y": 0.2, "4y": 0.25, "3y": 1./3, "2y": 0.5, "1y": 1, "6mo": 2, "3mo": 4}
    yd = yearly_divisor[tf] if tf in yearly_divisor.keys() else 0
    return int(ta.RATE["TRADING_DAYS_PER_YEAR"] / yd) if yd > 0 else df.shape[0]

def ctitle(indicator_name, ticker="SPY", length=100):
    """Create a chart title"""
    return f"{indicator_name.upper()} • {ticker} • Last {length} bars"

# Test loading and basic functionality
recent_startdate = df.tail(recent_bars(df)).index[0]
recent_enddate = df.tail(recent_bars(df)).index[-1]
print(f"{ticker}{df.tail(recent_bars(df)).shape} from {recent_startdate} to {recent_enddate}")
print("Data loaded successfully!")
print(f"Columns: {list(df.columns)}")
print(f"Data shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Test basic TA functionality
print("\nTesting basic TA functionality...")
sma_test = df.ta.sma(length=20)
print(f"SMA calculated: {sma_test.name}")

print("\nAll tests passed!")