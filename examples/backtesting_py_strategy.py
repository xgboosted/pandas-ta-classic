import pandas as pd
import pandas_ta_classic as ta
from backtesting import Backtest, Strategy
from backtesting.test import GOOG
from backtesting.lib import crossover

def indicator_bridge(data, indicator_name, **kwargs):
    """
    Helper function to securely pass full OHLCV data from backtesting.py 
    into a pandas-ta-classic indicator.
    """
    # 1. Reconstruct the full OHLCV DataFrame
    df = pd.DataFrame({
        'Open': data.Open,
        'High': data.High,
        'Low': data.Low,
        'Close': data.Close,
        'Volume': data.Volume
    })
    
    # 2. Security Validation: Prevent arbitrary string reflection
    if indicator_name not in dir(df.ta):
        raise ValueError(f"Indicator '{indicator_name}' not found in pandas-ta-classic.")
    
    # 3. Safely execute the indicator
    indicator_method = getattr(df.ta, indicator_name)
    result = indicator_method(**kwargs)
    
    # 4. Multi-output handler (Included for future-proofing other indicators)
    if isinstance(result, pd.DataFrame):
        return tuple(result[col].to_numpy() for col in result.columns)
            
    # 5. Single output handler (Used for SMA)
    return result.to_numpy()

class SMACrossover(Strategy):
    """
    A Simple Moving Average (SMA) crossover strategy utilizing pandas-ta-classic.
    """
    fast_length = 10
    slow_length = 20

    def init(self):
        # We pass the validated string 'sma'. The bridge handles OHLCV data securely.
        self.sma_fast = self.I(indicator_bridge, self.data, 'sma', length=self.fast_length)
        self.sma_slow = self.I(indicator_bridge, self.data, 'sma', length=self.slow_length)

    def next(self):
        # Buy when the fast SMA crosses above the slow SMA
        if crossover(self.sma_fast, self.sma_slow):
            if not self.position:
                self.buy(size=0.1) # 10% position sizing to prevent margin calls
        
        # Close the position when the slow SMA crosses back above the fast SMA
        elif crossover(self.sma_slow, self.sma_fast):
            if self.position:
                self.position.close()

if __name__ == '__main__':
    # Initialise and run the backtest with sample Google data
    bt = Backtest(GOOG, SMACrossover, cash=10000, commission=.002)
    stats = bt.run()
    
    print("\n--- SMA Crossover Performance ---")
    print(stats)
