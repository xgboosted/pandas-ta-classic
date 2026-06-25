import pandas as pd
import pandas_ta_classic as ta
from backtesting import Backtest, Strategy
from backtesting.test import GOOG
from backtesting.lib import crossover

_ = ta.__name__  # registers df.ta accessor


def ta_bridge(data, indicator_fn):
    df = pd.DataFrame(
        {
            'Open': data.Open,
            'High': data.High,
            'Low': data.Low,
            'Close': data.Close,
            'Volume': data.Volume,
        }
    )
    result = indicator_fn(df)
    if isinstance(result, pd.DataFrame):
        return tuple(result[col].to_numpy() for col in result.columns)
    return result.to_numpy()


class SMACrossover(Strategy):
    fast_length = 10
    slow_length = 20

    def init(self):
        self.sma_fast = self.I(ta_bridge, self.data, lambda df: df.ta.sma(length=self.fast_length))
        self.sma_slow = self.I(ta_bridge, self.data, lambda df: df.ta.sma(length=self.slow_length))

    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            if not self.position:
                self.buy(size=0.1)
        elif crossover(self.sma_slow, self.sma_fast):
            if self.position:
                self.position.close()


if __name__ == '__main__':
    bt = Backtest(GOOG, SMACrossover, cash=10000, commission=0.002)
    stats = bt.run()

    print("\n--- SMA Crossover Performance ---")
    print(stats)
