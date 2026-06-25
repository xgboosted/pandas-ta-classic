import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import backtrader as bt

_ = ta.__name__  # registers df.ta accessor


def make_feed(df: pd.DataFrame, *extra_cols: str) -> type:
    """Return a PandasData subclass that exposes precomputed indicator columns as lines."""
    lines = tuple(extra_cols)
    params = tuple((col, -1) for col in extra_cols)

    return type(
        'PandasDataWithTA',
        (bt.feeds.PandasData,),
        {'lines': lines, 'params': params},
    )


class SMACrossover(bt.Strategy):
    params = (('fast_length', 10), ('slow_length', 20))

    def __init__(self):
        self.crossover = bt.indicators.CrossOver(self.data.sma_fast, self.data.sma_slow)
        self.order = None

    def next(self):
        if self.order:
            return
        if self.crossover > 0 and not self.position:
            self.order = self.buy(size=0.1)
        elif self.crossover < 0 and self.position:
            self.order = self.sell()

    def notify_order(self, order):
        if order.status in (order.Completed, order.Cancelled, order.Rejected):
            self.order = None


def synthetic_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.clip(close * (1 + rng.normal(0, 0.002, n)), low, high)
    return pd.DataFrame(
        {
            'Open': open_,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=pd.date_range('2020-01-01', periods=n, freq='B'),
    )


def run(fast_length: int = 10, slow_length: int = 20) -> None:
    df = synthetic_ohlcv()

    # Precompute indicators — pandas-ta-classic works over the full Series here,
    # not bar-by-bar, so all vectorized output is available before cerebro starts.
    df['sma_fast'] = ta.sma(df['Close'], length=fast_length)
    df['sma_slow'] = ta.sma(df['Close'], length=slow_length)
    df = df.dropna()

    FeedCls = make_feed(df, 'sma_fast', 'sma_slow')
    data = FeedCls(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(SMACrossover, fast_length=fast_length, slow_length=slow_length)
    cerebro.broker.setcash(10_000.0)
    cerebro.broker.setcommission(commission=0.002)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print(f"Starting portfolio: ${cerebro.broker.getvalue():,.2f}")
    results = cerebro.run()
    strat = results[0]
    print(f"Final portfolio:    ${cerebro.broker.getvalue():,.2f}")

    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', float('nan'))
    dd = strat.analyzers.drawdown.get_analysis().max.drawdown
    ret = strat.analyzers.returns.get_analysis()['rtot'] * 100

    print(f"\n--- SMA Crossover ({fast_length}/{slow_length}) ---")
    print(f"Total return:  {ret:.2f}%")
    print(f"Sharpe ratio:  {sharpe:.2f}")
    print(f"Max drawdown:  {dd:.2f}%")


if __name__ == '__main__':
    run()
