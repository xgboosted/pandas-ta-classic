import shutil
import tempfile
from pathlib import Path

import manifoldbt as mbt
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from manifoldbt.expr import col, exo, lit, when
from manifoldbt.helpers import Interval, Slippage, time_range

_ = ta.__name__  # registers df.ta accessor


def synthetic_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.clip(close * (1 + rng.normal(0, 0.002, n)), low, high)
    return pd.DataFrame(
        {
            'timestamp': pd.date_range('2020-01-01', periods=n, freq='D', tz='UTC'),
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': rng.integers(1_000_000, 10_000_000, n).astype(float),
        }
    )


def run(fast_length: int = 10, slow_length: int = 20) -> None:
    df = synthetic_ohlcv()

    # Precompute indicators - pandas-ta-classic works over the full Series here,
    # so every value is available before the engine starts. manifoldbt evaluates
    # expressions over whole columns rather than bar by bar, so the indicators are
    # handed over as a data series instead of being recomputed inside a callback.
    signals = pd.DataFrame(
        {
            'timestamp': df['timestamp'],
            'sma_fast': ta.sma(df['close'], length=fast_length),
            'sma_slow': ta.sma(df['close'], length=slow_length),
        }
    ).dropna()

    # The store keeps its metadata database open for the lifetime of the process,
    # so the directory is removed with errors ignored rather than by a context
    # manager, which would fail on Windows.
    tmp = tempfile.mkdtemp()
    try:
        root = Path(tmp)
        store = mbt.import_dataframe(
            df,
            symbol='SYNTH',
            symbol_id=1,
            interval='1d',
            data_root=str(root / 'data'),
            metadata_db=str(root / 'metadata.sqlite'),
            exchange='dataframe',
            asset_class='equity',
        )
        # Exogenous series: the bridge that carries pandas-ta-classic output into
        # the engine. Referenced in expressions as exo('pandas_ta', '<column>').
        mbt.register_exo('pandas_ta', signals, store=store)

        position = when(
            exo('pandas_ta', 'sma_fast') > exo('pandas_ta', 'sma_slow'),
            lit(1.0),
            lit(0.0),
        )
        strategy = mbt.Strategy.create('sma-crossover').signal('pos', position).size(col('pos'))

        start, end = time_range('2020-01-01', '2021-06-01')
        config = mbt.BacktestConfig(
            universe=[1],
            time_range_start=start,
            time_range_end=end,
            bar_interval=Interval.hours(24),
            initial_capital=10_000.0,
            warmup_bars=slow_length,
            exo_data=['pandas_ta'],
            execution=mbt.ExecutionConfig(
                signal_delay=1,
                execution_price='AtClose',
                allow_short=False,
                position_sizing_mode='FractionOfEquity',
            ),
            fees=mbt.FeeConfig(taker_fee_bps=20.0, maker_fee_bps=20.0),
            slippage=Slippage.none(),
        )

        result = mbt.run(strategy, config, store)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    m = result.metrics
    print(f"\n--- SMA Crossover ({fast_length}/{slow_length}) ---")
    print(f"Total return:  {m['total_return'] * 100:.2f}%")
    print(f"Sharpe ratio:  {m['sharpe']:.2f}")
    print(f"Max drawdown:  {abs(m['max_drawdown']) * 100:.2f}%")
    print(f"Round trips:   {m['trade_stats']['round_trips']}")


if __name__ == '__main__':
    run()
