"""
Priority 4 — Snapshot regression tests.

Each test re-computes an indicator on SPY_D.csv and compares spot-check
values at five fixed positional indices against stored golden snapshots:

    indices: 50, 200, 500, 1500, 3000

This catches algorithm regressions that only affect the interior of the
series (e.g. EMA initialisation, window boundary handling) — complementing
test_indicator_values.py which checks only the last value and non-NaN count.

Run the full suite:
    pytest tests/test_regression.py

Run only regression tests across the whole project:
    pytest -m regression
"""

import json
import math
from pathlib import Path
from unittest import TestCase

import pandas as pd
import pytest

import pandas_ta_classic as ta

# ---------------------------------------------------------------------------
# Load snapshots and sample data
# ---------------------------------------------------------------------------

_SNAP_PATH = Path(__file__).parent / "fixtures" / "regression_snapshots.json"
with open(_SNAP_PATH) as _fh:
    _SNAPSHOTS: dict[str, dict] = json.load(_fh)

_DATA_PATH = Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv"


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH, index_col="date", parse_dates=True)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    df.columns = df.columns.str.lower()
    return df


# ---------------------------------------------------------------------------
# Same indicator compute function re-used from generate_fixtures
# ---------------------------------------------------------------------------


def _compute_all(df: pd.DataFrame) -> dict[str, object]:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    return {
        "sma_20": ta.sma(c, length=20),
        "ema_20": ta.ema(c, length=20, talib=False),
        "dema_10": ta.dema(c, length=10, talib=False),
        "tema_10": ta.tema(c, length=10, talib=False),
        "wma_10": ta.wma(c, length=10, talib=False),
        "hma_10": ta.hma(c, length=10),
        "alma_10": ta.alma(c, length=10),
        "trima_10": ta.trima(c, length=10, talib=False),
        "hl2": ta.hl2(h, l),
        "hlc3": ta.hlc3(h, l, c, talib=False),
        "ohlc4": ta.ohlc4(o, h, l, c),
        "rsi_14": ta.rsi(c, length=14, talib=False),
        "macd_12_26_9": ta.macd(c, fast=12, slow=26, signal=9, talib=False),
        "stoch": ta.stoch(h, l, c),
        "cci_14": ta.cci(h, l, c, length=14, talib=False),
        "roc_10": ta.roc(c, length=10, talib=False),
        "willr_14": ta.willr(h, l, c, length=14, talib=False),
        "ao": ta.ao(h, l),
        "bop": ta.bop(o, h, l, c, talib=False),
        "mom_10": ta.mom(c, length=10, talib=False),
        "atr_14": ta.atr(h, l, c, length=14, talib=False),
        "bbands_20": ta.bbands(c, length=20, talib=False),
        "donchian_20": ta.donchian(h, l, lower_length=20, upper_length=20),
        "kc_20": ta.kc(h, l, c, length=20),
        "natr_14": ta.natr(h, l, c, length=14, talib=False),
        "true_range": ta.true_range(h, l, c, talib=False),
        "adx_14": ta.adx(h, l, c, length=14, talib=False),
        "aroon_14": ta.aroon(h, l, length=14, talib=False),
        "psar": ta.psar(h, l, c),
        "decreasing": ta.decreasing(c),
        "increasing": ta.increasing(c),
        "obv": ta.obv(c, v, talib=False),
        "mfi_14": ta.mfi(h, l, c, v, length=14, talib=False),
        "cmf_20": ta.cmf(h, l, c, v, length=20),
        "ad": ta.ad(h, l, c, v, talib=False),
        "stdev_20": ta.stdev(c, length=20, talib=False),
        "zscore_20": ta.zscore(c, length=20),
        "kurtosis_20": ta.kurtosis(c, length=20),
        "skew_20": ta.skew(c, length=20),
        "ebsw": ta.ebsw(c),
        "log_return": ta.log_return(c),
        "percent_return": ta.percent_return(c),
        "ha": ta.ha(o, h, l, c),
    }


# ---------------------------------------------------------------------------
# Tolerance — same as test_indicator_values.py
# ---------------------------------------------------------------------------

REL_TOL = 1e-4


def _approx_equal(actual: float, expected: float) -> bool:
    if expected == 0.0:
        return abs(actual) <= REL_TOL
    return abs(actual - expected) / abs(expected) <= REL_TOL


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestRegressionSnapshots(TestCase):
    """Spot-check indicator values at five interior positions across SPY_D.csv."""

    @classmethod
    def setUpClass(cls):
        cls.df = _load_data()
        cls.results = _compute_all(cls.df)

    @classmethod
    def tearDownClass(cls):
        del cls.df
        del cls.results

    def _check_snapshot(self, fixture_key: str) -> None:
        self.assertIn(
            fixture_key, self.results, f"No result computed for {fixture_key!r}"
        )
        result = self.results[fixture_key]
        self.assertIsNotNone(result, f"{fixture_key!r} returned None")

        if isinstance(result, pd.Series):
            result = result.to_frame(name=result.name)

        col_snaps = _SNAPSHOTS[fixture_key]

        for col, checkpoints in col_snaps.items():
            with self.subTest(col=col):
                self.assertIn(
                    col, result.columns, f"Column {col!r} missing from {fixture_key!r}"
                )
                series = result[col]

                for idx_str, expected_val in checkpoints.items():
                    idx = int(idx_str)
                    with self.subTest(idx=idx):
                        if expected_val is None:
                            # Snapshot was NaN — actual must also be NaN
                            actual = series.iloc[idx]
                            self.assertTrue(
                                pd.isna(actual),
                                f"{fixture_key!r}[{col!r}] at idx={idx}: "
                                f"snapshot was NaN but got {actual}",
                            )
                        else:
                            actual_raw = series.iloc[idx]
                            self.assertFalse(
                                pd.isna(actual_raw),
                                f"{fixture_key!r}[{col!r}] at idx={idx}: "
                                f"got NaN but snapshot has value {expected_val}",
                            )
                            actual = float(actual_raw)
                            self.assertFalse(
                                math.isnan(actual) or math.isinf(actual),
                                f"{fixture_key!r}[{col!r}] at idx={idx}: value is NaN/Inf",
                            )
                            self.assertTrue(
                                _approx_equal(actual, expected_val),
                                f"{fixture_key!r}[{col!r}] at idx={idx}: "
                                f"actual={actual:.8f} != snapshot={expected_val:.8f}",
                            )

    # ------------------------------------------------------------------
    # One test per fixture key (43 total)
    # ------------------------------------------------------------------

    # Overlap
    def test_sma_20(self):
        self._check_snapshot("sma_20")

    def test_ema_20(self):
        self._check_snapshot("ema_20")

    def test_dema_10(self):
        self._check_snapshot("dema_10")

    def test_tema_10(self):
        self._check_snapshot("tema_10")

    def test_wma_10(self):
        self._check_snapshot("wma_10")

    def test_hma_10(self):
        self._check_snapshot("hma_10")

    def test_alma_10(self):
        self._check_snapshot("alma_10")

    def test_trima_10(self):
        self._check_snapshot("trima_10")

    def test_hl2(self):
        self._check_snapshot("hl2")

    def test_hlc3(self):
        self._check_snapshot("hlc3")

    def test_ohlc4(self):
        self._check_snapshot("ohlc4")

    # Momentum
    def test_rsi_14(self):
        self._check_snapshot("rsi_14")

    def test_macd_12_26_9(self):
        self._check_snapshot("macd_12_26_9")

    def test_stoch(self):
        self._check_snapshot("stoch")

    def test_cci_14(self):
        self._check_snapshot("cci_14")

    def test_roc_10(self):
        self._check_snapshot("roc_10")

    def test_willr_14(self):
        self._check_snapshot("willr_14")

    def test_ao(self):
        self._check_snapshot("ao")

    def test_bop(self):
        self._check_snapshot("bop")

    def test_mom_10(self):
        self._check_snapshot("mom_10")

    # Volatility
    def test_atr_14(self):
        self._check_snapshot("atr_14")

    def test_bbands_20(self):
        self._check_snapshot("bbands_20")

    def test_donchian_20(self):
        self._check_snapshot("donchian_20")

    def test_kc_20(self):
        self._check_snapshot("kc_20")

    def test_natr_14(self):
        self._check_snapshot("natr_14")

    def test_true_range(self):
        self._check_snapshot("true_range")

    # Trend
    def test_adx_14(self):
        self._check_snapshot("adx_14")

    def test_aroon_14(self):
        self._check_snapshot("aroon_14")

    def test_psar(self):
        self._check_snapshot("psar")

    def test_decreasing(self):
        self._check_snapshot("decreasing")

    def test_increasing(self):
        self._check_snapshot("increasing")

    # Volume
    def test_obv(self):
        self._check_snapshot("obv")

    def test_mfi_14(self):
        self._check_snapshot("mfi_14")

    def test_cmf_20(self):
        self._check_snapshot("cmf_20")

    def test_ad(self):
        self._check_snapshot("ad")

    # Statistics
    def test_stdev_20(self):
        self._check_snapshot("stdev_20")

    def test_zscore_20(self):
        self._check_snapshot("zscore_20")

    def test_kurtosis_20(self):
        self._check_snapshot("kurtosis_20")

    def test_skew_20(self):
        self._check_snapshot("skew_20")

    # Cycles
    def test_ebsw(self):
        self._check_snapshot("ebsw")

    # Performance
    def test_log_return(self):
        self._check_snapshot("log_return")

    def test_percent_return(self):
        self._check_snapshot("percent_return")

    # Candles
    def test_ha(self):
        self._check_snapshot("ha")
