from unittest import TestCase
from unittest.mock import patch

import sys
import types

import pandas as pd
from pandas import DataFrame

import pandas_ta_classic.utils.data.alphavantage as av_module


class _FakeTimeSeries:
    def __init__(self, key=None, output_format=None):
        self.key = key
        self.output_format = output_format

    def get_daily(self, symbol=None, outputsize=None):
        df = DataFrame(
            {
                "1. open": [102.0, 101.0],
                "2. high": [103.0, 102.0],
                "3. low": [100.0, 99.0],
                "4. close": [101.5, 100.5],
                "5. volume": [2000, 1000],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-01"]),
        )
        return df, {"symbol": symbol, "outputsize": outputsize}


class TestAlphaVantageIntegration(TestCase):

    def test_av_supports_alpha_vantage_package(self):
        alpha_vantage_pkg = types.ModuleType("alpha_vantage")
        alpha_vantage_ts = types.ModuleType("alpha_vantage.timeseries")
        alpha_vantage_ts.TimeSeries = _FakeTimeSeries

        with patch.dict(
            sys.modules,
            {
                "alpha_vantage": alpha_vantage_pkg,
                "alpha_vantage.timeseries": alpha_vantage_ts,
            },
            clear=False,
        ):
            with patch.dict(av_module.Imports, {"alphaVantage-api": True}, clear=False):
                result = av_module.av(
                    "spy",
                    interval="D",
                    av_kwargs={"api_key": "test-key", "output_size": "full"},
                )

        self.assertIsInstance(result, DataFrame)
        self.assertFalse(result.empty)
        self.assertEqual(result.name, "SPY")
        self.assertListEqual(
            list(result.columns), ["Open", "High", "Low", "Close", "Volume"]
        )
        self.assertTrue(result.index.is_monotonic_increasing)

    def test_av_returns_empty_dataframe_when_dependency_unavailable(self):
        with patch.dict(av_module.Imports, {"alphaVantage-api": False}, clear=False):
            result = av_module.av("SPY")

        self.assertIsInstance(result, DataFrame)
        self.assertTrue(result.empty)
