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

    def get_monthly(self, symbol=None):
        df = DataFrame(
            {
                "1. open": [200.0],
                "2. high": [201.0],
                "3. low": [199.0],
                "4. close": [200.5],
                "5. volume": [5000],
            },
            index=pd.to_datetime(["2024-01-01"]),
        )
        return df, {"symbol": symbol, "kind": "monthly"}

    def get_intraday(self, symbol=None, interval=None, outputsize=None):
        df = DataFrame(
            {
                "1. open": [302.0, 301.0],
                "2. high": [303.0, 302.0],
                "3. low": [300.0, 299.0],
                "4. close": [302.5, 301.5],
                "5. volume": [900, 800],
            },
            index=pd.to_datetime(["2024-01-01 09:31", "2024-01-01 09:30"]),
        )
        return df, {
            "symbol": symbol,
            "kind": "intraday",
            "interval": interval,
            "outputsize": outputsize,
        }


class TestAlphaVantageIntegration(TestCase):

    @staticmethod
    def _install_alpha_vantage_mock():
        alpha_vantage_pkg = types.ModuleType("alpha_vantage")
        alpha_vantage_ts = types.ModuleType("alpha_vantage.timeseries")
        alpha_vantage_ts.TimeSeries = _FakeTimeSeries
        return patch.dict(
            sys.modules,
            {
                "alpha_vantage": alpha_vantage_pkg,
                "alpha_vantage.timeseries": alpha_vantage_ts,
            },
            clear=False,
        )

    @staticmethod
    def _spy_monthly_and_intraday():
        calls = {"monthly": 0, "intraday": 0}
        original_monthly = _FakeTimeSeries.get_monthly
        original_intraday = _FakeTimeSeries.get_intraday

        def monthly_side_effect(instance, *args, **kwargs):
            calls["monthly"] += 1
            return original_monthly(instance, *args, **kwargs)

        def intraday_side_effect(instance, *args, **kwargs):
            calls["intraday"] += 1
            return original_intraday(instance, *args, **kwargs)

        monthly_patch = patch.object(
            _FakeTimeSeries,
            "get_monthly",
            autospec=True,
            side_effect=monthly_side_effect,
        )
        intraday_patch = patch.object(
            _FakeTimeSeries,
            "get_intraday",
            autospec=True,
            side_effect=intraday_side_effect,
        )

        return calls, monthly_patch, intraday_patch

    def test_av_supports_alpha_vantage_package(self):
        with self._install_alpha_vantage_mock():
            with patch.dict(av_module.Imports, {"alpha-vantage": True}, clear=False):
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

    def test_av_1m_uses_intraday_not_monthly(self):
        with self._install_alpha_vantage_mock():
            with patch.dict(av_module.Imports, {"alpha-vantage": True}, clear=False):
                calls, monthly_patch, intraday_patch = self._spy_monthly_and_intraday()
                with monthly_patch, intraday_patch:
                    result = av_module.av(
                        "spy",
                        interval="1m",
                        av_kwargs={"api_key": "test-key", "output_size": "compact"},
                    )

        self.assertIsInstance(result, DataFrame)
        self.assertFalse(result.empty)
        self.assertEqual(result.name, "SPY")
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertEqual(calls["monthly"], 0)
        self.assertEqual(calls["intraday"], 1)

    def test_av_monthly_alias_uses_monthly_endpoint(self):
        with self._install_alpha_vantage_mock():
            with patch.dict(av_module.Imports, {"alpha-vantage": True}, clear=False):
                calls, monthly_patch, intraday_patch = self._spy_monthly_and_intraday()
                with monthly_patch, intraday_patch:
                    result = av_module.av(
                        "spy",
                        interval="monthly",
                        av_kwargs={"api_key": "test-key", "output_size": "full"},
                    )

        self.assertIsInstance(result, DataFrame)
        self.assertFalse(result.empty)
        self.assertEqual(result.name, "SPY")
        self.assertEqual(calls["monthly"], 1)
        self.assertEqual(calls["intraday"], 0)

    def test_av_returns_empty_dataframe_when_dependency_unavailable(self):
        with patch.dict(av_module.Imports, {"alpha-vantage": False}, clear=False):
            result = av_module.av("SPY")

        self.assertIsInstance(result, DataFrame)
        self.assertTrue(result.empty)
