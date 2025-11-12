# -*- coding: utf-8 -*-
"""
Performance Benchmarks for Numba-optimized Indicators

This module contains benchmark tests to measure the performance improvements
gained from Numba JIT compilation.

Run with: pytest tests/test_performance_benchmarks.py -v
Or with: pytest tests/test_performance_benchmarks.py --benchmark-only
"""

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

# Import indicators to benchmark
from pandas_ta_classic.momentum import fisher, qqe, rsx, stc
from pandas_ta_classic.overlap import supertrend
from pandas_ta_classic.trend import psar
from pandas_ta_classic.utils._numba import NUMBA_AVAILABLE, get_numba_status


# Fixtures for test data
@pytest.fixture(scope="module")
def sample_data_small():
    """Generate small dataset (100 rows) for quick tests"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    data = {
        "open": pd.Series(100 + np.cumsum(np.random.randn(n) * 2), index=dates),
        "high": pd.Series(102 + np.cumsum(np.random.randn(n) * 2), index=dates),
        "low": pd.Series(98 + np.cumsum(np.random.randn(n) * 2), index=dates),
        "close": pd.Series(100 + np.cumsum(np.random.randn(n) * 2), index=dates),
    }
    return data


@pytest.fixture(scope="module")
def sample_data_medium():
    """Generate medium dataset (1000 rows) for realistic tests"""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    data = {
        "open": pd.Series(100 + np.cumsum(np.random.randn(n) * 2), index=dates),
        "high": pd.Series(102 + np.cumsum(np.random.randn(n) * 2), index=dates),
        "low": pd.Series(98 + np.cumsum(np.random.randn(n) * 2), index=dates),
        "close": pd.Series(100 + np.cumsum(np.random.randn(n) * 2), index=dates),
    }
    return data


@pytest.fixture(scope="module")
def sample_data_large():
    """Generate large dataset (10000 rows) for stress tests"""
    np.random.seed(42)
    n = 10000
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    data = {
        "open": pd.Series(100 + np.cumsum(np.random.randn(n) * 2), index=dates),
        "high": pd.Series(102 + np.cumsum(np.random.randn(n) * 2), index=dates),
        "low": pd.Series(98 + np.cumsum(np.random.randn(n) * 2), index=dates),
        "close": pd.Series(100 + np.cumsum(np.random.randn(n) * 2), index=dates),
    }
    return data


class TestNumbaStatus:
    """Test Numba availability and configuration"""

    def test_numba_status(self):
        """Test that get_numba_status returns valid information"""
        status = get_numba_status()
        assert isinstance(status, dict)
        assert "available" in status
        assert "version" in status
        assert isinstance(status["available"], bool)

    def test_numba_available_constant(self):
        """Test that NUMBA_AVAILABLE is set correctly"""
        assert isinstance(NUMBA_AVAILABLE, bool)


class TestRSXPerformance:
    """Benchmark tests for RSX indicator with and without Numba"""

    def test_rsx_correctness_small(self, sample_data_small):
        """Test that Numba and pure Python implementations produce identical results"""
        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available, skipping comparison test")

        # Calculate with Numba
        result_numba = rsx(sample_data_small["close"], length=14, use_numba=True)

        # Calculate without Numba
        result_python = rsx(sample_data_small["close"], length=14, use_numba=False)

        # Results should be very close (allowing for floating point differences)
        assert result_numba is not None
        assert result_python is not None
        pd.testing.assert_series_equal(
            result_numba, result_python, check_names=False, atol=1e-10
        )

    def test_rsx_small_dataset(self, sample_data_small, benchmark):
        """Benchmark RSX on small dataset (100 rows)"""
        result = benchmark(rsx, sample_data_small["close"], length=14)
        assert result is not None
        assert len(result) == len(sample_data_small["close"])

    def test_rsx_medium_dataset(self, sample_data_medium, benchmark):
        """Benchmark RSX on medium dataset (1000 rows)"""
        result = benchmark(rsx, sample_data_medium["close"], length=14)
        assert result is not None
        assert len(result) == len(sample_data_medium["close"])

    def test_rsx_large_dataset(self, sample_data_large, benchmark):
        """Benchmark RSX on large dataset (10000 rows)"""
        result = benchmark(rsx, sample_data_large["close"], length=14)
        assert result is not None
        assert len(result) == len(sample_data_large["close"])

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_rsx_with_numba(self, sample_data_large, benchmark):
        """Benchmark RSX with Numba enabled"""
        result = benchmark(rsx, sample_data_large["close"], length=14, use_numba=True)
        assert result is not None

    def test_rsx_without_numba(self, sample_data_large, benchmark):
        """Benchmark RSX with Numba disabled (pure Python)"""
        result = benchmark(rsx, sample_data_large["close"], length=14, use_numba=False)
        assert result is not None


class TestRSXManualBenchmark:
    """Manual timing tests for RSX to show performance differences"""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_rsx_performance_comparison(self, sample_data_large):
        """Compare RSX performance with and without Numba"""
        import time

        # Warm up Numba (compilation happens on first call)
        _ = rsx(sample_data_large["close"][:100], length=14, use_numba=True)

        # Time with Numba
        start = time.time()
        for _ in range(10):  # Run multiple times for better average
            result_numba = rsx(sample_data_large["close"], length=14, use_numba=True)
        time_numba = (time.time() - start) / 10

        # Time without Numba
        start = time.time()
        for _ in range(10):
            result_python = rsx(sample_data_large["close"], length=14, use_numba=False)
        time_python = (time.time() - start) / 10

        print(f"\n{'='*60}")
        print(f"RSX Performance Comparison (10000 rows, 10 iterations)")
        print(f"{'='*60}")
        print(f"Pure Python: {time_python*1000:.2f} ms per call")
        print(f"With Numba:  {time_numba*1000:.2f} ms per call")
        print(f"Speedup:     {time_python/time_numba:.2f}x faster")
        print(f"{'='*60}")

        # Numba should be faster
        assert time_numba < time_python, "Numba should be faster than pure Python"

    def test_rsx_output_validity(self, sample_data_medium):
        """Test that RSX output is valid (not NaN except for warmup period)"""
        result = rsx(sample_data_medium["close"], length=14)
        assert result is not None

        # First 13 values should be NaN (warmup period for length=14)
        assert result.iloc[:13].isna().all()

        # Remaining values should be valid numbers between 0 and 100
        valid_values = result.iloc[13:]
        assert valid_values.notna().all()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()


class TestFisherPerformance:
    """Benchmark tests for Fisher Transform indicator"""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_fisher_correctness(self, sample_data_small):
        """Test that Numba and pure Python implementations produce identical results"""
        result_numba = fisher(
            sample_data_small["high"], sample_data_small["low"], use_numba=True
        )
        result_python = fisher(
            sample_data_small["high"], sample_data_small["low"], use_numba=False
        )
        pd.testing.assert_frame_equal(result_numba, result_python, atol=1e-10)

    def test_fisher_large_dataset(self, sample_data_large, benchmark):
        """Benchmark Fisher Transform on large dataset"""
        result = benchmark(fisher, sample_data_large["high"], sample_data_large["low"])
        assert result is not None
        assert len(result) == len(sample_data_large["close"])


class TestQQEPerformance:
    """Benchmark tests for QQE indicator"""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_qqe_correctness(self, sample_data_small):
        """Test that Numba and pure Python implementations produce identical results"""
        result_numba = qqe(sample_data_small["close"], use_numba=True)
        result_python = qqe(sample_data_small["close"], use_numba=False)
        pd.testing.assert_frame_equal(result_numba, result_python, atol=1e-10)

    def test_qqe_large_dataset(self, sample_data_large, benchmark):
        """Benchmark QQE on large dataset"""
        result = benchmark(qqe, sample_data_large["close"])
        assert result is not None
        assert len(result) == len(sample_data_large["close"])


class TestSupertrendPerformance:
    """Benchmark tests for Supertrend indicator"""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_supertrend_correctness(self, sample_data_small):
        """Test that Numba and pure Python implementations produce identical results"""
        result_numba = supertrend(
            sample_data_small["high"],
            sample_data_small["low"],
            sample_data_small["close"],
            use_numba=True,
        )
        result_python = supertrend(
            sample_data_small["high"],
            sample_data_small["low"],
            sample_data_small["close"],
            use_numba=False,
        )
        pd.testing.assert_frame_equal(result_numba, result_python, atol=1e-10)

    def test_supertrend_large_dataset(self, sample_data_large, benchmark):
        """Benchmark Supertrend on large dataset"""
        result = benchmark(
            supertrend,
            sample_data_large["high"],
            sample_data_large["low"],
            sample_data_large["close"],
        )
        assert result is not None
        assert len(result) == len(sample_data_large["close"])


class TestPSARPerformance:
    """Benchmark tests for PSAR indicator"""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_psar_correctness(self, sample_data_small):
        """Test that Numba and pure Python implementations produce identical results"""
        result_numba = psar(
            sample_data_small["high"], sample_data_small["low"], use_numba=True
        )
        result_python = psar(
            sample_data_small["high"], sample_data_small["low"], use_numba=False
        )
        pd.testing.assert_frame_equal(result_numba, result_python, atol=1e-10)

    def test_psar_large_dataset(self, sample_data_large, benchmark):
        """Benchmark PSAR on large dataset"""
        result = benchmark(psar, sample_data_large["high"], sample_data_large["low"])
        assert result is not None
        assert len(result) == len(sample_data_large["close"])


class TestSTCPerformance:
    """Benchmark tests for STC indicator"""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_stc_correctness(self, sample_data_small):
        """Test that Numba and pure Python implementations produce identical results"""
        result_numba = stc(sample_data_small["close"], use_numba=True)
        result_python = stc(sample_data_small["close"], use_numba=False)
        pd.testing.assert_frame_equal(result_numba, result_python, atol=1e-8)

    def test_stc_large_dataset(self, sample_data_large, benchmark):
        """Benchmark STC on large dataset"""
        result = benchmark(stc, sample_data_large["close"])
        assert result is not None
        assert len(result) == len(sample_data_large["close"])


# Module-level info function
def print_benchmark_info():
    """Print information about the benchmark environment"""
    status = get_numba_status()
    print("\n" + "=" * 60)
    print("Performance Benchmark Environment")
    print("=" * 60)
    print(f"Numba Available: {status['available']}")
    if status["available"]:
        print(f"Numba Version: {status['version']}")
        print("JIT-compiled optimizations will be tested")
    else:
        print("Pure Python/NumPy implementations will be tested")
        print("Install numba for performance comparisons: pip install numba")
    print("=" * 60)


if __name__ == "__main__":
    print_benchmark_info()
    print("\nRun with: pytest tests/test_performance_benchmarks.py -v")
    print(
        "Or with benchmarking: pytest tests/test_performance_benchmarks.py --benchmark-only"
    )
