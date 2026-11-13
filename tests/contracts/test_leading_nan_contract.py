"""A leading NaN run is not data.

Chained input (one indicator's output fed into another) always starts with a
NaN warm-up run. For every registered indicator, the result on
``NaN * k ++ x`` must equal the result on ``x`` over the real bars: no value
may appear where the clean result has none, and no value may change.

Before this contract ``rsx`` stuck at 50.0 on every bar, ``ha`` returned all
NaN, ``aroon``/``maxindex``/``entropy`` published values on windows that were
still filling, and ``obv`` was offset by the first bar's volume.

Indicators are discovered through ``ta.Category``, so new ones are covered.
"""

import inspect
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.config import get_sample_data
from tests.contracts import OHLCV_INDICATORS as INDICATORS, call as _call, spy

# 1 is what diff(), shift(1) and returns hand on; 40 covers a long indicator warm-up.
# skip_leading_nan used to be checked with 40 only, so treating a run of 1 as clean
# input (start > 1 instead of start > 0) failed no test.
PADS = (1, 40)


@pytest.fixture(scope="module")
def clean():
    return spy(600)


def _padded(clean, pad):
    lead = pd.DataFrame(np.nan, index=pd.date_range(end=clean.index[0] - pd.Timedelta(days=1), periods=pad, freq="D"), columns=clean.columns)
    return pd.concat([lead, clean])


@pytest.mark.parametrize("pad", PADS)
@pytest.mark.parametrize("name", INDICATORS)
def test_leading_nan_run_does_not_change_the_result(name, pad, clean):
    padded = _padded(clean, pad)
    expected = _call(name, clean)
    got = _call(name, padded)
    assert got is not None and len(got) == len(padded), f"{name}: result has {None if got is None else len(got)} rows for {len(padded)}"
    got = got.iloc[pad:]
    assert list(got.columns) == list(expected.columns)
    x, y = got.to_numpy(float, na_value=np.nan), expected.to_numpy(float, na_value=np.nan)
    np.testing.assert_array_equal(np.isnan(x), np.isnan(y), err_msg=f"{name}: NaN pattern differs")
    np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=name)


# Columns that hold a value over bars with no data, pinned as strict xfails so a
# change either way is noticed.
_NO_DATA_EXCEPTIONS = {
    "cdl_doji": "KEPT: a candle pattern reads 0 where it cannot match, as TA-Lib's does (#282)",
    "cdl_pattern": "KEPT: a candle pattern reads 0 where it cannot match, as TA-Lib's does (#282)",
}


def _columns_with_values(result: pd.DataFrame) -> list[str]:
    return [str(c) for c in result.columns if pd.to_numeric(result[c], errors="coerce").notna().any()]


@pytest.mark.parametrize("name", INDICATORS)
def test_the_leading_nan_run_itself_reads_nan(name, clean, request):
    """No value over the padded bars: mfi read 0.0 there, nvi and pvi 1000, wad and decay 0.0."""
    exceptions = _NO_DATA_EXCEPTIONS | {"ichimoku": "KEPT: the Chikou span is close.shift(-kijun) and looks forward"}
    if name in exceptions:
        request.applymarker(pytest.mark.xfail(strict=True, reason=exceptions[name]))
    pad = PADS[-1]
    head = _call(name, _padded(clean, pad)).iloc[:pad]
    assert not _columns_with_values(head), f"{name}: values over {pad} bars of no data in {_columns_with_values(head)[:3]}"


@pytest.mark.parametrize("name", INDICATORS)
def test_an_all_nan_input_reads_nan(name, clean, request):
    """An input with no data reports no value.

    ``skip_leading_nan`` passed all-NaN input through, so a recursion published
    its seed: rsx 50.0, mfi, fisher, ebsw and psar 0.0, psar's af 0.02.
    """
    if name in _NO_DATA_EXCEPTIONS:
        request.applymarker(pytest.mark.xfail(strict=True, reason=_NO_DATA_EXCEPTIONS[name]))
    result = _call(name, clean * np.nan)
    assert not _columns_with_values(result), f"{name}: values from all-NaN input in {_columns_with_values(result)[:3]}"


# ---------------------------------------------------------------------------
# indicators that went all-NaN on a leading NaN run
# ---------------------------------------------------------------------------


class TestLeadingNanRun(TestCase):
    """Recursive indicators seeded from bar 0 returned all-NaN on chained input.

    A 30-bar NaN prefix -- what any indicator's output starts with -- made
    adosc, fisher, hwc, hwma, jma, kama, lrsi, macd, macdfix, mama, mcgd,
    ssf, tos_stdevall and vidya return NaN for every bar. mfi instead read the
    run as zero money flow and reported values from incomplete windows.
    """

    PREFIX = 30
    NAMES = ("adosc", "fisher", "hwc", "hwma", "jma", "kama", "lrsi", "macd", "macdfix", "mama", "mcgd", "mfi", "ssf", "tos_stdevall", "vidya")

    @classmethod
    def setUpClass(cls):
        data = get_sample_data().iloc[:600].reset_index(drop=True)
        cls.prefixed = data.copy()
        cls.prefixed.iloc[: cls.PREFIX] = np.nan
        cls.trimmed = data.iloc[cls.PREFIX :].reset_index(drop=True)

    def _call(self, name, frame):
        fn = getattr(ta, name)
        params = inspect.signature(fn).parameters
        cols = {p: frame["open" if p == "open_" else p] for p in ("high", "low", "close", "volume") if p in params}
        return fn(**cols)

    def test_prefix_result_equals_trimmed_result(self):
        for name in self.NAMES:
            with self.subTest(indicator=name):
                with_prefix = np.asarray(self._call(name, self.prefixed), dtype=float)[self.PREFIX :]
                without = np.asarray(self._call(name, self.trimmed), dtype=float)
                self.assertTrue(np.isfinite(with_prefix).any(), f"{name} is still all-NaN")
                np.testing.assert_allclose(with_prefix, without, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_result_keeps_index_name_and_category(self):
        result = ta.macd(self.prefixed["close"])
        self.assertTrue(result.index.equals(self.prefixed.index))
        self.assertEqual(result.name, "MACD_12_26_9")
        self.assertEqual(result.category, "momentum")

    def test_accessor_signature_unchanged(self):
        """The decorator must keep inspect.signature, which the df.ta wrapper builds on."""
        self.assertEqual(list(inspect.signature(ta.macd).parameters)[:4], ["close", "fast", "slow", "signal"])
        self.assertEqual(self.prefixed.ta.macd().shape, (600, 3))
