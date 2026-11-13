"""Guard tests for tests/fixtures/exact_reference.py.

That module produces the golden values for the statistics group in
``expected_values.json``, but it only runs during the manual ``make fixtures``
step — CI would otherwise never execute it, and a break would stay invisible
until someone regenerated the fixtures and got a large, unexplained diff.

Two kinds of assertion here:

* **Alignment** — the exact reference must place its first defined value at the
  same index as ``pandas`` rolling and produce the same number of non-NaN
  values.  This is a hard equality: warmup length is integer bookkeeping, not
  floating-point, so there is nothing to tolerate.

* **Agreement** — the values must match ``pandas`` rolling within
  ``|exact - pandas| <= ATOL + RTOL * |pandas|``.  This is deliberately loose.
  It is a smoke test that catches a wrong formula or an off-by-one window,
  *not* an oracle: ``pandas`` rolling is the version-dependent implementation
  this module exists to replace.  Measured over the full SPY series on pandas
  3.0.3, its own divergence from exact arithmetic reaches 1.7e-5 absolute /
  4.7e-4 relative on ``kurt`` and 6.4e-8 / 2.4e-5 on ``skew`` — the tolerance
  has to sit above that.  An off-by-one window or a wrong constant moves these
  indicators by order 0.1 to 1, so the check still bites.  Never tighten this
  into an equality check.
"""

from unittest import TestCase

import numpy as np
from pandas import Series

from tests.config import get_sample_data
from tests.fixtures.exact_reference import (
    rolling_beta,
    rolling_entropy,
    rolling_kurtosis,
    rolling_mad,
    rolling_median,
    rolling_quantile,
    rolling_skew,
    rolling_ui,
    rolling_zscore,
)

# See the module docstring: sized to clear pandas rolling's own error, not to
# express how accurate the exact reference is.
SMOKE_RTOL = 1e-3
SMOKE_ATOL = 1e-4


def _pandas_entropy(close: Series, length: int) -> Series:
    def _ent(x: np.ndarray) -> float:
        x = np.abs(x)
        p = x / x.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    return close.rolling(length).apply(_ent, raw=True)


class TestExactReference(TestCase):
    @classmethod
    def setUpClass(cls):
        data = get_sample_data()
        data.columns = data.columns.str.lower()
        cls.open_ = data["open"]
        cls.close = data["close"]

    @classmethod
    def tearDownClass(cls):
        del cls.open_
        del cls.close

    def _compare(self, name: str, exact: Series, reference: Series) -> None:
        exact_ok = np.asarray(exact.notna())
        ref_ok = np.asarray(reference.notna())

        self.assertEqual(
            int(np.argmax(exact_ok)),
            int(np.argmax(ref_ok)),
            f"{name}: first defined index differs from pandas rolling",
        )
        self.assertEqual(
            int(exact_ok.sum()),
            int(ref_ok.sum()),
            f"{name}: non-NaN count differs from pandas rolling",
        )

        both = exact_ok & ref_ok
        self.assertTrue(both.any(), f"{name}: no overlapping defined values")
        e = np.asarray(exact, dtype=float)[both]
        r = np.asarray(reference, dtype=float)[both]
        budget = SMOKE_ATOL + SMOKE_RTOL * np.abs(r)
        overrun = np.abs(e - r) / budget
        worst = int(np.argmax(overrun))
        self.assertLessEqual(
            float(overrun[worst]),
            1.0,
            f"{name}: exact={e[worst]!r} vs pandas={r[worst]!r} exceeds the smoke tolerance",
        )

    def test_zscore(self):
        c = self.close
        self._compare("zscore", rolling_zscore(c, 20), (c - c.rolling(20).mean()) / c.rolling(20).std())

    def test_kurtosis(self):
        c = self.close
        self._compare("kurtosis", rolling_kurtosis(c, 20), c.rolling(20).kurt())

    def test_skew(self):
        c = self.close
        self._compare("skew", rolling_skew(c, 20), c.rolling(20).skew())

    def test_median(self):
        c = self.close
        self._compare("median", rolling_median(c, 14), c.rolling(14).median())

    def test_quantile(self):
        c = self.close
        self._compare("quantile", rolling_quantile(c, 14, 0.5), c.rolling(14).quantile(0.5))

    def test_mad(self):
        c = self.close
        reference = c.rolling(10).apply(lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True)
        self._compare("mad", rolling_mad(c, 10), reference)

    def test_entropy(self):
        c = self.close
        self._compare("entropy", rolling_entropy(c, 10), _pandas_entropy(c, 10))

    def test_beta(self):
        c, o = self.close, self.open_
        c_ret = c / c.shift(1) - 1
        o_ret = o / o.shift(1) - 1
        reference = c_ret.rolling(30).cov(o_ret) / o_ret.rolling(30).var()
        self._compare("beta", rolling_beta(c, o, 30), reference)

    def test_ui(self):
        c = self.close
        rmax = c.rolling(14).max()
        drawdown = (c - rmax) / rmax * 100
        reference = np.sqrt((drawdown**2).rolling(14).mean())
        self._compare("ui", rolling_ui(c, 14), reference)

    def test_exact_beats_pandas_on_kurtosis(self):
        """The whole point: the exact reference is closer to the true value.

        The final 20-close window's excess kurtosis, evaluated in exact
        rational arithmetic outside this module, is -0.10015487372143803.
        """
        truth = -0.10015487372143803
        c = self.close
        exact_err = abs(float(rolling_kurtosis(c, 20).iloc[-1]) - truth)
        pandas_err = abs(float(c.rolling(20).kurt().iloc[-1]) - truth)
        self.assertLess(exact_err, 1e-15, f"exact reference drifted from the true value by {exact_err:.3e}")
        self.assertGreater(
            pandas_err,
            exact_err,
            "pandas rolling matched the true value better than exact arithmetic — check the test data",
        )
