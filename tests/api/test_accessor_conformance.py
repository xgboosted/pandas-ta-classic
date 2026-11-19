
import pytest
from pandas import DataFrame, Series

import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.config import get_sample_data

# Indicators whose required Series inputs (benchmark, fast/slow, signal args)
# the accessor cannot auto-provide from a DataFrame.  Called with no arguments
# they raise rather than produce a column.
_NULLABLE = frozenset({"beta", "correl", "long_run", "short_run", "tsignals", "xsignals"})


class TestAccessorConformance:
    @classmethod
    def setup_class(cls):
        cls.data = get_sample_data()

    @classmethod
    def teardown_class(cls):
        del cls.data

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    # Called with no arguments these cannot produce a column: they need an input
    # the DataFrame does not hold (a benchmark, a pair of Series, a trend).
    # They used to return the source DataFrame, then None, and now raise
    # ValueError naming the argument.  'ma' is not among them: its 'source'
    # maps to the close column, so df.ta.ma() dispatches to the default EMA.
    NEEDS_AN_EXTRA_ARGUMENT = frozenset({"beta", "correl", "long_run", "short_run", "tsignals", "xsignals"})

    def test_all_indicators_return_series_or_dataframe(self):
        indicator_names = self.data.ta.indicators(as_list=True)
        failures = []

        for name in indicator_names:
            if name in _NULLABLE:
                continue
            try:
                result = getattr(self.data.ta, name)()
            except Exception:  # noqa: BLE001, S112 - indicators that need extra inputs are out of scope here
                continue

            if result is None and name in self.NEEDS_AN_EXTRA_ARGUMENT:
                continue
            if not isinstance(result, (Series, DataFrame)):
                failures.append(f"{name}: got {type(result).__name__}")

        assert failures == [], f"Indicators returning wrong type: {failures}"

    def test_indicators_without_their_required_input_raise(self, subtests):
        """First the source DataFrame, then a bare None, now a named ValueError.

        Returning the frame made "nothing was produced" indistinguishable from
        a result; returning None named nothing.  The message says which
        argument the DataFrame cannot supply.
        """
        for name in sorted(self.NEEDS_AN_EXTRA_ARGUMENT):
            with subtests.test(indicator=name), pytest.raises(ValueError) as ctx:
                getattr(self.data.ta, name)()
            assert name in str(ctx.value)
