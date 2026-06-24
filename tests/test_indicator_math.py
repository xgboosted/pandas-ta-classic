from tests.assertions import (
    assert_indicator_standard,
    assert_talib,
    IndicatorSpec,
)
from tests.config import get_sample_data
import pandas_ta_classic as pandas_ta

from unittest import TestCase
from pandas import DataFrame

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class TestMath(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        cls.volume = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open, cls.high, cls.low, cls.close, cls.volume, cls.data

    # -----------------------------------------------------------------------
    # Two-series operators
    # -----------------------------------------------------------------------

    def test_add(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.add,
                args=[self.close, self.open],
                expected_name="ADD",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.ADD(self.close, self.open))

    def test_sub(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.sub,
                args=[self.close, self.open],
                expected_name="SUB",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.SUB(self.close, self.open))

    def test_div(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.div,
                args=[self.close, self.open],
                expected_name="DIV",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.DIV(self.close, self.open))

    def test_mult(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.mult,
                args=[self.close, self.open],
                expected_name="MULT",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.MULT(self.close, self.open))

    # -----------------------------------------------------------------------
    # Rolling operators
    # -----------------------------------------------------------------------

    def test_rolling_max(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rolling_max,
                args=[self.close],
                expected_name="MAX_30",
                length_override=20,
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.MAX(self.close, 30))

    def test_rolling_min(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rolling_min,
                args=[self.close],
                expected_name="MIN_30",
                length_override=20,
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.MIN(self.close, 30))

    def test_rolling_sum(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rolling_sum,
                args=[self.close],
                expected_name="SUM_30",
                length_override=20,
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.SUM(self.close, 30))

    def test_maxindex(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.maxindex,
                args=[self.close],
                expected_name="MAXINDEX_30",
                length_override=20,
            ),
        )

    def test_minindex(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.minindex,
                args=[self.close],
                expected_name="MININDEX_30",
                length_override=20,
            ),
        )

    def test_minmax(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.minmax,
                args=[self.close],
                expected_name="MINMAX_30",
                expected_type=DataFrame,
                expected_columns=["MIN_30", "MAX_30"],
                length_override=20,
            ),
        )

    def test_minmaxindex(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.minmaxindex,
                args=[self.close],
                expected_name="MINMAXINDEX_30",
                expected_type=DataFrame,
                expected_columns=["MINIDX_30", "MAXIDX_30"],
                length_override=20,
            ),
        )

    # -----------------------------------------------------------------------
    # Math transforms
    # -----------------------------------------------------------------------

    def test_acos(self):
        # close prices are out of arccos domain [-1, 1]; result is all-NaN but not None
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.acos,
                args=[self.close],
                expected_name="ACOS",
            ),
        )

    def test_asin(self):
        # close prices are out of arcsin domain [-1, 1]; result is all-NaN but not None
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.asin,
                args=[self.close],
                expected_name="ASIN",
            ),
        )

    def test_atan(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.atan,
                args=[self.close],
                expected_name="ATAN",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.ATAN(self.close))

    def test_ceil(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ceil,
                args=[self.close],
                expected_name="CEIL",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.CEIL(self.close))

    def test_cos(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cos,
                args=[self.close],
                expected_name="COS",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.COS(self.close))

    def test_cosh(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cosh,
                args=[self.close],
                expected_name="COSH",
            ),
        )

    def test_exp(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.exp,
                args=[self.close],
                expected_name="EXP",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.EXP(self.close))

    def test_floor(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.floor,
                args=[self.close],
                expected_name="FLOOR",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.FLOOR(self.close))

    def test_ln(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ln,
                args=[self.close],
                expected_name="LN",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.LN(self.close))

    def test_log10(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.log10,
                args=[self.close],
                expected_name="LOG10",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.LOG10(self.close))

    def test_sin(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.sin,
                args=[self.close],
                expected_name="SIN",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.SIN(self.close))

    def test_sinh(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.sinh,
                args=[self.close],
                expected_name="SINH",
            ),
        )

    def test_sqrt(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.sqrt,
                args=[self.close],
                expected_name="SQRT",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.SQRT(self.close))

    def test_tan(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.tan,
                args=[self.close],
                expected_name="TAN",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.TAN(self.close))

    def test_tanh(self):
        result = assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.tanh,
                args=[self.close],
                expected_name="TANH",
            ),
        )
        if HAS_TALIB:
            assert_talib(self, result, talib.TANH(self.close))

    # -----------------------------------------------------------------------
    # tulipy extras
    # -----------------------------------------------------------------------

    def test_npabs(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.npabs,
                args=[self.close],
                expected_name="ABS",
            ),
        )

    def test_npround(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.npround,
                args=[self.close],
                expected_name="ROUND",
            ),
        )

    def test_trunc(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.trunc,
                args=[self.close],
                expected_name="TRUNC",
            ),
        )

    def test_todeg(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.todeg,
                args=[self.close],
                expected_name="TODEG",
            ),
        )

    def test_torad(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.torad,
                args=[self.close],
                expected_name="TORAD",
            ),
        )
