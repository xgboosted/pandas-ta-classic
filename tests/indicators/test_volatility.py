
import pytest

import pandas_ta_classic as pandas_ta
from tests.config import get_sample_data


class TestVolatility:
    @classmethod
    def setup_class(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    @classmethod
    def teardown_class(cls):
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        if hasattr(cls, "volume"):
            del cls.volume
        del cls.data

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_cvi(self):
        assert pandas_ta.cvi(self.high, None) is None

    def test_rvi_mode_arguments_fail_fast(self, subtests):
        # refined/thirds used to be bool()-coerced, so rvi(refined="yes") ran
        # the refined branch without a word.
        for kwarg in ("refined", "thirds"):
            with subtests.test(kwarg=kwarg), pytest.raises(ValueError, match=rf"rvi\(\) {kwarg} must be True or False"):
                pandas_ta.rvi(self.close, self.high, self.low, **{kwarg: "yes"})

        # They are alternative modes; refined used to win silently.
        with pytest.raises(ValueError, match="alternative modes"):
            pandas_ta.rvi(self.close, self.high, self.low, refined=True, thirds=True)

        # Both modes are defined over high/low. Omitting them used to return
        # None, which reads as an ordinary short-input result.
        for kwarg in ("refined", "thirds"):
            with subtests.test(kwarg=kwarg), pytest.raises(ValueError, match=rf"rvi\(\) {kwarg}=True needs both high and low"):
                pandas_ta.rvi(self.close, **{kwarg: True})
        with pytest.raises(ValueError, match="needs both high and low"):
            pandas_ta.rvi(self.close, high=self.high, refined=True)
