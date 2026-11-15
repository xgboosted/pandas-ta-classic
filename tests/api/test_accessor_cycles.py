import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.assertions import AccessorValueCase


class TestExtCyclesDeeper(AccessorValueCase):
    def test_ebsw_ext(self):
        df = self.fresh()
        df.ta.ebsw(append=True)
        self._check_last_col(df)
