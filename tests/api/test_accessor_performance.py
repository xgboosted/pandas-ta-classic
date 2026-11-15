from tests.assertions import AccessorValueCase


class TestExtPerformanceDeeper(AccessorValueCase):
    def test_log_return_ext(self):
        df = self.fresh()
        df.ta.log_return(append=True)
        lr_cols = [c for c in df.columns if c.startswith("LOGRET")]
        self.assertTrue(len(lr_cols) >= 1, "No LOGRET column appended")
        self._check_appended(df, lr_cols[0])

    def test_percent_return_ext(self):
        df = self.fresh()
        df.ta.percent_return(append=True)
        pct_cols = [c for c in df.columns if c.startswith("PCTRET")]
        self.assertTrue(len(pct_cols) >= 1, "No PCTRET column appended")
        self._check_appended(df, pct_cols[0])
