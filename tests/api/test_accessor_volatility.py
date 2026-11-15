import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.assertions import AccessorValueCase


class TestExtVolatilityDeeper(AccessorValueCase):
    def test_atr_ext(self):
        df = self.fresh()
        df.ta.atr(append=True)
        atr_cols = [c for c in df.columns if c.startswith("ATR")]
        self.assertTrue(len(atr_cols) >= 1, "No ATR column appended")
        self._check_appended(df, atr_cols[0])
        # ATR must be non-negative
        valid = df[atr_cols[0]].dropna()
        self.assertTrue((valid >= 0).all(), "ATR has negative values")

    def test_chop_ext(self):
        df = self.fresh()
        df.ta.chop(append=True)
        self._check_appended(df, "CHOP_14_1_100")
        valid = df["CHOP_14_1_100"].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 100).all(), "CHOP out of [0,100]")

    def test_donchian_ext(self):
        df = self.fresh()
        df.ta.donchian(append=True)
        self._check_appended(df, "DCL_20_20")
        self._check_appended(df, "DCM_20_20")
        self._check_appended(df, "DCU_20_20")

    def test_kc_ext(self):
        df = self.fresh()
        df.ta.kc(append=True)
        kc_cols = [c for c in df.columns if c.startswith("KC")]
        self.assertTrue(len(kc_cols) >= 1, "No KC columns appended")
        for col in kc_cols:
            self._check_appended(df, col)

    def test_pdist_ext(self):
        df = self.fresh()
        df.ta.pdist(append=True)
        self._check_appended(df, "PDIST")
        valid = df["PDIST"].dropna()
        self.assertTrue((valid >= 0).all(), "PDIST has negative values")

    def test_squeeze_ext(self):
        df = self.fresh()
        df.ta.squeeze(append=True)
        sqz_cols = [c for c in df.columns if c.startswith("SQZ")]
        self.assertTrue(len(sqz_cols) >= 1, "No SQZ columns appended")
        for col in sqz_cols:
            self._check_appended(df, col)

    def test_thermo_ext(self):
        df = self.fresh()
        df.ta.thermo(append=True)
        self._check_last_col(df)

    def test_ui_ext(self):
        df = self.fresh()
        df.ta.ui(append=True)
        self._check_appended(df, "UI_14")
        valid = df["UI_14"].dropna()
        self.assertTrue((valid >= 0).all(), "UI has negative values")
