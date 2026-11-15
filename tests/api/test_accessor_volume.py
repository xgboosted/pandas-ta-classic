import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.assertions import AccessorValueCase


class TestExtVolumeDeeper(AccessorValueCase):
    def test_ad_ext(self):
        df = self.fresh()
        df.ta.ad(append=True)
        ad_cols = [c for c in df.columns if c.startswith("AD")]
        self.assertTrue(len(ad_cols) >= 1, "No AD column appended")
        self._check_appended(df, ad_cols[0])

    def test_aobv_ext(self):
        df = self.fresh()
        df.ta.aobv(append=True)
        self._check_appended(df, "OBV")

    def test_cmf_ext(self):
        df = self.fresh()
        df.ta.cmf(append=True)
        self._check_last_col(df)

    def test_efi_ext(self):
        df = self.fresh()
        df.ta.efi(append=True)
        self._check_appended(df, "EFI_13")

    def test_mfi_ext(self):
        df = self.fresh()
        df.ta.mfi(append=True)
        mfi_cols = [c for c in df.columns if c.startswith("MFI")]
        self.assertTrue(len(mfi_cols) >= 1, "No MFI column appended")
        self._check_appended(df, mfi_cols[0])
        # MFI in [0, 100]
        valid = df[mfi_cols[0]].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 100).all(), "MFI out of [0,100]")

    def test_obv_ext(self):
        df = self.fresh()
        df.ta.obv(append=True)
        obv_cols = [c for c in df.columns if c.startswith("OBV")]
        self.assertTrue(len(obv_cols) >= 1, "No OBV column appended")
        self._check_appended(df, obv_cols[0])

    def test_pvol_ext(self):
        df = self.fresh()
        df.ta.pvol(append=True)
        self._check_appended(df, "PVOL")
        valid = df["PVOL"].dropna()
        self.assertTrue((valid >= 0).all(), "PVOL has negative values")

    def test_pvt_ext(self):
        df = self.fresh()
        df.ta.pvt(append=True)
        self._check_appended(df, "PVT")

    def test_vwmacd_ext(self):
        df = self.fresh()
        df.ta.vwmacd(append=True)
        vwm_cols = [c for c in df.columns if c.startswith("VWMACD")]
        self.assertTrue(len(vwm_cols) >= 1, "No VWMACD columns appended")
        for col in vwm_cols:
            self._check_appended(df, col)
