import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.assertions import AccessorValueCase


class TestExtStatisticsDeeper(AccessorValueCase):
    def test_entropy_ext(self):
        df = self.fresh()
        df.ta.entropy(append=True)
        self._check_appended(df, "ENTP_10")
        valid = df["ENTP_10"].dropna()
        self.assertTrue((valid >= 0).all(), "Entropy has negative values")

    def test_kurtosis_ext(self):
        df = self.fresh()
        df.ta.kurtosis(append=True)
        self._check_last_col(df)

    def test_mad_ext(self):
        df = self.fresh()
        df.ta.mad(append=True)
        self._check_appended(df, "MAD_30")
        valid = df["MAD_30"].dropna()
        self.assertTrue((valid >= 0).all(), "MAD has negative values")

    def test_skew_ext(self):
        df = self.fresh()
        df.ta.skew(append=True)
        self._check_last_col(df)

    def test_variance_ext(self):
        df = self.fresh()
        df.ta.variance(append=True)
        self._check_last_col(df)
        # Variance must be non-negative
        col = df.columns[-1]
        valid = df[col].dropna()
        self.assertTrue((valid >= 0).all(), "Variance has negative values")

    def test_zscore_ext(self):
        df = self.fresh()
        df.ta.zscore(append=True)
        self._check_last_col(df)
