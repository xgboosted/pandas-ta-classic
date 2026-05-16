from tests.config import get_sample_data
from tests.context import pandas_ta_classic as pandas_ta

from unittest import TestCase, skip
from pandas import DataFrame


class TestCandleExtension(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cdl_doji_ext(self):
        self.data.ta.cdl_pattern("doji", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_DOJI_10_0.1")

    def test_cdl_inside_ext(self):
        self.data.ta.cdl_pattern("inside", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_INSIDE")

    def test_cdl_z_ext(self):
        self.data.ta.cdl_z(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-4:]),
            ["open_Z_30_1", "high_Z_30_1", "low_Z_30_1", "close_Z_30_1"],
        )

    def test_ha_ext(self):
        self.data.ta.ha(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-4:]), ["HA_open", "HA_high", "HA_low", "HA_close"]
        )

    def test_cdl_2crows_ext(self):
        self.data.ta.cdl_pattern("2crows", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_2CROWS")

    def test_cdl_3blackcrows_ext(self):
        self.data.ta.cdl_pattern("3blackcrows", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_3BLACKCROWS")

    def test_cdl_3inside_ext(self):
        self.data.ta.cdl_pattern("3inside", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_3INSIDE")

    def test_cdl_3linestrike_ext(self):
        self.data.ta.cdl_pattern("3linestrike", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_3LINESTRIKE")

    def test_cdl_3outside_ext(self):
        self.data.ta.cdl_pattern("3outside", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_3OUTSIDE")

    def test_cdl_3starsinsouth_ext(self):
        self.data.ta.cdl_pattern("3starsinsouth", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_3STARSINSOUTH")

    def test_cdl_3whitesoldiers_ext(self):
        self.data.ta.cdl_pattern("3whitesoldiers", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_3WHITESOLDIERS")

    def test_cdl_abandonedbaby_ext(self):
        self.data.ta.cdl_pattern("abandonedbaby", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_ABANDONEDBABY")

    def test_cdl_advanceblock_ext(self):
        self.data.ta.cdl_pattern("advanceblock", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_ADVANCEBLOCK")

    def test_cdl_belthold_ext(self):
        self.data.ta.cdl_pattern("belthold", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_BELTHOLD")

    def test_cdl_breakaway_ext(self):
        self.data.ta.cdl_pattern("breakaway", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_BREAKAWAY")

    def test_cdl_closingmarubozu_ext(self):
        self.data.ta.cdl_pattern("closingmarubozu", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_CLOSINGMARUBOZU")

    def test_cdl_concealbabyswall_ext(self):
        self.data.ta.cdl_pattern("concealbabyswall", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_CONCEALBABYSWALL")

    def test_cdl_counterattack_ext(self):
        self.data.ta.cdl_pattern("counterattack", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_COUNTERATTACK")

    def test_cdl_darkcloudcover_ext(self):
        self.data.ta.cdl_pattern("darkcloudcover", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_DARKCLOUDCOVER")

    def test_cdl_dojistar_ext(self):
        self.data.ta.cdl_pattern("dojistar", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_DOJISTAR")

    def test_cdl_dragonflydoji_ext(self):
        self.data.ta.cdl_pattern("dragonflydoji", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_DRAGONFLYDOJI")

    def test_cdl_engulfing_ext(self):
        self.data.ta.cdl_pattern("engulfing", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_ENGULFING")

    def test_cdl_eveningdojistar_ext(self):
        self.data.ta.cdl_pattern("eveningdojistar", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_EVENINGDOJISTAR")

    def test_cdl_eveningstar_ext(self):
        self.data.ta.cdl_pattern("eveningstar", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_EVENINGSTAR")

    def test_cdl_gapsidesidewhite_ext(self):
        self.data.ta.cdl_pattern("gapsidesidewhite", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_GAPSIDESIDEWHITE")

    def test_cdl_gravestonedoji_ext(self):
        self.data.ta.cdl_pattern("gravestonedoji", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_GRAVESTONEDOJI")

    def test_cdl_hammer_ext(self):
        self.data.ta.cdl_pattern("hammer", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_HAMMER")

    def test_cdl_hangingman_ext(self):
        self.data.ta.cdl_pattern("hangingman", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_HANGINGMAN")

    def test_cdl_harami_ext(self):
        self.data.ta.cdl_pattern("harami", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_HARAMI")

    def test_cdl_haramicross_ext(self):
        self.data.ta.cdl_pattern("haramicross", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_HARAMICROSS")

    def test_cdl_highwave_ext(self):
        self.data.ta.cdl_pattern("highwave", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_HIGHWAVE")

    def test_cdl_hikkake_ext(self):
        self.data.ta.cdl_pattern("hikkake", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_HIKKAKE")

    def test_cdl_hikkakemod_ext(self):
        self.data.ta.cdl_pattern("hikkakemod", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_HIKKAKEMOD")

    def test_cdl_homingpigeon_ext(self):
        self.data.ta.cdl_pattern("homingpigeon", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_HOMINGPIGEON")

    def test_cdl_identical3crows_ext(self):
        self.data.ta.cdl_pattern("identical3crows", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_IDENTICAL3CROWS")

    def test_cdl_inneck_ext(self):
        self.data.ta.cdl_pattern("inneck", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_INNECK")

    def test_cdl_invertedhammer_ext(self):
        self.data.ta.cdl_pattern("invertedhammer", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_INVERTEDHAMMER")

    def test_cdl_kicking_ext(self):
        self.data.ta.cdl_pattern("kicking", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_KICKING")

    def test_cdl_kickingbylength_ext(self):
        self.data.ta.cdl_pattern("kickingbylength", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_KICKINGBYLENGTH")

    def test_cdl_ladderbottom_ext(self):
        self.data.ta.cdl_pattern("ladderbottom", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_LADDERBOTTOM")

    def test_cdl_longleggeddoji_ext(self):
        self.data.ta.cdl_pattern("longleggeddoji", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_LONGLEGGEDDOJI")

    def test_cdl_longline_ext(self):
        self.data.ta.cdl_pattern("longline", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_LONGLINE")

    def test_cdl_marubozu_ext(self):
        self.data.ta.cdl_pattern("marubozu", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_MARUBOZU")

    def test_cdl_matchinglow_ext(self):
        self.data.ta.cdl_pattern("matchinglow", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_MATCHINGLOW")

    def test_cdl_mathold_ext(self):
        self.data.ta.cdl_pattern("mathold", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_MATHOLD")

    def test_cdl_morningdojistar_ext(self):
        self.data.ta.cdl_pattern("morningdojistar", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_MORNINGDOJISTAR")

    def test_cdl_morningstar_ext(self):
        self.data.ta.cdl_pattern("morningstar", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_MORNINGSTAR")

    def test_cdl_onneck_ext(self):
        self.data.ta.cdl_pattern("onneck", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_ONNECK")

    def test_cdl_piercing_ext(self):
        self.data.ta.cdl_pattern("piercing", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_PIERCING")

    def test_cdl_rickshawman_ext(self):
        self.data.ta.cdl_pattern("rickshawman", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_RICKSHAWMAN")

    def test_cdl_risefall3methods_ext(self):
        self.data.ta.cdl_pattern("risefall3methods", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_RISEFALL3METHODS")

    def test_cdl_separatinglines_ext(self):
        self.data.ta.cdl_pattern("separatinglines", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_SEPARATINGLINES")

    def test_cdl_shootingstar_ext(self):
        self.data.ta.cdl_pattern("shootingstar", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_SHOOTINGSTAR")

    def test_cdl_shortline_ext(self):
        self.data.ta.cdl_pattern("shortline", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_SHORTLINE")

    def test_cdl_spinningtop_ext(self):
        self.data.ta.cdl_pattern("spinningtop", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_SPINNINGTOP")

    def test_cdl_stalledpattern_ext(self):
        self.data.ta.cdl_pattern("stalledpattern", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_STALLEDPATTERN")

    def test_cdl_sticksandwich_ext(self):
        self.data.ta.cdl_pattern("sticksandwich", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_STICKSANDWICH")

    def test_cdl_takuri_ext(self):
        self.data.ta.cdl_pattern("takuri", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_TAKURI")

    def test_cdl_tasukigap_ext(self):
        self.data.ta.cdl_pattern("tasukigap", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_TASUKIGAP")

    def test_cdl_thrusting_ext(self):
        self.data.ta.cdl_pattern("thrusting", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_THRUSTING")

    def test_cdl_tristar_ext(self):
        self.data.ta.cdl_pattern("tristar", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_TRISTAR")

    def test_cdl_unique3river_ext(self):
        self.data.ta.cdl_pattern("unique3river", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_UNIQUE3RIVER")

    def test_cdl_upsidegap2crows_ext(self):
        self.data.ta.cdl_pattern("upsidegap2crows", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_UPSIDEGAP2CROWS")

    def test_cdl_xsidegap3methods_ext(self):
        self.data.ta.cdl_pattern("xsidegap3methods", append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CDL_XSIDEGAP3METHODS")
