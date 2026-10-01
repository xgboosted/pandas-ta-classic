from typing import TYPE_CHECKING

from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage

install_lazy_subpackage(__name__)

if TYPE_CHECKING:
    from .alma import alma as alma
    from .avgprice import avgprice as avgprice
    from .dema import dema as dema
    from .ema import ema as ema
    from .fwma import fwma as fwma
    from .hilo import hilo as hilo
    from .hl2 import hl2 as hl2
    from .hlc3 import hlc3 as hlc3
    from .hma import hma as hma
    from .ht_trendline import ht_trendline as ht_trendline
    from .hwma import hwma as hwma
    from .ichimoku import ichimoku as ichimoku
    from .jma import jma as jma
    from .kama import kama as kama
    from .linreg import linreg as linreg
    from .linregangle import linregangle as linregangle
    from .linregintercept import linregintercept as linregintercept
    from .linregslope import linregslope as linregslope
    from .ma import ma as ma
    from .mama import mama as mama
    from .mavp import mavp as mavp
    from .mcgd import mcgd as mcgd
    from .medprice import medprice as medprice
    from .midpoint import midpoint as midpoint
    from .midprice import midprice as midprice
    from .mmar import mmar as mmar
    from .ohlc4 import ohlc4 as ohlc4
    from .pwma import pwma as pwma
    from .rainbow import rainbow as rainbow
    from .rma import rma as rma
    from .sinwma import sinwma as sinwma
    from .sma import sma as sma
    from .ssf import ssf as ssf
    from .supertrend import supertrend as supertrend
    from .swma import swma as swma
    from .t3 import t3 as t3
    from .tema import tema as tema
    from .trima import trima as trima
    from .tsf import tsf as tsf
    from .typprice import typprice as typprice
    from .vidya import vidya as vidya
    from .vwap import vwap as vwap
    from .vwma import vwma as vwma
    from .wcp import wcp as wcp
    from .wma import wma as wma
    from .zlma import zlma as zlma
