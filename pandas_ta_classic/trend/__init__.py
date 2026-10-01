from typing import TYPE_CHECKING

from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage

install_lazy_subpackage(__name__)

if TYPE_CHECKING:
    from .adx import adx as adx
    from .adxr import adxr as adxr
    from .amat import amat as amat
    from .aroon import aroon as aroon
    from .chop import chop as chop
    from .cksp import cksp as cksp
    from .cpr import cpr as cpr
    from .decay import decay as decay
    from .decreasing import decreasing as decreasing
    from .dpo import dpo as dpo
    from .dx import dx as dx
    from .edecay import edecay as edecay
    from .increasing import increasing as increasing
    from .long_run import long_run as long_run
    from .minus_dm import minus_dm as minus_dm
    from .plus_dm import plus_dm as plus_dm
    from .pmax import pmax as pmax
    from .psar import psar as psar
    from .qstick import qstick as qstick
    from .sarext import sarext as sarext
    from .short_run import short_run as short_run
    from .tsignals import tsignals as tsignals
    from .ttm_trend import ttm_trend as ttm_trend
    from .vhf import vhf as vhf
    from .vortex import vortex as vortex
    from .xsignals import xsignals as xsignals
