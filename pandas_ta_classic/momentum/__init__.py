from typing import TYPE_CHECKING

from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage

install_lazy_subpackage(__name__)

if TYPE_CHECKING:
    from .ao import ao as ao
    from .apo import apo as apo
    from .bias import bias as bias
    from .bop import bop as bop
    from .brar import brar as brar
    from .cci import cci as cci
    from .cfo import cfo as cfo
    from .cg import cg as cg
    from .cmo import cmo as cmo
    from .coppock import coppock as coppock
    from .cti import cti as cti
    from .dm import dm as dm
    from .er import er as er
    from .eri import eri as eri
    from .fisher import fisher as fisher
    from .fosc import fosc as fosc
    from .inertia import inertia as inertia
    from .kdj import kdj as kdj
    from .kst import kst as kst
    from .lrsi import lrsi as lrsi
    from .macd import macd as macd
    from .macdext import macdext as macdext
    from .macdfix import macdfix as macdfix
    from .mom import mom as mom
    from .pgo import pgo as pgo
    from .po import po as po
    from .ppo import ppo as ppo
    from .psl import psl as psl
    from .pvo import pvo as pvo
    from .qqe import qqe as qqe
    from .roc import roc as roc
    from .rocp import rocp as rocp
    from .rocr import rocr as rocr
    from .rocr100 import rocr100 as rocr100
    from .rsi import rsi as rsi
    from .rsx import rsx as rsx
    from .rvgi import rvgi as rvgi
    from .slope import slope as slope
    from .smc_sweep import smc_sweep as smc_sweep
    from .smi import smi as smi
    from .squeeze import squeeze as squeeze
    from .squeeze_pro import squeeze_pro as squeeze_pro
    from .stc import stc as stc
    from .stoch import stoch as stoch
    from .stochf import stochf as stochf
    from .stochrsi import stochrsi as stochrsi
    from .td_seq import td_seq as td_seq
    from .trix import trix as trix
    from .trixh import trixh as trixh
    from .tsi import tsi as tsi
    from .uo import uo as uo
    from .vwmacd import vwmacd as vwmacd
    from .willr import willr as willr
