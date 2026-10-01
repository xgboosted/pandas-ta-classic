from typing import TYPE_CHECKING

from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage

install_lazy_subpackage(__name__)

if TYPE_CHECKING:
    from .ad import ad as ad
    from .adosc import adosc as adosc
    from .aobv import aobv as aobv
    from .cmf import cmf as cmf
    from .efi import efi as efi
    from .emv import emv as emv
    from .eom import eom as eom
    from .kvo import kvo as kvo
    from .marketfi import marketfi as marketfi
    from .mfi import mfi as mfi
    from .nvi import nvi as nvi
    from .obv import obv as obv
    from .pvi import pvi as pvi
    from .pvol import pvol as pvol
    from .pvr import pvr as pvr
    from .pvt import pvt as pvt
    from .vfi import vfi as vfi
    from .vosc import vosc as vosc
    from .vp import vp as vp
    from .wad import wad as wad
