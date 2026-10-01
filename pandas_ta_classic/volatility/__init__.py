from typing import TYPE_CHECKING

from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage

install_lazy_subpackage(__name__)

if TYPE_CHECKING:
    from .aberration import aberration as aberration
    from .accbands import accbands as accbands
    from .atr import atr as atr
    from .avolume import avolume as avolume
    from .bbands import bbands as bbands
    from .ce import ce as ce
    from .cvi import cvi as cvi
    from .donchian import donchian as donchian
    from .hvol import hvol as hvol
    from .hwc import hwc as hwc
    from .kc import kc as kc
    from .massi import massi as massi
    from .natr import natr as natr
    from .pdist import pdist as pdist
    from .rvi import rvi as rvi
    from .thermo import thermo as thermo
    from .true_range import true_range as true_range
    from .ui import ui as ui
