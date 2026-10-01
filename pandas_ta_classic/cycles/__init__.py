from typing import TYPE_CHECKING

from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage

install_lazy_subpackage(__name__)

if TYPE_CHECKING:
    from .dsp import dsp as dsp
    from .ebsw import ebsw as ebsw
    from .ht_dcperiod import ht_dcperiod as ht_dcperiod
    from .ht_dcphase import ht_dcphase as ht_dcphase
    from .ht_phasor import ht_phasor as ht_phasor
    from .ht_sine import ht_sine as ht_sine
    from .ht_trendmode import ht_trendmode as ht_trendmode
    from .msw import msw as msw
