from typing import TYPE_CHECKING

from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage

install_lazy_subpackage(__name__)

if TYPE_CHECKING:
    from .drawdown import drawdown as drawdown
    from .log_return import log_return as log_return
    from .percent_return import percent_return as percent_return
