"""Math Operators and Transforms for pandas-ta-classic.

Covers TA-Lib's Math Operator (ADD, SUB, DIV, MULT, MAX, MIN, SUM,
MAXINDEX, MININDEX, MINMAX, MINMAXINDEX) and Math Transform (ACOS, ASIN,
ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH)
groups, plus tulipy extras (ABS, ROUND, TRUNC, TODEG, TORAD).
"""

from typing import TYPE_CHECKING

from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage
from pandas_ta_classic._meta import _MATH_ALIASES

install_lazy_subpackage(
    __name__,
    aliases=_MATH_ALIASES,
)

if TYPE_CHECKING:
    from .acos import acos as acos
    from .add import add as add
    from .asin import asin as asin
    from .atan import atan as atan
    from .ceil import ceil as ceil
    from .cos import cos as cos
    from .cosh import cosh as cosh
    from .div import div as div
    from .exp import exp as exp
    from .floor import floor as floor
    from .ln import ln as ln
    from .log10 import log10 as log10
    from .maxindex import maxindex as maxindex
    from .minindex import minindex as minindex
    from .minmax import minmax as minmax
    from .minmaxindex import minmaxindex as minmaxindex
    from .mult import mult as mult
    from .npabs import npabs as npabs
    from .npround import npround as npround
    from .rolling_max import rolling_max as rolling_max
    from .rolling_min import rolling_min as rolling_min
    from .rolling_sum import rolling_sum as rolling_sum
    from .sin import sin as sin
    from .sinh import sinh as sinh
    from .sqrt import sqrt as sqrt
    from .sub import sub as sub
    from .tan import tan as tan
    from .tanh import tanh as tanh
    from .todeg import todeg as todeg
    from .torad import torad as torad
    from .trunc import trunc as trunc
