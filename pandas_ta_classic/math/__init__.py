"""Math Operators and Transforms for pandas-ta-classic.

Covers TA-Lib's Math Operator (ADD, SUB, DIV, MULT, MAX, MIN, SUM,
MAXINDEX, MININDEX, MINMAX, MINMAXINDEX) and Math Transform (ACOS, ASIN,
ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH)
groups, plus tulipy extras (ABS, ROUND, TRUNC, TODEG, TORAD).
"""

from pandas_ta_classic._meta import _MATH_ALIASES
from pandas_ta_classic._lazy_subpackage import install_lazy_subpackage

install_lazy_subpackage(
    __name__,
    aliases=_MATH_ALIASES,
)
