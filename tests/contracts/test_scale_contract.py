"""Price scale is not information: every indicator is equivariant under it.

The same market quoted in a different unit -- a crypto pair at 3e-4, an index at
3e5, a stock after a 1:1000 split -- must give the same answer in that unit.
Multiplying every price by ``k`` (volume unchanged) leaves each output column
either unchanged (oscillators, ratios, signals), multiplied by ``k`` (price
levels, bands, ranges) or by ``k**2`` (variances).

What breaks this is an absolute constant hidden in a formula: a floor such as
``max(x, 1e-3)``, an ``atol`` that decides whether a window is degenerate, an
epsilon. On ``SPY_D``'s prices of a few hundred such a constant never binds, so
nothing else in the suite sees it; at ``k = 1e-6`` it decides the result.

Indicators are discovered through ``ta.Category``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests import contracts
from tests.contracts import BOUNDARY_COLUMNS, ELEMENTWISE, OHLCV_INDICATORS, call as _call, params

# Elementwise transforms of their input (sin(k x) is not k**p sin(x)); scale is
# meaningless for them. abs, todeg and torad are linear, so they keep the contract.
NOT_A_PRICE_FUNCTION = ELEMENTWISE - {"npabs", "todeg", "torad"}
# Unit-dependent by definition, not by accident.
SCALE_IS_THE_ANSWER = {
    "linregangle": "the angle of a slope in price per bar; TA-Lib's LINEARREG_ANGLE depends on the unit too",
    "decay": "decays a signal by a fixed step per bar; its input is a 0/1 signal, not a price",
}
INDICATORS = sorted(set(OHLCV_INDICATORS) - NOT_A_PRICE_FUNCTION - set(SCALE_IS_THE_ANSWER))
SCALES = {"tiny": 1e-6, "huge": 1e6}
EXPONENTS = (0, 1, 2)
RTOL = 1e-6

# (indicator, scale) pairs that break the contract today, as strict xfails: one
# that starts passing fails, so the table cannot go stale.
OPEN_FINDINGS: dict[tuple[str, str], str] = {}


@pytest.fixture(scope="module")
def spy() -> pd.DataFrame:
    return contracts.spy(1000)


def _scales_by(base: np.ndarray, scaled: np.ndarray, k: float) -> int | None:
    """The exponent p with scaled == k**p * base, or None."""
    if not np.array_equal(np.isnan(base), np.isnan(scaled)):
        return None
    finite = np.isfinite(base) & np.isfinite(scaled)
    for p in EXPONENTS:
        expected = base[finite] * k**p
        atol = RTOL * float(np.abs(expected).max()) if expected.size else 0.0
        if np.allclose(scaled[finite], expected, rtol=RTOL, atol=atol):
            return p
    return None


@pytest.mark.parametrize(("name", "scale"), params([(name, scale) for scale in SCALES for name in INDICATORS], OPEN_FINDINGS))
def test_output_scales_with_the_price(name: str, scale: str, spy: pd.DataFrame) -> None:
    k = SCALES[scale]
    scaled = spy.copy()
    scaled[["open", "high", "low", "close"]] *= k
    base, moved = _call(name, spy), _call(name, scaled)
    assert list(moved.columns) == list(base.columns), f"{name}: columns {list(moved.columns)} at x{k:g}, {list(base.columns)} at x1"

    broken = []
    for column in base.columns:
        if column in BOUNDARY_COLUMNS:
            continue
        b = base[column].to_numpy(float, na_value=np.nan)
        m = moved[column].to_numpy(float, na_value=np.nan)
        if _scales_by(b, m, k) is None:
            both = np.isfinite(b) & np.isfinite(m)
            ratio = m[both] / np.where(b[both] == 0, np.nan, b[both])
            broken.append(
                f"{column}: NaN mask {'differs' if not np.array_equal(np.isnan(b), np.isnan(m)) else 'same'}, "
                f"ratio to the x1 value spans {np.nanmin(ratio):.3g}..{np.nanmax(ratio):.3g}"
            )
    assert not broken, f"{name} at x{k:g}: " + "; ".join(broken)
