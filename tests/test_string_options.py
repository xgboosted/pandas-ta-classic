"""An unknown string option raises ValueError naming the indicator called (AGENTS rule 2).

`vwap(anchor="bogus")` failed inside pandas with "Invalid frequency: BOGUS",
which names neither vwap() nor anchor. Every str-typed parameter of every
registered indicator gets "bogus".
"""

import inspect
import typing

import pytest

import pandas_ta_classic as ta
from tests.contracts import INDICATORS, NOT_OHLCV, inputs, params, spy

CASES = [
    (name, param)
    for name in INDICATORS
    if name not in NOT_OHLCV and name != "ma"  # ma's name is checked in test_parameter_validation
    for param, spec in inspect.signature(getattr(ta, name)).parameters.items()
    if spec.annotation is str or str in typing.get_args(spec.annotation)
]
_MAMODE = "OPEN: ma() validates the name, so the error names ma() and its parameter 'name', not the indicator and mamode"
KNOWN = {
    ("accbands", "mamode"): _MAMODE,
    ("adx", "mamode"): _MAMODE,
    ("adxr", "mamode"): _MAMODE,
    ("amat", "mamode"): _MAMODE,
    ("aobv", "mamode"): _MAMODE,
    ("apo", "mamode"): _MAMODE,
    ("atr", "mamode"): _MAMODE,
    ("bbands", "mamode"): _MAMODE,
    ("bias", "mamode"): _MAMODE,
    ("ce", "mamode"): _MAMODE,
    ("dm", "mamode"): _MAMODE,
    ("dx", "mamode"): _MAMODE,
    ("efi", "mamode"): _MAMODE,
    ("hilo", "mamode"): _MAMODE,
    ("inertia", "mamode"): _MAMODE,
    ("kc", "mamode"): _MAMODE,
    ("kvo", "mamode"): _MAMODE,
    ("natr", "mamode"): _MAMODE,
    ("pmax", "mamode"): _MAMODE,
    ("ppo", "mamode"): _MAMODE,
    ("qqe", "mamode"): _MAMODE,
    ("rvi", "mamode"): _MAMODE,
    ("stoch", "mamode"): _MAMODE,
    ("stochf", "mamode"): _MAMODE,
    ("stochrsi", "mamode"): _MAMODE,
    ("thermo", "mamode"): _MAMODE,
    ("tsi", "mamode"): _MAMODE,
    ("vfi", "mamode"): _MAMODE,
}


@pytest.fixture(scope="module")
def frame():
    return spy(300)


def test_sweep_found_the_options():
    assert ("vwap", "anchor") in CASES and len(CASES) > 30


@pytest.mark.parametrize(("name", "param"), params(CASES, KNOWN, id=lambda row: "-".join(row)))
def test_unknown_string_option_names_the_indicator(name, param, frame):
    with pytest.raises(ValueError, match=rf"^{name}\(\) {param}\b"):
        getattr(ta, name)(**inputs(name, frame), **{param: "bogus"})


def test_vwap_anchor_error_shows_the_value_passed(frame):
    with pytest.raises(ValueError, match=r"got 'bogus'$"):
        ta.vwap(**inputs("vwap", frame), anchor="bogus")
