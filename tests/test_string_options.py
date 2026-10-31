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
KNOWN: dict = {}


@pytest.fixture(scope="module")
def frame():
    return spy(300)


def test_sweep_found_the_options():
    assert ("vwap", "anchor") in CASES and len(CASES) > 30


@pytest.mark.parametrize(("name", "param"), params(CASES, KNOWN, id=lambda row: "-".join(row)))
def test_unknown_string_option_names_the_indicator(name, param, frame):
    with pytest.raises(ValueError, match=rf"^{name}\(\) {param}\b"):
        getattr(ta, name)(**inputs(name, frame), **{param: "bogus"})


@pytest.mark.parametrize("metric", ["downside_deviation", "volatility"])
def test_unknown_metric_tf_names_the_metric(metric, frame):
    """The tf check lived in total_time(), so the error named total_time() (#372's note)."""
    with pytest.raises(ValueError, match=rf"^{metric}\(\) tf must be one of .*got 'bogus'$"):
        getattr(ta.utils, metric)(frame.close.pct_change().dropna() if metric == "downside_deviation" else frame.close, tf="bogus")


def test_vwap_anchor_error_shows_the_value_passed(frame):
    with pytest.raises(ValueError, match=r"got 'bogus'$"):
        ta.vwap(**inputs("vwap", frame), anchor="bogus")
