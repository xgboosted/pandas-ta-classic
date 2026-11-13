"""A parameter that changes the values goes into the column name when it is not at its default.

Two calls that differed only in such a parameter produced the same column name,
so appending both overwrote one with the other: ``zscore(ddof=0)`` and
``zscore(ddof=1)`` were both ``ZS_30``, and ``dpo()``, ``dpo(centered=False)``
and ``dpo(lookahead=False)`` were all ``DPO_20`` (#343). A non-default value now
adds ``_<name><value>`` (``ZS_30_ddof0``, ``DPO_20_centered0``); default calls
keep their names.

Every parameter of every registered indicator whose default is a literal in the
code is changed once on its own (int + 1, float x 0.75 or + 0.01 from 0, bool
flipped). No column may keep its name while its values change, unless the
parameter is in ``EXEMPT`` with its reason.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.contracts import INDICATORS, code_defaults, inputs, spy

# Parameters that change the values and stay out of the name, with the reason.
EXEMPT_PARAMETERS = {
    "scalar": "KEPT: rescales the same series; its unit, not a different indicator",
    "drift": "KEPT: the difference period of the same series, rarely varied within one strategy",
    "asint": "KEPT: the output type of the same flags (1.0/0.0 or bool)",
    "asbool": "KEPT: the output type of the same flags (1.0/0.0 or bool)",
    "append_span": "KEPT: appends the future span rows; it adds rows, not other values in a column",
}
EXEMPT_CASES = {
    ("vp", "width"): "KEPT: the number of price bins, i.e. of rows; VP_{width} names the frame",
}


def _cases():
    cases = []
    for name in INDICATORS:
        func = getattr(ta, name)
        signature = inspect.signature(func).parameters
        source = inspect.getsource(inspect.getmodule(inspect.unwrap(func)))
        for param, default in code_defaults(source).items():
            if param in signature and param not in ("offset", "talib") and default is not None and not isinstance(default, str):
                cases.append((name, param, default))
    return cases


CASES = _cases()


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return spy(600)


def _frame(result) -> pd.DataFrame:
    return result.to_frame() if isinstance(result, pd.Series) else result


def _other(default):
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 1
    return default * 0.75 if default else 0.01


def test_sweep_found_the_parameters():
    found = {(name, param) for name, param, _ in CASES}
    assert {("zscore", "ddof"), ("dpo", "centered"), ("sarext", "startvalue"), ("uo", "fast_w")} <= found and len(found) > 300


@pytest.mark.parametrize(("name", "param", "default"), CASES, ids=[f"{n}-{p}" for n, p, _ in CASES])
def test_a_value_changing_parameter_is_in_the_name(name, param, default, frame):
    if param in EXEMPT_PARAMETERS or (name, param) in EXEMPT_CASES:
        pytest.skip(EXEMPT_PARAMETERS.get(param) or EXEMPT_CASES[(name, param)])
    func = getattr(ta, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = _frame(func(**inputs(name, frame)))
        try:
            changed = _frame(func(**inputs(name, frame), **{param: _other(default)}))
        except ValueError as error:  # outside the parameter's range or its combination
            if name not in str(error):
                raise
            pytest.skip(f"{name}({param}={_other(default)!r}) is out of range: {error}")
    same_name = [c for c in changed.columns if c in base.columns]
    silent = [c for c in same_name if not np.array_equal(changed[c].to_numpy(float, na_value=np.nan), base[c].to_numpy(float, na_value=np.nan), equal_nan=True)]
    assert not silent, f"{name}({param}={_other(default)!r}) changes {silent} under the default call's name"


@pytest.mark.parametrize(("kwargs", "expected"), [({}, "DPO_20"), ({"centered": False}, "DPO_20_centered0"), ({"lookahead": False}, "DPO_20_centered0")])
def test_dpo_names_its_uncentered_line(kwargs, expected, frame):
    """lookahead=False is read from **kwargs, out of the sweep's reach; it gives the
    uncentered line, so it carries the same token."""
    assert ta.dpo(frame.close, **kwargs).name == expected


def test_the_token_format():
    assert ta.zscore(spy(100).close, ddof=0, std=1.5).name == "ZS_30_std1.5_ddof0"
    assert ta.uo(*(spy(100)[c] for c in ("high", "low", "close")), fast_w=2.0).name == "UO_7_14_28_fastw2"
