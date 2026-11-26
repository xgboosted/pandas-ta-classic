"""A small whole-number parameter is rejected or gives clean values on long input.

`kama(drift=2)` published only its seed bar on 1000 bars: its efficiency
ratio needs `length` price changes `drift` bars apart, the first of which
does not exist until bar `drift`, and the recursion started at bar
`length` from a NaN ratio, so the NaN rode it to the end. `linreg(length=1)`
fit a line through one point: 0/0 on every bar, a RuntimeWarning and an
empty column. Every int parameter of every registered indicator is set to
0, 1, 2, 3 and 5, and so is every int option read from **kwargs that the
default call validates (min_periods=0 made median, mad, stdev and variance
drop the last bar). A value the indicator accepts may not raise anything but
ValueError or warn a RuntimeWarning from any file (numpy's own for the mean
of an empty window inside np.median); compared with the same call at its
default, it may not leave a column with at most one value, open a gap in a
column that has none, or add an infinity. A value its formula cannot use is
rejected by its validation, with an error that names the parameter or the
path's option (the wording is checked in test_parameter_validation.py).

Each value is tried on every option path, not only the default call:
kc(mamode="linreg", length=1) and squeeze(lazybear=True, kc_length=1)
divided 0 by 0 while the default call of both was clean. A path is one
flag flipped or one choice of a string option. Options in the signature
are known by their annotation; an option read from **kwargs is known by
what its validation says about "bogus".
"""

import ast
import inspect
import re
import typing
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from pandas_ta_classic.utils._keywords import _KWARGS_OPTIONS
from tests.contracts import INDICATORS, NOT_OHLCV, SKIP_BOOL_KWARGS, boolean_kwargs, inputs, params, spy

VALUES = (0, 1, 2, 3, 5)
_CHOICES = re.compile(r" must be one of (\[.*\]), got 'bogus'$")


def _annotated(name: str, kind: type) -> list[str]:
    return [p for p, spec in inspect.signature(getattr(ta, name)).parameters.items() if spec.annotation is kind or kind in typing.get_args(spec.annotation)]


def _rejection(name: str, option: str, frame: pd.DataFrame) -> str:
    """What the validation says about option="bogus": it names the option's kind ("" when it takes any value)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            getattr(ta, name)(**inputs(name, frame), **{option: "bogus"})
        except (TypeError, ValueError) as error:  # TypeError: a removed option, which says so
            return str(error)
    return ""


def _options(name: str) -> tuple[list[str], dict[str, dict]]:
    """The int parameters of *name* and its option paths ("" is the default call), keyed by label.

    The signature gives the annotated ones. An option read from **kwargs (utils/_keywords.py) is
    sorted by what its validation says about "bogus": "must be True or False" is a flag,
    "must be one of [...]" lists the choices, "must be an integer" makes it one more int parameter.
    """
    frame = spy(300)
    signature = inspect.signature(getattr(ta, name)).parameters
    ints = [p for p in _annotated(name, int) if p != "offset"]
    flags = boolean_kwargs(name)
    strings = {param: _rejection(name, param, frame) for param in _annotated(name, str)}
    for option in sorted(_KWARGS_OPTIONS.get(name, set()) - set(signature) - SKIP_BOOL_KWARGS):
        message = _rejection(name, option, frame)
        if "must be True or False" in message:
            flags.append(option)
        elif "must be one of" in message:
            strings[option] = message
        elif "must be an integer" in message:
            ints.append(option)
    paths: dict[str, dict] = {"": {}}
    paths.update({f"{flag}={value}": {flag: value} for flag in flags for value in (True, False)})
    for option, message in strings.items():
        found = _CHOICES.search(message)  # none for a free-form string (vwap's anchor)
        paths.update({f"{option}={choice}": {option: choice} for choice in (ast.literal_eval(found.group(1)) if found else [])})
    return ints, paths


# Paths with no result to check, with the reason.
NO_RESULT = {
    ("tos_stdevall", "lookahead=False"): "no causal mode: returns None with a UserWarning, by design",
    ("vp", "lookahead=False"): "no causal mode: returns None with a UserWarning, by design",
}
OPTIONS = {name: _options(name) for name in INDICATORS if name not in NOT_OHLCV}
PATHS = {name: paths for name, (ints, paths) in OPTIONS.items() if ints}
CASES = [(name, param, path) for name, (ints, paths) in OPTIONS.items() for param in ints for path in paths if (name, path) not in NO_RESULT]
KNOWN = {
    ("brar", "length"): "KEPT: x/0 with a positive numerator is +inf on purpose (brar.py): in a short window BR is +inf where every low is at or"
    " above the previous close (length 1, 2, 3), AR where open == low (length 1)",
    ("vp", "width"): "KEPT: width=1 is one price bin, a one-row result, not an empty column",
}


def _known(row: tuple) -> tuple:
    """A (name, param, path) entry covers one path; a (name, param) entry covers every path."""
    return row if row in KNOWN else row[:2]


@pytest.fixture(scope="module")
def frame():
    return spy(1000)


def _frame(result) -> pd.DataFrame:
    return result.to_frame() if isinstance(result, pd.Series) else result


def _holes(column: pd.Series) -> int:
    """Missing values after the first one: 0 for a column that, once it starts, has a value on every bar."""
    valid = column.notna().to_numpy()
    return int((~valid[valid.argmax() :]).sum()) if valid.any() else 0


def _infinite(column: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(column) and bool(np.isinf(column.to_numpy(float, na_value=np.nan)).any())


def test_sweep_found_the_parameters_and_paths():
    assert ("kama", "drift", "") in CASES and ("linreg", "length", "") in CASES
    assert ("squeeze", "kc_length", "lazybear=True") in CASES and ("kc", "length", "mamode=linreg") in CASES
    assert ("median", "min_periods", "") in CASES and ("qstick", "length", "ma=linreg") in CASES  # read from **kwargs
    assert ("dema", "length", "adjust=True") in CASES and ("kc", "length", "tr=False") in CASES
    assert ("vwap", "anchor") not in {row[:2] for row in CASES}  # str, not int; and its anchor is free-form
    assert len({row[:2] for row in CASES}) > 250 and len(CASES) > 1500


@pytest.mark.parametrize(("name", "param", "path"), params(CASES, KNOWN, key=_known, id=lambda row: "-".join(filter(None, row))))
def test_a_small_int_parameter_is_rejected_or_clean(name, param, path, frame):
    func = getattr(ta, name)
    option = PATHS[name][path]
    default = _frame(func(**inputs(name, frame), **option))
    found = []
    for value in VALUES:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            try:
                result = _frame(func(**inputs(name, frame), **option, **{param: value}))
            except ValueError as error:  # outside the parameter's range (ddof >= length, poles not 2 or 3, ...)
                # A rejection names the parameter or the path's option; any other ValueError is a finding.
                if param not in str(error) and not any(key in str(error) for key in option):
                    found.append(f"{param}={value}: ValueError naming neither {param} nor the path: {error}")
                continue
            except Exception as error:  # noqa: BLE001 -- any other exception is a finding: quantile(min_periods=0) raised IndexError
                found.append(f"{param}={value}: {type(error).__name__}: {error}")
                continue
        where = f"{param}={value}"
        # Any file: a 0/0 inside numpy (the mean of an empty window in np.median) is attributed to numpy's own file.
        found += sorted({f"{where}: RuntimeWarning at {Path(w.filename).name}:{w.lineno}" for w in caught if issubclass(w.category, RuntimeWarning)})
        # The parameter is often in the column name (LR_1 for LR_14): pair the columns by position.
        pairs = list(zip(default.columns, result.columns)) if default.shape[1] == result.shape[1] else [(c, c) for c in result.columns if c in default]
        if not pairs:
            found.append(f"{where}: columns {list(result.columns)} pair with none of {list(default.columns)}")
        for d, c in pairs:
            if default[d].notna().any() and result[c].notna().sum() <= 1:
                found.append(f"{where}: {c} blank")
            elif default[d].notna().any() and not _holes(default[d]) and _holes(result[c]):  # min_periods=0 lost the last bar
                found.append(f"{where}: {c} has {_holes(result[c])} missing bars after its first value")
            if _infinite(result[c]) and not _infinite(default[d]):
                found.append(f"{where}: {c} infinite")
    assert not found, f"{name}: {found}"
