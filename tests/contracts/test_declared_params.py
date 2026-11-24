"""Declared parameters (``length: Annotated[int, Bound(gt=1)] = 10``, issue #397).

Three contracts:

* every sweep that finds parameters by scraping the body for validation calls
  also finds the declared ones (moving ``_pos_int(length, 10, ...)`` into the
  signature silently dropped 221 parametrized tests until the scrapers read the
  markers too);
* a window handed to ``ma(mamode, ...)`` is checked against the chosen MA's own
  declared bound, so the error names the caller's parameter
  (``stoch(d=1, mamode="hma")`` raised ``stoch() length must be ...``);
* ``ta.defaults()`` reports what the markers declare.
"""

import inspect
import typing
import warnings

import pytest

import pandas_ta_classic as ta
from pandas_ta_classic.overlap.ma import _MA_DISPATCH
from pandas_ta_classic.utils._core import Bound, OneOf, declared_params
from tests.contracts import INDICATORS, code_defaults, declared, inputs, plain_annotation, spy, string_choices

DECLARING = [name for name in INDICATORS if declared(name)]


@pytest.mark.parametrize("name", DECLARING)
def test_the_scrapers_see_every_declared_default(name):
    source = inspect.getsource(inspect.getmodule(inspect.unwrap(getattr(ta, name))))
    scraped, choices = code_defaults(source), string_choices(name)
    for param, spec in declared(name).items():
        if spec.default is not None:
            assert scraped.get(param) == spec.default, f"{name}({param}): code_defaults() reads {scraped.get(param)!r}, declared {spec.default!r}"
        if isinstance(spec.marker, OneOf):
            assert choices.get(param) == sorted(spec.marker.choices), f"{name}({param}): string_choices() misses the declared choices"


def test_some_indicators_declare():
    assert len(DECLARING) >= 7


# The MAs whose declared length bound is stricter than > 0, with the smallest value it rejects.
_STRICT_MAS = {
    mamode: int(spec.marker.gt)
    for mamode, fn in _MA_DISPATCH.items()
    if (spec := declared_params(fn).get("length")) is not None and isinstance(spec.marker, Bound) and (spec.marker.gt or 0) >= 1
}
_EXCLUDED = {"offset", "drift"}


def _mamode_cases():
    rows = []
    for name in INDICATORS:
        sig = inspect.signature(getattr(ta, name))
        if "mamode" not in sig.parameters or string_choices(name).get("mamode", ()) not in (None, sorted(_MA_DISPATCH)):
            continue  # mavp takes TA-Lib's integer enum
        for param, spec in sig.parameters.items():
            annotation = plain_annotation(spec.annotation)
            if param not in _EXCLUDED and (annotation is int or int in typing.get_args(annotation)):
                rows += [(name, param, mamode) for mamode in sorted(_STRICT_MAS)]
    return rows


def _error(name: str, kwargs: dict) -> str | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            getattr(ta, name)(**inputs(name, spy()), **kwargs)
    except ValueError as error:
        return str(error)
    return None


@pytest.mark.parametrize(("name", "param", "mamode"), _mamode_cases())
def test_a_window_too_small_for_the_chosen_ma_names_the_parameter_passed(name, param, mamode):
    value = _STRICT_MAS[mamode]
    error = _error(name, {param: value, "mamode": mamode})
    default_error = _error(name, {param: value})
    if default_error is not None:  # rejected whatever the MA (apo(slow=1): fast must be <= slow)
        assert error == default_error
    elif error is not None:
        assert error.startswith(f"{name}() {param} "), error


def test_the_mamode_sweep_found_the_strict_mas():
    assert set(_STRICT_MAS) >= {"hma", "linreg"}
    assert len(_mamode_cases()) > 40


def test_defaults_reports_the_declaration():
    assert ta.defaults("hma") == {"length": {"kind": "int", "default": 10, "gt": 1, "ge": None, "lt": None, "in_kwargs": False}}
    adx = ta.defaults("adx")
    assert adx["lensig"]["default_from"] == "length" and adx["lensig"]["default"] is None
    assert adx["mamode"]["default"] == "rma" and "hma" in adx["mamode"]["choices"]
    assert ta.defaults("squeeze")["lazybear"] == {"kind": "bool", "default": False, "in_kwargs": True}
    assert ta.defaults(ta.sma) == ta.defaults("sma")


@pytest.mark.parametrize(
    ("indicator", "message"),
    [
        (None, r"defaults\(\) requires 'indicator', got None"),
        ("nope", r"defaults\(\) indicator must be the name"),
        ("ema", r"declares no parameters"),
    ],
)
def test_defaults_rejects_what_it_cannot_describe(indicator, message):
    with pytest.raises(ValueError, match=message):
        ta.defaults(indicator)


def test_a_declared_option_is_validated_and_resolved():
    close = spy()["close"]
    with pytest.raises(ValueError, match=r"linreg\(\) slope must be True or False, got 1"):
        ta.linreg(close, slope=1)
    assert ta.linreg(close, slope=None).equals(ta.linreg(close))
    assert ta.adx(spy()["high"], spy()["low"], close, length=7).columns.tolist() == ["ADX_7", "DMP_7", "DMN_7"]
