#!/usr/bin/env python3
"""
Migrate core.py indicator wrapper methods to use auto-dispatch via __getattr__.

Removes ~130 boilerplate wrapper methods and replaces them with:
  - _make_indicator_method() factory at module level
  - __getattr__() on AnalysisIndicators for lazy auto-dispatch

Special-case methods that cannot be auto-generated are preserved verbatim.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CORE_PY = REPO_ROOT / "pandas_ta_classic" / "core.py"

# These methods deviate from the standard pattern and must stay explicit.
SPECIAL_METHODS = {
    # Tuple return — cannot use _post_process directly
    "ichimoku",
    # Conditional optional series params (open_=None)
    "ad", "adosc", "cmf", "psl",
    # Conditional optional series params (high=None, low=None)
    "inertia",
    # Early return when primary arg is None; non-OHLCV positional arg
    "long_run", "short_run", "tsignals", "xsignals",
    # Datetime index mutation before calling underlying function
    "vwap",
    # Non-OHLCV series params (series_a, series_b) — not extractable by name
    "above", "above_value", "below", "below_value", "cross", "cross_value",
}

# Module-level infrastructure inserted just before the class definition block
# (before the comment + decorator + class).
_MODULE_INFRASTRUCTURE = """\
import inspect as _inspect

# ─────────────────────────────────────────────────────────────────────────────
# Auto-wrapper infrastructure
# Maps indicator function parameter names that represent OHLCV Series to the
# corresponding DataFrame column name used in kwargs.pop(col_name, col_name).
# ─────────────────────────────────────────────────────────────────────────────
_SERIES_PARAM_MAP = {
    "open_": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


def _make_indicator_method(func):
    \"\"\"Create an AnalysisIndicators method that auto-wraps an indicator function.

    Inspects *func*'s signature to determine which parameters are OHLCV Series
    (i.e. named ``open_``, ``high``, ``low``, ``close``, or ``volume`` **and**
    are required — no ``None`` default).  Those series are extracted from the
    DataFrame via ``_get_column``; all remaining kwargs are forwarded to the
    underlying function unchanged.

    The generated method is cached on the class after first use so that
    subsequent calls bypass ``__getattr__``.
    \"\"\"
    sig = _inspect.signature(func)
    # Collect (param_name, df_col_name) pairs for required series params only.
    # Optional series (default=None) are left to explicit special-case methods.
    series_params = [
        (pname, _SERIES_PARAM_MAP[pname])
        for pname, param in sig.parameters.items()
        if pname in _SERIES_PARAM_MAP
        and param.default is _inspect.Parameter.empty
    ]

    def method(self, **kwargs):
        call_kwargs = {}
        for param_name, col_name in series_params:
            col_key = kwargs.pop(col_name, col_name)
            call_kwargs[param_name] = self._get_column(col_key)
        result = func(**call_kwargs, **kwargs)
        return self._post_process(result, **kwargs)

    method.__name__ = func.__name__
    method.__qualname__ = f"AnalysisIndicators.{func.__name__}"
    method.__doc__ = func.__doc__
    return method


"""

# __getattr__ method body (8-space indented for class methods).
_GETATTR_BODY = """\
    def __getattr__(self, name: str):
        \"\"\"Auto-dispatch to indicator functions without explicit wrapper methods.

        Any indicator registered in ``Category`` that does not have an explicit
        wrapper method defined on this class is wrapped on-the-fly via
        :func:`_make_indicator_method`.  The resulting bound method is cached on
        the class so that subsequent lookups bypass ``__getattr__`` entirely.
        \"\"\"
        # Bail out immediately for private/dunder names to avoid recursion.
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        # Confirm it is a known indicator; give a proper AttributeError otherwise.
        if not any(name in cats for cats in Category.values()):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        func = globals().get(name)
        if func is None or not callable(func):
            raise AttributeError(
                f"'{type(self).__name__}': indicator '{name}' is registered in "
                f"Category but not available in the module namespace"
            )
        method = _make_indicator_method(func)
        # Cache on the class so __getattr__ is not called again for this name.
        setattr(type(self), name, method)
        return method.__get__(self)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

_CATEGORY_RE = re.compile(r"^    # ([A-Z][A-Za-z ]+)$")
_METHOD_RE = re.compile(r"^    def (\w+)\(")


def _parse_wrapper_section(lines):
    """Split the wrapper section into a list of annotated entries.

    Returns a list of dicts:
        {"kind": "method", "name": str, "body": list[str]}
        {"kind": "block",  "name": None, "body": list[str]}

    A "block" entry is any run of lines between method definitions (category
    comments, blank lines, etc.).
    """
    entries = []
    cur_kind = "block"
    cur_name = None
    cur_body = []

    for line in lines:
        m = _METHOD_RE.match(line)
        if m:
            # Flush current accumulator.
            if cur_body or cur_name:
                entries.append({"kind": cur_kind, "name": cur_name, "body": cur_body})
            cur_kind = "method"
            cur_name = m.group(1)
            cur_body = [line]
        else:
            cur_body.append(line)

    if cur_body or cur_name:
        entries.append({"kind": cur_kind, "name": cur_name, "body": cur_body})

    return entries


def _last_category_comment(body_lines):
    """Return the category comment line from a block's trailing lines, or None."""
    for line in reversed(body_lines):
        stripped = line.strip()
        if not stripped:
            continue  # skip blank lines
        if _CATEGORY_RE.match(line):
            return line
        break  # first non-blank, non-category line — stop
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main transformation
# ─────────────────────────────────────────────────────────────────────────────

def main():
    src = CORE_PY.read_text(encoding="utf-8")
    original_lines = src.splitlines()

    # ── Locate the indicator-wrapper anchor. ─────────────────────────────────
    ANCHOR = "    # Public DataFrame Methods: Indicators and Utilities"
    try:
        anchor_idx = next(
            i for i, l in enumerate(original_lines) if l == ANCHOR
        )
    except StopIteration:
        sys.exit(f"[ERROR] Anchor not found: {ANCHOR!r}")

    infra_lines = original_lines[:anchor_idx]
    wrapper_lines = original_lines[anchor_idx:]

    # ── Parse wrapper section into annotated entries. ─────────────────────────
    entries = _parse_wrapper_section(wrapper_lines)

    # ── Build new wrapper section. ────────────────────────────────────────────
    new_wrapper = [
        "    # Public DataFrame Methods: Indicators and Utilities",
        "    #",
        "    # Standard indicator wrappers are auto-generated via __getattr__ +",
        "    # _make_indicator_method().  Only special-case methods that cannot",
        "    # be auto-dispatched are defined explicitly below.",
        "",
        _GETATTR_BODY.rstrip("\n"),
    ]

    current_category = None

    for entry in entries:
        if entry["kind"] == "block":
            # Non-method block: may contain a category comment at the end.
            cat = _last_category_comment(entry["body"])
            if cat:
                current_category = cat
            continue

        # Method entry.
        if entry["name"] not in SPECIAL_METHODS:
            # Still check the method's trailing lines for a category comment
            # (methods accumulate trailing whitespace + next-category comment).
            cat = _last_category_comment(entry["body"])
            if cat:
                current_category = cat
            continue  # discard standard wrapper

        # Emit category header (once) before this special method.
        if current_category is not None:
            new_wrapper.append("")
            new_wrapper.append(current_category)
            current_category = None  # reset so we don't emit it again

        new_wrapper.append("")
        new_wrapper.extend(entry["body"])

    # ── Insert module-level infrastructure before the class block. ────────────
    # Find the "# Pandas TA - DataFrame Analysis Indicators" comment that
    # precedes the @decorator + class definition so we can insert just before.
    CLASS_COMMENT = "# Pandas TA - DataFrame Analysis Indicators"
    try:
        insert_idx = next(
            i for i, l in enumerate(infra_lines) if l.strip() == CLASS_COMMENT
        )
    except StopIteration:
        # Fallback: insert just before @pd.api.extensions decorator.
        try:
            insert_idx = next(
                i for i, l in enumerate(infra_lines)
                if l.startswith("@pd.api.extensions.register_dataframe_accessor")
            )
        except StopIteration:
            sys.exit("[ERROR] Could not find class definition anchor in infra section")

    new_infra = (
        infra_lines[:insert_idx]
        + _MODULE_INFRASTRUCTURE.splitlines()
        + infra_lines[insert_idx:]
    )

    # ── Assemble and write. ───────────────────────────────────────────────────
    final_lines = new_infra + new_wrapper
    output = "\n".join(final_lines) + "\n"
    CORE_PY.write_text(output, encoding="utf-8")
    print(f"[OK] Wrote {len(final_lines)} lines to {CORE_PY}")
    print(f"     (was {len(original_lines)} lines; removed {len(original_lines) - len(final_lines)})")


if __name__ == "__main__":
    main()
