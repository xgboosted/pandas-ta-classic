"""Shared lazy-loading installer for category subpackages.

The 10 category subpackages (candles, cycles, math, momentum, overlap,
performance, statistics, trend, volatility, volume) each expose a flat
namespace where ``pandas_ta_classic.<cat>.<name>`` resolves to the
function ``<name>`` defined in ``pandas_ta_classic.<cat>.<name>.py``
(one indicator per submodule). To avoid importing every submodule at
package import time, each category installs a ``types.ModuleType``
subclass that resolves names lazily on first access.

This module factors that shared logic into :func:`install_lazy_subpackage`.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, Mapping, Optional


def install_lazy_subpackage(
    module_name: str,
    *,
    aliases: Optional[Mapping[str, str]] = None,
    special: Optional[Mapping[str, tuple]] = None,
) -> None:
    """Replace ``sys.modules[module_name]`` with a lazy-loading module.

    Parameters
    ----------
    module_name:
        Fully-qualified subpackage name (typically ``__name__`` from the
        caller).
    aliases:
        Mapping of requested attribute name to canonical submodule/attr
        name. Used by ``math`` so that ``max`` resolves to
        ``rolling_max`` (which lives in ``math/rolling_max.py``).
    special:
        Mapping of requested attribute name to ``(submodule, attr)``
        pairs where the attribute lives in a *different* submodule than
        the requested name. Used by ``candles`` so that ``ALL_PATTERNS``
        resolves to the ``ALL_PATTERNS`` attribute of
        ``candles/cdl_pattern.py``.
    """

    aliases_map: Mapping[str, str] = aliases or {}
    special_map: Mapping[str, tuple] = special or {}

    from pandas_ta_classic._meta import Category

    _cat = module_name.rsplit(".", 1)[-1]
    _known_names: frozenset[str] = frozenset(list(Category.get(_cat, [])) + list(aliases_map.keys()) + list(special_map.keys()))

    original = sys.modules[module_name]

    class _LazySubpackage(types.ModuleType):
        """Lazy-loading package module: resolves names to attrs in submodules."""

        def __setattr__(self, name: str, value: Any) -> None:
            # Python's import machinery binds imported submodules as attrs on the
            # parent package. Intercept those bindings and unwrap to the function.
            if name not in _known_names:
                object.__setattr__(self, name, value)
                return
            pkg = object.__getattribute__(self, "__name__")
            if isinstance(value, types.ModuleType) and value.__name__ == f"{pkg}.{name}":
                canonical = aliases_map.get(name, name)
                func = getattr(value, canonical, None)
                if func is not None and callable(func):
                    object.__setattr__(self, name, func)
                    return
                raise ImportError(f"module '{value.__name__}' has no callable attribute '{canonical}'")
            object.__setattr__(self, name, value)

        def __getattr__(self, name: str) -> Any:
            pkg = object.__getattribute__(self, "__name__")
            if name not in _known_names:
                raise AttributeError(f"module {pkg!r} has no attribute {name!r}")
            if name in special_map:
                submod_name, attr = special_map[name]
                mod = importlib.import_module(f"{pkg}.{submod_name}")
                value = getattr(mod, attr)
                object.__setattr__(self, name, value)
                return value
            canonical = aliases_map.get(name, name)
            try:
                mod = importlib.import_module(f"{pkg}.{canonical}")
                func = getattr(mod, canonical, None)
                if func is not None and callable(func):
                    object.__setattr__(self, name, func)
                    return func
            except ModuleNotFoundError:
                pass
            except ImportError:
                raise
            raise AttributeError(f"module {pkg!r} has no attribute {name!r}")

        def __dir__(self) -> list[str]:
            return sorted(_known_names)

    new_mod = _LazySubpackage(module_name)
    # Expose all lazily-loadable names so `from pkg import *` works without prior access.
    # Exclude aliases (e.g. max/min/sum) from __all__: wildcard import would
    # shadow Python builtins. They remain accessible via explicit attribute access.
    new_mod.__all__ = sorted(_known_names - aliases_map.keys())
    # Copy module attributes from the original so docstring, path, spec, etc. survive.
    new_mod.__doc__ = getattr(original, "__doc__", None)
    if hasattr(original, "__path__"):
        new_mod.__path__ = original.__path__  # type: ignore[attr-defined]
    new_mod.__spec__ = original.__spec__
    new_mod.__file__ = original.__file__
    new_mod.__package__ = original.__package__
    sys.modules[module_name] = new_mod
