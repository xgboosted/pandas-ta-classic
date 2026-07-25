"""Charting colour-palette helper for the example notebooks.

This module previously also bundled a ``Watchlist`` class that downloaded market
data via yfinance / Alpha Vantage. That has been removed: data fetching is out
of scope for pandas-ta-classic. Fetch OHLCV with yfinance / alpha-vantage
directly and pass the DataFrame in — see ``examples/fetch_market_data.py``.

Only the ``colors()`` palette helper remains, used by the plotting cells in the
example notebooks.
"""


def colors(colors: str | None = None, default: str = "GrRd"):
    """Return a list of matplotlib colour names for the given palette alias."""
    aliases = {
        # Pairs
        "BkGy": ["black", "gray"],
        "BkSv": ["black", "silver"],
        "BkPr": ["black", "purple"],
        "BkBl": ["black", "blue"],
        "FcLi": ["fuchsia", "lime"],
        "GrRd": ["green", "red"],
        "GyBk": ["gray", "black"],
        "GyBl": ["gray", "blue"],
        "GyOr": ["gray", "orange"],
        "GyPr": ["gray", "purple"],
        "GySv": ["gray", "silver"],
        "RdGr": ["red", "green"],
        "SvGy": ["silver", "gray"],
        # Triples
        "BkGrRd": ["black", "green", "red"],
        "BkBlPr": ["black", "blue", "purple"],
        "GrOrRd": ["green", "orange", "red"],
        "RdOrGr": ["red", "orange", "green"],
        # Quads
        "BkGrOrRd": ["black", "green", "orange", "red"],
        # Quints
        "BkGrOrRdMr": ["black", "green", "orange", "red", "maroon"],
        # Indicators
        "bbands": ["blue", "navy", "blue"],
        "kc": ["purple", "fuchsia", "purple"],
    }
    aliases["default"] = aliases[default]
    if colors in aliases:
        return aliases[colors]
    return aliases["default"]
