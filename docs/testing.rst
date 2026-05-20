Testing
=======

**Pandas TA Classic** uses a multi-layered testing strategy to ensure
indicator correctness, robustness, and reliability.

.. contents:: :local: :depth: 1


Unit Tests
----------

**Why:** Verify individual indicators produce correct values for known inputs.

**Files:** ``test_indicator_candle.py``, ``test_indicator_cycles.py``,
``test_indicator_momentum.py``, ``test_indicator_overlap.py``,
``test_indicator_performance.py``, ``test_indicator_statistics.py``,
``test_indicator_trend.py``, ``test_indicator_volatility.py``,
``test_indicator_volume.py``, ``test_indicator_math.py``.

Uses ``IndicatorSpec``-based assertions (``assert_indicator_standard``)
against real market data from ``SPY_D.csv``.

**Run:** ``python -m unittest tests.test_indicator_momentum -v``


Extension API Tests
-------------------

**Why:** Confirm indicators work correctly through the ``df.ta`` DataFrame
accessor with ``append=True``.

**Files:** ``test_ext_indicator_candle.py``, ``test_ext_indicator_cycles.py``,
``test_ext_indicator_momentum.py``, ``test_ext_indicator_overlap_ext.py``,
``test_ext_indicator_performance.py``, ``test_ext_indicator_statistics.py``,
``test_ext_indicator_trend.py``, ``test_ext_indicator_volatility.py``,
``test_ext_indicator_volume.py``.

**Run:** ``python -m pytest tests/test_ext_indicator_momentum.py -v``


Accessor API Tests
------------------

**Why:** Validate DataFrame accessor metadata and utilities: ``prefix``/``suffix``
naming, ``indicators()`` discovery, ``ticker()`` data fetching, time range
filtering, and ``constants()``.

**Files:** ``test_accessor_api.py``, ``test_ext_assertions.py``.

**Run:** ``python -m pytest tests/test_accessor_api.py -v``


Oracle / Comparison Tests
-------------------------

**Why:** Compare native (``talib=False``) implementations against
TA-Lib (C library) and tulipy outputs to catch numerical divergence.
Requires ``ta-lib`` and ``tulipy`` installed.

**Files:** ``test_oracle_talib.py``, ``test_oracle_tulipy.py``.

**Run:** ``python -m pytest tests/test_oracle_talib.py -v``


Native Indicator Tests
----------------------

**Why:** Cover ~45 indicators that have no TA-Lib alternative, validating
return type, non-NaN row count, value finiteness, and mathematical bounds.

**Files:** ``test_native_indicators.py``.

**Run:** ``python -m pytest tests/test_native_indicators.py -v``


Regression Tests
----------------

**Why:** Prevent reintroduction of known bugs and catch silent value drift.

- ``test_regression.py`` — Spot-checks indicator values at 5 fixed indices
  (50, 200, 500, 1500, 3000) against stored fixture data.
- ``test_regression_bugfixes.py`` — Pins ~12 documented fixes from CHANGELOG.
- ``test_indicator_values.py`` — Golden fixture tests: checks last non-NaN
  values and per-column NaN counts against snapshots in ``tests/fixtures/``.

**Run:** ``python -m pytest tests/test_regression.py -v``


Edge-Case Tests
---------------

**Why:** Verify indicators don't crash on degenerate inputs.

- ``test_indicator_edge_cases.py`` — All-NaN series, constant-price series,
  ±Inf injection at mid-series positions, and mismatched OHLCV lengths.
- ``test_nan_behaviour.py`` — NaN prefix warmup periods, minimum length
  requirements, boundary conditions.

**Run:** ``python -m pytest tests/test_indicator_edge_cases.py -v``


Integration / E2E Tests
-----------------------

**Why:** Exercise full workflows end-to-end.

**Files:** ``test_integration_e2e.py`` — Multi-indicator chaining,
Strategy execution with ``df.ta.strategy()``, plugin binding, and
category-strategy runs.

**Run:** ``python -m pytest tests/test_integration_e2e.py -v``


Fluent API Tests
----------------

**Why:** Validate the ``df.ta.chain()`` fluent programming API.

**Files:** ``test_fluent_chaining.py`` — Chained indicator calls,
auto-append behaviour, ``unchain()``.

**Run:** ``python -m pytest tests/test_fluent_chaining.py -v``


Strategy Tests
--------------

**Why:** Confirm the ``Strategy`` class executes correctly, including
multi-core processing.

**Files:** ``test_strategy.py`` (runs separately from the main suite).

**Run:** ``python -m pytest tests/test_strategy.py -v``


Custom / Plugin Tests
---------------------

**Why:** Verify the custom indicator registration system.

**Files:** ``test_custom.py`` — ``ta.custom.bind()``, ``import_dir()``,
module loading, and custom indicator discovery.

**Run:** ``python -m pytest tests/test_custom.py -v``


Property-Based Tests
--------------------

**Why:** Randomized input testing using `Hypothesis
<https://hypothesis.readthedocs.io/>`_ to discover edge cases that
deterministic tests miss — overflow conditions, NaN propagation bugs,
boundary violations.

**Files:** ``test_property_based.py``.

**What's tested:**

* **Output invariants** — Type correctness, length preservation, naming.
* **Mathematical invariants** — Bollinger Band ordering, ATR/STDEV
  non-negativity, MOM/ROC relationship.
* **Core utilities** — ``verify_series``, ``apply_offset``, ``apply_fill``.
* **None-guard safety** — 7 indicators return ``None`` for ``None`` input.
* **NaN propagation** — All-NaN input → all-NaN output, no crash.
* **Idempotence** — Same args twice → identical result.
* **Category discovery** — Dynamic discovery stays consistent.
* **Boundedness** — RSI, stochastic oscillator within expected ranges
  (where input assumptions hold).

**Strategies used:**

* Random walks — Cumulative sum of normal increments.
* OHLCV DataFrames — Derived OHLC with high ≥ low, close ∈ [low, high].
* Constant series — Degenerate arithmetic testing.
* Controlled NaN injection — Finite floats with proportionally sampled NaN.

**Run:**

.. code-block:: bash

   python -m pytest tests/test_property_based.py -v
   python -m pytest tests/test_property_based.py -v --hypothesis-show-statistics
   python -m pytest tests/test_property_based.py -v --hypothesis-profile=ci

**Adding property tests for a new indicator:**

.. code-block:: python

   from hypothesis import assume, given, settings

   @given(price_series(min_size=30, max_size=200), _small_positive_int)
   @settings(max_examples=100)
   def test_my_indicator_output_invariant(s, length):
       assume(len(s) >= length + 2)
       result = ta.my_indicator(s, length=length)
       assert isinstance(result, pd.Series)
       assert len(result) == len(s)
       assert str(length) in result.name


Utility Tests
-------------

**Files:** ``test_utils.py`` (``verify_series``, ``apply_offset``,
``apply_fill``, cross detection), ``test_utils_metrics.py`` (Sharpe ratio,
drawdown, CAGR, Jensen's alpha), ``test_utils_data_alphavantage.py``
(AlphaVantage data fetching).

**Run:** ``python -m pytest tests/test_utils.py -v``


Running All Tests
-----------------

.. code-block:: bash

   # Full test suite (primary — matches CI)
   python -m unittest discover tests/ -v

   # pytest equivalent
   python -m pytest tests/ -v

   # With coverage
   python -m pytest --cov=pandas_ta_classic --cov-report=html tests/
