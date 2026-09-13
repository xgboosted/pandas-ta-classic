Testing
=======

**Pandas TA Classic** uses a multi-layered testing strategy to ensure
indicator correctness, robustness, and reliability.

.. contents::
   :local:
   :depth: 1


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

**Why:** Cover indicators that have no TA-Lib alternative, validating
return type, non-NaN row count, value finiteness, and mathematical bounds.

**Files:** ``test_native_indicators.py``.

**Run:** ``python -m pytest tests/test_native_indicators.py -v``


Regression Tests
----------------

**Why:** Prevent reintroduction of known bugs and catch silent value drift.

- ``test_regression.py`` — Spot-checks all 223 tracked indicators at 24 fixed
  indices spanning 50 to 5221 (the final bar) against stored snapshot data.
  Test methods are generated from ``regression_snapshots.json`` itself, so the
  asserted set cannot drift from the stored set.
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
* **None-guard safety** — Indicators return ``None`` for ``None`` input.
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

   import hypothesis.strategies as st
   from hypothesis import assume, given, settings

   @given(price_series(min_size=30, max_size=200), st.integers(min_value=2, max_value=20))
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

   # Same, via make
   make test-all

   # Regenerate fixture JSONs (requires TA-Lib; only after an intentional
   # algorithm change — review the diff before committing)
   make fixtures

   # pytest equivalent
   python -m pytest tests/ -v

   # With coverage
   python -m pytest --cov=pandas_ta_classic --cov-report=html tests/


Fixture Files
-------------

``tests/fixtures/expected_values.json`` and
``tests/fixtures/regression_snapshots.json`` are **frozen** golden files.
They are the source of truth and are never rewritten by a test run.
Regeneration is a deliberate, reviewed step, performed only when an
indicator algorithm changed on purpose:

.. code-block:: bash

   python -m tests.fixtures.generate_fixtures
   python -m tests.fixtures.generate_regression_snapshots

Both scripts can also be invoked directly (``python tests/fixtures/generate_*.py``)
and require the project root to be on ``sys.path``.  Always review
``git diff tests/fixtures/`` before committing the result.

Why they must stay frozen
~~~~~~~~~~~~~~~~~~~~~~~~~

The golden files exist to provide two properties, and regenerating them
during a test run destroys both:

*Detecting development errors.*  If the test run rewrites the expectation
from the code under test, a bug simply becomes the new expectation and the
test passes.  This is not hypothetical — with regeneration enabled, changing
the kurtosis excess-adjustment constant from ``3.0`` to ``2.9`` (a ~5 %
error) still passed ``test_regression.py``, and a 7 % error injected into
``psl`` passed every fixture test, because both JSON files were silently
rewritten first.

*Independence from dependency versions.*  ``expected_values.json`` derives
166 of its 223 entries from an external reference.  Recomputing at test time
imports that reference's floating-point behaviour, and one of those
references used to be ``pandas`` rolling: ``pandas`` 3.x and 2.x disagree on
``rolling().kurt()`` in the 8th decimal, because ``roll_kurt`` accumulates
running power sums whose error grows with series length (≈1.5e-8 over 5222
rows on pandas 3.0, ≈6.9e-10 on pandas 2.3, versus ≈1e-14 for this
package's own scratch-recomputed ``np_rolling_moments``).  A frozen literal
has neither problem.

Exact reference values
~~~~~~~~~~~~~~~~~~~~~~

``tests/fixtures/exact_reference.py`` replaces the ``pandas`` rolling oracles
for the statistics group (zscore, kurtosis, skew, median, quantile, mad,
entropy, beta, ui).  It reads the exact decimal written in the CSV
(``Fraction(str(x))``, not the float64 approximation), evaluates the textbook
formula in exact rational arithmetic, and converts to float only at the end.
The result is the mathematically correct value for the input data, identical
on every platform and dependency version, and still derived independently of
the code under test.

``tests/test_exact_reference.py`` guards that module — CI would otherwise
never execute it, since it only runs during ``make fixtures``.  Its
comparison against ``pandas`` rolling is a deliberately loose smoke test
(``atol=1e-4``, ``rtol=1e-3``), sized to clear pandas' own divergence from
exact arithmetic, which reaches 1.7e-5 absolute on ``kurt`` across the SPY
series.  It must never be tightened into an equality check.

Comparison tolerance
~~~~~~~~~~~~~~~~~~~~

``test_indicator_values.py`` and ``test_regression.py`` share one criterion,
defined in ``tests/assertions.py``::

    |actual - golden| <= GOLDEN_ATOL + GOLDEN_RTOL * |golden|
    GOLDEN_ATOL = 1e-8
    GOLDEN_RTOL = 1e-6

Both terms are load-bearing.  The JSON files store ``round(v, 8)``, so no
comparison can be tighter than the last stored decimal — that is the absolute
floor.  The relative term covers large-magnitude indicators where float64
cannot represent 8 decimals at all: ``ad`` peaks near 3.8e10, where one ULP is
already ~7.6e-6.  With only the absolute term the large columns overrun by
~2.7e4x; with only the relative term the small ones overrun on storage
rounding.

Measured across all 423 tracked columns the worst native-vs-golden
disagreement needs a relative term of 3.6e-15, so ``GOLDEN_RTOL`` keeps nine
orders of margin for platform and BLAS differences across the 3.10–3.14 CI
matrix.  Snapshot checkpoints use at most 29 % of the budget.

This replaced a flat ``REL_TOL = 1e-4``, which was ~10 orders looser than the
data required.  Concretely: perturbing the kurtosis excess-adjustment constant
from ``3.0`` to ``3.000001`` moves the golden value by 1.2e-6 — inside the old
1e-5 budget and therefore invisible, but 11x over the new one.

Snapshot coverage
~~~~~~~~~~~~~~~~~

``regression_snapshots.json`` stores all 223 tracked indicators, but
``test_regression.py`` used to assert a hand-written list of 43 — the other
180 were generated and never checked.  That gap covered 55 of the 57
indicators which have no independent oracle and are therefore protected by
their snapshot alone.  Test methods are now generated from the snapshot keys,
so the two cannot diverge again.

Indicators whose output is shorter than the input (``vp`` aggregates into 10
volume bins, ``tos_stdevall`` into 30) have every checkpoint past the end of
their result.  Those store ``null`` and are asserted to stay out of range,
which makes the checkpoint a length-regression check; their values are
covered by ``test_indicator_values.py`` instead.

Note what a snapshot can and cannot do.  It is taken from this package's own
output, so it detects *change*, never *correctness*.  The 57 oracle-less
indicators still have no independent verification of their mathematics — the
long-term fix is a deliberately naive reference implementation per indicator,
written from the published formula, added incrementally.
