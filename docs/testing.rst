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
naming, ``indicators()`` discovery, and time range filtering.
``test_accessor_conformance.py`` calls every registered indicator through
``df.ta``: each returns a Series or DataFrame, and one whose required input the
frame cannot supply (a benchmark, a ``fast``/``slow`` pair, a trend) raises.

**Files:** ``test_accessor_api.py``, ``test_accessor_conformance.py``,
``test_ext_assertions.py``.

**Run:** ``python -m pytest tests/test_accessor_api.py -v``


Oracle / Comparison Tests
-------------------------

**Why:** Compare native (``talib=False``) implementations against
TA-Lib (C library) and tulipy outputs to catch numerical divergence.
``test_oracle_talib.py`` needs ``ta-lib`` installed and skips without it;
``test_oracle_tulipy.py`` compares against tulipy values frozen in
``tests/fixtures/tulipy_oracle.json``, so tulipy itself is not needed.
No oracle test lets ``talib=True`` reach TA-Lib: that would compare TA-Lib
with itself. (``test_oracle_registry.py`` passes it once, for ``cmo``, together
with a ``scalar`` TA-Lib cannot express, which keeps the call native.)

``test_oracle_registry.py`` compares every indicator and candle pattern with a
TA-Lib counterpart from one table, on ``SPY_D`` and on frames that plant the
data problems met so far: a fully flat series, a flat block, ``high == low``
with a moving close, scattered flat and bodiless bars, malformed OHLC,
zero-volume sessions, prices scaled by ``1e-6`` and ``1e6``, a flash spike and
crash, and very large volume. It compares the NaN mask as well as the values, from the
first bar both sides hold a value, so native giving up where TA-Lib answers is
a failure rather than a bar dropped by ``dropna()``. An indicator listed with
TA-Lib in ``indicator_support_matrix.rst`` must have a case in the table or a
reason in ``NO_ORACLE``; known differences are strict xfails in
``KNOWN_DIFFERENCES``.

For indicators neither library covers, ``test_reference_ports.py`` compares
against independent ports of the cited definitions (see *Reference ports*
below). ``test_adversarial_review_fixes.py`` pins the defects found by the
2026-09 review, each against a loop reference, TA-Lib or a decision table.
``test_talib_parameters.py`` checks that ``talib=True`` honours every
parameter: a call with a value TA-Lib cannot express (``scalar``, ``drift``,
``mamode``, ``ddof``, ``min_periods``, ``uo``'s weights, an optional price
input, an option read from ``**kwargs`` or handed on to another function)
runs natively and equals ``talib=False`` value for value. It also checks that
both paths publish each column's first value on the same bar, with TA-Lib's
by-design exceptions listed with their reason.

**Files:** ``test_oracle_talib.py``, ``test_oracle_registry.py``,
``test_oracle_tulipy.py``, ``test_reference_ports.py``,
``test_adversarial_review_fixes.py``, ``test_talib_parameters.py``.

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
- ``test_regression_bugfixes.py`` — Pins documented fixes from CHANGELOG
  (33 test classes, 128 tests).
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
- ``test_lookahead.py`` — for every registered indicator, at its defaults
  and with each boolean option flipped, the first ``K`` bars read the same
  over ``K`` bars as over the whole series, on data that plants flat and
  other pathological bars after the last cut point. The calls that look
  forward on purpose (centred ``dpo``, the Ichimoku spans,
  ``cpr(virgin_cpr=True)``, ``tos_stdevall``, ``vp``) are strict xfails;
  ``lookahead=False`` turns the first three causal, and ``tos_stdevall`` and
  ``vp``, which have no causal mode, decline it.
- ``test_short_input_contract.py`` — for every registered indicator, input
  shorter than its window returns an all-NaN result indexed like the input
  and shaped like a normal one (it returned ``None`` before 0.9.0); an
  indicator with no window, or a window the input fits, returns values.
- ``test_leading_nan_contract.py`` — for every registered indicator, a
  leading NaN run (chained input) leaves the result on the real bars
  unchanged, the padded bars themselves read NaN, and an all-NaN input reads
  NaN throughout; candle patterns, which read 0 there as TA-Lib's do, are
  pinned as decided.
- ``test_interior_nan_contract.py`` — for every registered indicator, one
  missing bar inside the series recovers, or differs persistently only for
  cumulative indicators and whole-series fits (see "Missing Values" in the
  indicator reference).
- ``test_warmup_contract.py`` — for every registered indicator, no column
  holds a value before its own warm-up, bar-describing flag columns are
  NaN wherever the continuous columns of the same result are, and the first
  values differ between five unrelated random price paths, so a recursion's
  seed or a placeholder (a fixed 0.0 or 50.0) is never published as a result.
- ``test_nullable_input_contract.py`` — for every registered indicator,
  nullable ``Float64`` / ``Int64`` input gives the same result, dtype
  included, as the same data in float64, with ``pd.NA`` read as NaN; so do the
  indicators that take derived series and, where pyarrow is installed,
  ``double[pyarrow]`` / ``int64[pyarrow]`` input.
- ``test_degenerate_input.py`` — for every registered indicator, a flat
  series, a flat block, bars with no range, malformed OHLC and zero-volume
  sessions give no all-NaN column, no infinity, no output inflated beyond the
  control frame, no NaN bleeding past the degenerate part and no
  epsilon-scale value.
- ``test_data_gaps_contract.py`` — for every registered indicator, a missing
  whole bar, a five-bar outage, a bar missing only its volume or only its
  range, and a missing newest bar leave earlier bars untouched and are
  forgotten once they leave every window; the same bars under a gappy
  calendar give the same numbers unless the indicator anchors to the calendar.
- ``test_scale_contract.py`` — for every registered indicator, prices scaled
  by ``1e-6`` and ``1e6`` leave each column unchanged or scale it by ``k`` or
  ``k**2``; an absolute constant hidden in a formula breaks this.
- ``test_shift_contract.py`` — for every registered indicator, prices moved
  by a constant (staying positive, crossing zero, all negative) leave each
  column unchanged or move it by the same constant, so the indicator works
  on negative prices; indicators relative to the price level by definition
  (percent changes, log returns, ratios to the close) are listed with the
  reason.
- ``test_index_contract.py`` — for every registered indicator, the result
  keeps the input's length and index, ``offset`` is a plain shift of every
  column, ``fillna`` reaches every column, a repeated index label gives the
  same values as a unique index (with and without a leading NaN run), and
  ``result.category`` names the category the indicator is registered under.
  The ``offset`` and ``fillna`` checks run on every option path: each
  ``mamode``, ``talib=True`` and each boolean option.
- ``test_int_parameter_contract.py`` — for every registered indicator, each
  whole-number parameter set to 2, 3 and 5 leaves values in every column the
  default call fills.
- ``test_option_effect_contract.py`` — for every registered indicator, each
  MA choice changes every column it smooths, and each boolean parameter
  changes the values or the column names when flipped, so an option that is
  accepted but never read fails.
- ``test_parameter_validation.py`` — besides rejecting invalid values, every
  numeric default of every registered indicator is passed explicitly, as an
  int and as a float, and must name the output as the default call does.

The data-problem modules keep their known exceptions in a table of strict xfails with the
reason, so an entry that starts passing fails until it is removed.

**Run:** ``python -m pytest tests/test_indicator_edge_cases.py -v``


API Consistency Map
-------------------

**Why:** Keep an up-to-date list of where the public API is not uniform.

``test_api_consistency.py`` asks registry-wide questions of the API: one table
per question (output dtype, flag encoding, float32 input, all-NaN input,
duplicate or unsorted index, column-name style, explicit defaults, signature
defaults, docstrings, accessor vs function, parameter names, percent vs
fraction, shared column names, names shadowing builtins, pandas or the stdlib),
each listing the indicators that answer differently from the majority. A new
deviation fails until it is added to its table, and a fixed one fails until
its row is removed, so the tables are an up-to-date map of where the API is
not uniform.

Each table carries a status: ``OPEN:`` for a defect or an undecided question,
``KEPT:`` for a decided exception; single rows can override their table's
status, and a failure names the status of every row it reports.

**Run:** ``python -m pytest tests/test_api_consistency.py -v``


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
serial and parallel execution, and chained Custom Strategies, which run in
ordered stages so the result does not depend on the core count.

``test_strategy_execution.py`` runs the same strategies serially and on an
executor and requires identical frames, and checks the ordered stages and the
argument checks of ``strategy()``.

**Files:** ``test_strategy.py`` (part of the main suite),
``test_strategy_execution.py``.

**Run:** ``python -m pytest tests/test_strategy.py tests/test_strategy_execution.py -v``


Custom / Plugin Tests
---------------------

**Why:** Verify the custom indicator registration system.

**Files:** ``test_custom.py`` — ``ta.custom.bind()``, ``import_dir()``,
module loading, and custom indicator discovery. ``import_dir()`` is all or
nothing: a Hypothesis property checks that one refused file leaves
``pandas_ta_classic``, ``Category``, ``sys.path`` and ``sys.modules`` as they
were.

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


JIT Parity Tests
----------------

**Why:** ``@njit`` kernels run compiled when numba is installed and as plain
Python otherwise, so users with the ``performance`` extra run different code.
``test_numba_parity.py`` compares every indicator and candle pattern between
the two paths (1e-12 relative). It skips without numba; the CI job
``testing-numba`` installs it.

**Run:** ``python -m pytest tests/test_numba_parity.py -v``


Utility Tests
-------------

**Files:** ``test_utils.py`` (``verify_series``, ``apply_offset``,
``apply_fill``, cross detection), ``test_utils_metrics.py`` (Sharpe ratio,
drawdown, CAGR, Jensen's alpha), ``test_lazy_core.py`` (lazy loading of the
category subpackages and of top-level indicator names).

**Run:** ``python -m pytest tests/test_utils.py -v``


Repository Guards
-----------------

**Why:** Keep the rules in ``AGENTS.md`` true across the whole repository, not
only in the code a PR touches.

- ``test_no_fallbacks.py`` — each place that used to substitute a value, return
  ``None`` or hide an error now raises (rule 1).
- ``test_docstring_defaults.py`` — every ``Default:`` in an indicator docstring
  equals the default the code applies (rule 8).
- ``test_docstring_params.py`` — every parameter a docstring documents is one
  the indicator reads.
- ``test_assertions.py`` — the comparison helpers in ``tests/assertions.py``
  fail on wrong results, so the tests built on them can fail.
- ``test_no_invisible_characters.py`` — no tracked file holds an invisible or
  confusable character that a diff or a grep would not show.
- ``test_python_support_docs.py`` — the supported Python range in the docs and
  ``requires-python`` match ``LATEST_PYTHON_VERSION`` in CI (rule 9).
- ``test_changelog.py`` — each fragment in ``changelog.d/`` has a valid name, at
  most 1,000 characters and the **BREAKING** form, and ``CHANGELOG.md`` keeps the
  marker towncrier inserts after.

**Run:** ``python -m pytest tests/test_no_fallbacks.py tests/test_changelog.py -v``


Running All Tests
-----------------

.. code-block:: bash

   # Full test suite (primary — matches CI)
   python -m pytest tests/ -v

   # Same, via make
   make test-all

   # Regenerate fixture JSONs (requires TA-Lib; only after an intentional
   # algorithm change — review the diff before committing)
   make fixtures

   # Do not use `python -m unittest discover`: it skips the pytest-style
   # tests (lookahead, NaN contracts, numba parity, reference ports).

   # With coverage
   python -m pytest --cov=pandas_ta_classic --cov-report=html tests/

**Warnings are errors.** ``[tool.pytest.ini_options] filterwarnings`` in
``pyproject.toml`` turns any ``UserWarning``, ``DeprecationWarning`` or
``FutureWarning`` raised from ``pandas_ta_classic`` into a test failure. 0.9.0
ships no deprecation, and a deprecated pandas or numpy call made from the
package (such as pandas 3's ``Pandas4Warning``) fails CI instead of reaching
users. A test that expects a warning asserts it with ``pytest.warns``.


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
orders of margin for platform and BLAS differences across the 3.11–3.15 CI
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
output, so it detects *change*, never *correctness*.  The fix is a
deliberately naive reference implementation per indicator, written from the
definition the indicator cites, added incrementally.

Reference ports
~~~~~~~~~~~~~~~

``tests/fixtures/reference_ports.py`` holds plain-loop ports of cited
definitions (TradingView built-ins and scripts, ProRealCode, the original
papers), written without any package code.  ``tests/test_reference_ports.py``
compares the package with them over the full SPY series, after the warm-up.
They cover ``kst``, ``smi``, ``stc``, ``qqe``, ``squeeze_pro``,
``ttm_trend``, ``cksp``, ``vidya``, ``ichimoku``, ``squeeze``, ``supertrend``
(direction), daily ``cpr``, ``vwap``, ``maxindex``, ``minindex``,
``minmaxindex``, ``vfi`` and ``ebsw``.  Six of these were wrong against
their sources when first ported (``kst``, ``stc``, ``cksp``, ``ttm_trend``,
``vfi``, ``ebsw``).

Still regression-only, with no independent check of their values:
``aberration``, ``amat``, ``aobv``, ``brar``, ``cdl_inside``, ``cdl_z``,
``ce``, ``cti``, ``decay``, ``dsp``, ``hilo``, ``hwc``, ``hwma``,
``inertia``, ``jma``, ``kdj``, ``long_run``, ``lrsi``, ``mcgd``, ``mmar``,
``pmax``, ``psl``, ``rainbow``, ``rsx``, ``rvgi``, ``rvi_vol``,
``short_run``, ``ssf``, ``td_seq``, ``thermo``, ``tos_stdevall``, ``trixh``,
``tsignals``, ``vp``, ``vwmacd``, ``xsignals``, ``zlma``.
Add a port to ``reference_ports.py`` to move one off this list.
