Testing
=======

**Pandas TA Classic** uses a multi-layered testing strategy to ensure
indicator correctness, robustness, and reliability.

Test Layers
-----------

1. **Unit Tests** — Traditional deterministic tests for individual indicators
   and utilities, using real market data (``SPY_D.csv``) and
   ``IndicatorSpec``-based assertions.

2. **Edge-Case Tests** — Tests for degenerate inputs: all-NaN series,
   constant series, ±Inf values, and mismatched lengths.

3. **Property-Based Tests** — Randomized input testing using
   `Hypothesis <https://hypothesis.readthedocs.io/>`_ to discover edge
   cases that deterministic tests miss.

Property-Based Testing
----------------------

Property-based testing generates a wide range of inputs and verifies
invariants (properties) that must hold for all valid inputs. This
catches edge cases — like overflow conditions, NaN propagation bugs, and
boundary violations — that are hard to enumerate by hand.

Tested properties include:

* **Output invariants** — Type correctness, length preservation, naming
  conventions.
* **Mathematical invariants** — Bollinger Band ordering (lower ≤ mid ≤ upper),
  ATR non-negativity, standard deviation non-negativity.
* **Core utilities** — ``verify_series``, ``apply_offset``, ``apply_fill``
  behavior under diverse inputs.
* **None-guard safety** — All indicators return ``None`` (not raise) when
  passed ``None`` for required arguments.
* **NaN propagation** — All-NaN input → all-NaN output, no crash.
* **Idempotence** — Calling an indicator twice with the same arguments
  produces identical results.
* **Category discovery** — Dynamic indicator discovery stays consistent.

Running Property-Based Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all property-based tests
   python -m pytest tests/test_property_based.py -v

   # Show Hypothesis statistics (input distribution)
   python -m pytest tests/test_property_based.py -v --hypothesis-show-statistics

   # Use CI-optimized profile (more examples, longer deadline)
   python -m pytest tests/test_property_based.py -v --hypothesis-profile=ci

Strategy Design
~~~~~~~~~~~~~~~

Hypothesis strategies generate realistic price series by composing:

* **Random walks** — Cumulative sum of normal increments for price-like behavior.
* **OHLCV DataFrames** — Derived OHLC with high ≥ low constraints and volume.
* **Constant series** — For degenerate arithmetic testing.
* **Controlled NaN injection** — Finite floats with proportionally sampled NaN
  insertion at configurable rates.

These strategies are defined in ``tests/test_property_based.py`` and can be
reused for testing new indicators.

Adding Property Tests for a New Indicator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When adding a new indicator, consider testing these properties:

1. **Type and length** — Does the output have the expected type and length?
2. **Name convention** — Does the output name include the length parameter?
3. **None guard** — Does passing ``None`` return ``None`` without raising?
4. **Mathematical bounds** — Are there any invariants (e.g., oscillator ∈ [0,100])?
5. **Edge cases** — Does it handle NaN, constant, short, and zero-length inputs?

Example:

.. code-block:: python

   from hypothesis import given, settings, strategies as st

   @given(price_series(min_size=30, max_size=200), _small_positive_int)
   @settings(max_examples=100)
   def test_my_indicator_output_invariant(s, length):
       assume(len(s) >= length + 2)
       result = ta.my_indicator(s, length=length)
       assert isinstance(result, pd.Series)
       assert len(result) == len(s)
       assert str(length) in result.name

Running All Tests
-----------------

.. code-block:: bash

   # Full test suite (all layers)
   python -m unittest discover tests/ -v

   # pytest equivalent
   python -m pytest tests/ -v
