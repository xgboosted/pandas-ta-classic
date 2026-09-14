"""Test package.

``from tests.config import ...`` resolves without a ``sys.path`` hack: pytest
puts the project root on the path for this rootdir package, and the package
itself is installed in editable mode.

Fixture files are **not** regenerated here.  ``tests/fixtures/*.json`` are
frozen golden values: they are the source of truth, not a cache of a
computation.  Regenerating them as a side effect of running the tests
destroys the two properties they exist to provide —

* a golden value that the test run rewrites cannot detect a development
  error, because the buggy output simply becomes the new expectation;
* a golden value recomputed from ``pandas``/TA-Lib inherits those
  libraries' version-to-version floating-point drift, so the file churns
  on dependency upgrades that changed nothing in this package.

Regeneration is a deliberate, reviewed step — run ``make fixtures`` (or
``python -m tests.fixtures.generate_fixtures``) after an intentional
algorithm change and commit the resulting diff alongside it.
"""
