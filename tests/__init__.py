"""Regenerate JSON fixture files before running any tests.

When TA-Lib is installed the expected-values and regression-snapshot
JSON files are rebuilt from the oracle library + the native code under
test.  This ensures the fixtures are always in sync with the current
algorithms.

If TA-Lib is not available the generators are skipped — existing
fixture files (which should be committed to the repo) will be used
as-is.
"""

import contextlib
import io
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import talib  # noqa: F401
except ImportError:
    # TA-Lib not installed — cannot regenerate; use committed fixtures.
    pass
else:
    from tests.fixtures.generate_fixtures import generate as _gen_fixtures
    from tests.fixtures.generate_regression_snapshots import generate as _gen_snapshots

    _null = io.StringIO()
    with contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
        _gen_fixtures()
        _gen_snapshots()
    print("[fixtures] regenerated expected_values.json + regression_snapshots.json")
