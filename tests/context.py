import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas_ta_classic  # noqa: F401 — re-exported for "from tests.context import pandas_ta_classic"
