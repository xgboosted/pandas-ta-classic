"""What the built package ships besides the code."""

from pathlib import Path
from unittest import TestCase

import pandas_ta_classic as ta


class TestPep561Marker(TestCase):
    """The wheel shipped core.pyi but no py.typed, so a consumer's mypy ignored both.

    Proven from outside the repo against the installed wheel: before the marker,
    ``import pandas_ta_classic`` reported ``Skipping analyzing
    "pandas_ta_classic": module is installed, but missing library stubs or
    py.typed marker`` and every call through the stub went unchecked.
    """

    def test_py_typed_marker_ships_with_the_package(self):
        package_dir = Path(ta.__file__).parent
        marker = package_dir / "py.typed"
        self.assertTrue(marker.is_file(), "PEP 561 marker missing; consumers' mypy ignores core.pyi")
        # PEP 561 wants it empty; a "partial" line would mark a partial stub package.
        self.assertEqual(marker.read_bytes(), b"")
        self.assertTrue((package_dir / "core.pyi").is_file())
