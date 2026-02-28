#!/usr/bin/env python3
"""Migrate offset/fill boilerplate to apply_offset() calls.

Detects and replaces blocks of the form:

    # Offset
    if offset != 0:
        VAR = VAR.shift(offset)
        [more vars...]

    # Handle fills
    if "fillna" in kwargs:
        VAR.fillna(kwargs["fillna"], inplace=True)
        [more vars...]
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:
            if kwargs["fill_method"] == "ffill":
                VAR.ffill(inplace=True)
            elif kwargs["fill_method"] == "bfill":
                VAR.bfill(inplace=True)
        [repeats per var...]

With:

    # Offset
    VAR = apply_offset(VAR, offset, **kwargs)
    [more vars...]

Usage:
    python scripts/migrate_apply_offset.py [--dry-run]
"""

import re
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent / "pandas_ta_classic"
CATEGORIES = [
    "momentum",
    "overlap",
    "trend",
    "volume",
    "volatility",
    "statistics",
    "candles",
    "cycles",
    "performance",
]

IMPORT_STMT = "from pandas_ta_classic.utils import apply_offset"

# Matches "    VAR = VAR.shift(offset)" — both same-name assignment patterns
SHIFT_RE = re.compile(r"^(\s+)(\w+)\s*=\s*\2\.shift\(offset\)\s*$")


def parse_boilerplate(lines: list[str], start: int, base_indent: int):
    """Parse the boilerplate block starting at lines[start] ('# Offset' line).

    Returns (list_of_var_names, end_index) where end_index is the first line
    index AFTER the entire boilerplate block (offset + fills), or ([], start+1)
    if the pattern was not recognised.
    """
    n = len(lines)
    i = start + 1  # skip the '# Offset' comment itself

    # --- skip blank lines after '# Offset' ---
    while i < n and not lines[i].strip():
        i += 1

    # The next non-blank line must be 'if offset != 0:' or 'if offset:'
    if i >= n:
        return [], start + 1
    l = lines[i].strip()
    if not (l.startswith("if offset") and l.endswith(":")):
        return [], start + 1
    i += 1

    # --- collect VAR = VAR.shift(offset) lines ---
    var_names: list[str] = []
    while i < n:
        line = lines[i]
        ls = line.strip()
        if not ls:
            i += 1
            continue
        m = SHIFT_RE.match(line)
        if m and len(m.group(1)) > base_indent:
            var_names.append(m.group(2))
            i += 1
        else:
            break

    if not var_names:
        return [], start + 1

    # --- consume the fill section ---
    # Lines that belong to the fill boilerplate (all at base_indent or deeper):
    #   - blank lines
    #   - '# Handle fills'
    #   - 'if "fillna" in kwargs:' and its body
    #   - 'if "fill_method" in kwargs:' and its body
    # The section ends at the first non-blank line at base_indent that is not
    # one of those keywords.
    while i < n:
        line = lines[i]
        ls = line.strip()

        if not ls:  # blank line — may be inside boilerplate
            i += 1
            continue

        li = len(line) - len(line.lstrip())

        if li < base_indent:
            # Dedented past the function body — end of boilerplate
            break
        elif li == base_indent:
            if (
                ls == "# Handle fills"
                or ls.startswith('if "fillna"')
                or ls.startswith('if "fill_method"')
            ):
                i += 1  # still boilerplate
            else:
                break  # first non-boilerplate line at base indent
        else:
            i += 1  # deeper indent — inside a boilerplate block

    return var_names, i


def ensure_import(text: str) -> str:
    """Add 'from pandas_ta_classic.utils import apply_offset' if not present.

    Prefers to append 'apply_offset' to an existing
    'from pandas_ta_classic.utils import ...' line (sorted), otherwise inserts
    a new import after the last import statement in the file.
    """
    if IMPORT_STMT in text:
        return text

    lines = text.splitlines(keepends=True)
    PREFIX = "from pandas_ta_classic.utils import "

    # Try to find an existing utils import line to extend
    utils_idx = -1
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(PREFIX):
            utils_idx = idx

    if utils_idx >= 0:
        old_line = lines[utils_idx]
        old_stripped = old_line.rstrip()
        # Extract existing names
        existing = old_stripped[old_stripped.index(PREFIX) + len(PREFIX) :]
        names = [n.strip() for n in existing.split(",") if n.strip()]
        if "apply_offset" not in names:
            names = sorted(set(names + ["apply_offset"]))
            lines[utils_idx] = PREFIX + ", ".join(names) + "\n"
    else:
        # Insert after the last import line
        last_import = -1
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import = idx
        insert_at = last_import + 1 if last_import >= 0 else 0
        lines.insert(insert_at, IMPORT_STMT + "\n")

    return "".join(lines)


def migrate_file(path: Path, dry_run: bool = False) -> int:
    """Migrate one indicator file. Returns the number of blocks replaced."""
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    new_lines: list[str] = []
    i = 0
    replaced = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()
        lstripped = stripped.lstrip()

        if lstripped == "# Offset":
            base_indent = len(stripped) - len(lstripped)
            var_names, end_i = parse_boilerplate(lines, i, base_indent)

            if var_names:
                ind = " " * base_indent
                new_lines.append(f"{ind}# Offset\n")
                for v in var_names:
                    new_lines.append(
                        f"{ind}{v} = apply_offset({v}, offset, **kwargs)\n"
                    )
                new_lines.append("\n")  # blank line after offset block
                i = end_i
                replaced += 1
                continue

        new_lines.append(line)
        i += 1

    if replaced > 0:
        new_text = ensure_import("".join(new_lines))
        if not dry_run:
            path.write_text(new_text, encoding="utf-8")
        verb = "[DRY] Would modify" if dry_run else "Modified"
        print(f"{verb}: {path.relative_to(BASE.parent)} ({replaced} block(s))")

    return replaced


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    total_blocks = 0
    total_files = 0

    for category in CATEGORIES:
        cat_dir = BASE / category
        if not cat_dir.exists():
            continue
        for py_file in sorted(cat_dir.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            n = migrate_file(py_file, dry_run=dry_run)
            if n > 0:
                total_blocks += n
                total_files += 1

    prefix = "[DRY RUN] " if dry_run else ""
    print(
        f"\n{prefix}Done: {total_files} file(s) modified, "
        f"{total_blocks} block(s) replaced"
    )


if __name__ == "__main__":
    main()
