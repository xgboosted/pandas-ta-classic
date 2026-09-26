"""No tracked file may contain a character that cannot be seen.

Non-ASCII text is fine: the em dashes in this CHANGELOG, the macrons in
"Ichimoku Kinkō Hyō", the © in a source citation and the × in a benchmark table
all say what they mean, and Python 3 source is UTF-8 by default (PEP 3120).

What is not fine is a character that is invisible, or indistinguishable from an
ASCII one, because nothing in a diff, a review or a grep reveals it:

* the no-break spaces (U+00A0 and friends) read as a space but are not one, so
  ``str.split()`` and every editor's "trim whitespace" leave them alone;
* the zero-width family (U+200B..U+200D, U+2060, U+FEFF) occupies no space at
  all — ``candles/ha.py`` carried a docstring line holding two of them, written
  as escapes, which ``help(ta.ha)`` faithfully rendered as nothing;
* U+2011 NON-BREAKING HYPHEN looks exactly like ``-``, so searching for the word
  it sits in fails — ``README.md`` had ``per<U+2011>indicator``;
* the bidi controls (U+202A..U+202E, U+2066..U+2069) reorder how a line is
  displayed without changing what the parser reads, which is the Trojan Source
  attack (CVE-2021-42574).

Both spellings are rejected, because both reach the reader: the character itself
in the file's bytes, and, in Python sources, a unicode or hex escape inside a
non-raw string, which the interpreter turns back into the character.

Writing *about* such an escape is not the same as using one. A doubled backslash
escapes the backslash, so the rest is literal text, which is how this module and
the CHANGELOG name the defect without being it. The escape check therefore counts
the backslashes in front of the code point and only rejects an odd run. The
forbidden set itself is built from code points at run time, so no forbidden
character appears here either.
"""

import re
import subprocess
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Invisible, or visually identical to an ASCII character.
FORBIDDEN = {
    0x00A0,  # no-break space
    0x00AD,  # soft hyphen
    0x2007,  # figure space
    0x2009,  # thin space
    0x2011,  # non-breaking hyphen: looks like "-"
    0x202F,  # narrow no-break space
    0x2060,  # word joiner
    0x3000,  # ideographic space
    0xFEFF,  # zero width no-break space / byte order mark
    *range(0x200B, 0x2010),  # zero width space .. right-to-left mark
    *range(0x202A, 0x202F),  # bidi embeddings and overrides
    *range(0x2066, 0x206A),  # bidi isolates
}

# Files whose bytes are not text, or are text this check cannot say anything
# useful about.
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".xlsx", ".zip", ".whl", ".so", ".pyd"}

# A run of backslashes followed by a code point escape. The run length decides
# whether the last backslash escapes the code point or is itself escaped.
ESCAPE = re.compile(r"(\\+)(u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2})")


def _name(code):
    try:
        return unicodedata.name(chr(code))
    except ValueError:
        return "<unnamed>"


def _label(code):
    return f"U+{code:04X} {_name(code)}"


@pytest.fixture(scope="module")
def tracked():
    """(path, bytes) for every tracked file that holds text."""
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=False)
    if listing.returncode != 0:
        pytest.skip("not a git checkout, so there is no file list to check")
    out = []
    for name in listing.stdout.decode().split("\0"):
        if not name:
            continue
        path = ROOT / name
        if not path.is_file() or path.suffix.lower() in BINARY_SUFFIXES:
            continue
        raw = path.read_bytes()
        if b"\0" in raw[:4096]:
            continue
        out.append((name, raw))
    assert out, "the file list came back empty"
    return out


def test_no_invisible_characters(tracked):
    violations = []
    for name, raw in tracked:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            violations.append(f"{name}: not valid UTF-8 ({exc})")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for column, char in enumerate(line, 1):
                if ord(char) in FORBIDDEN:
                    violations.append(f"{name}:{lineno}:{column}: {_label(ord(char))}")
    assert not violations, "invisible or ASCII-lookalike characters:\n  " + "\n  ".join(violations)


def test_no_invisible_characters_written_as_escapes(tracked):
    violations = []
    for name, raw in tracked:
        if not name.endswith(".py"):
            continue
        for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            for match in ESCAPE.finditer(line):
                if len(match.group(1)) % 2 == 0:
                    continue  # the backslash is escaped, so this is literal text
                code = int(match.group(2)[1:], 16)
                if code in FORBIDDEN:
                    violations.append(f"{name}:{lineno}: {match.group(0)!r} spells {_label(code)}")
    assert not violations, "invisible characters written as escapes:\n  " + "\n  ".join(violations)


def test_no_byte_order_mark(tracked):
    marked = [name for name, raw in tracked if raw.startswith(b"\xef\xbb\xbf")]
    assert not marked, "files starting with a UTF-8 BOM: " + ", ".join(marked)
