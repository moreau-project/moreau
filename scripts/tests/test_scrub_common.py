"""Tests for scripts/scrub_common.py Python comment stripping."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scrub_common import strip_python_comments


def _compiles(src: str) -> bool:
    try:
        compile(src, "<test>", "exec")
    except SyntaxError:
        return False
    return True


def _assert_round_trips(src: str):
    stripped, _ = strip_python_comments(src)
    assert _compiles(stripped), f"stripped source does not compile:\n{stripped}"
    return stripped


class TestStripPythonComments:
    def test_end_of_line_comment(self):
        stripped, removed = strip_python_comments("x = 1  # inline comment\n")
        assert removed == 1
        assert "inline comment" not in stripped
        assert _compiles(stripped)

    def test_comment_only_line(self):
        src = "# leading comment\nx = 1\n"
        stripped, removed = strip_python_comments(src)
        assert removed == 1
        assert "leading comment" not in stripped
        assert _compiles(stripped)

    def test_multiple_comments(self):
        src = textwrap.dedent("""\
            # header
            x = 1  # a
            y = 2  # b
            # trailer
            """)
        stripped, removed = strip_python_comments(src)
        assert removed == 4
        assert "#" not in stripped
        assert _compiles(stripped)

    def test_preserves_shebang(self):
        src = "#!/usr/bin/env python3\nx = 1  # strip me\n"
        stripped, removed = strip_python_comments(src)
        assert removed == 1
        assert stripped.startswith("#!/usr/bin/env python3")
        assert "strip me" not in stripped
        assert _compiles(stripped)

    def test_preserves_encoding_declaration(self):
        src = "# -*- coding: utf-8 -*-\nx = 1  # strip me\n"
        stripped, removed = strip_python_comments(src)
        assert removed == 1
        assert "coding: utf-8" in stripped

    def test_preserves_shebang_and_encoding(self):
        src = "#!/usr/bin/env python3\n# -*- coding: latin-1 -*-\nx = 1  # strip\n"
        stripped, removed = strip_python_comments(src)
        assert removed == 1
        assert stripped.startswith("#!/usr/bin/env python3")
        assert "coding: latin-1" in stripped

    def test_preserves_docstring(self):
        src = textwrap.dedent('''\
            """Module docstring."""
            def f():
                """Function docstring."""
                return 1  # strip
            ''')
        stripped, removed = strip_python_comments(src)
        assert removed == 1
        assert '"""Module docstring."""' in stripped
        assert '"""Function docstring."""' in stripped
        assert _compiles(stripped)

    def test_hash_in_string_not_stripped(self):
        src = 'x = "not # a comment"  # real comment\n'
        stripped, removed = strip_python_comments(src)
        assert removed == 1
        assert '"not # a comment"' in stripped
        assert "real comment" not in stripped
        assert _compiles(stripped)

    def test_hash_in_fstring_not_stripped(self):
        src = 'x = f"hash: #{y}"  # strip\n'
        stripped, removed = strip_python_comments(src)
        assert removed == 1
        assert "hash: #" in stripped
        assert _compiles(stripped)

    def test_comment_inside_indented_block(self):
        src = textwrap.dedent("""\
            def f():
                if True:
                    # deep comment
                    x = 1  # trailing
                    return x
            """)
        stripped, removed = strip_python_comments(src)
        assert removed == 2
        assert "deep comment" not in stripped
        assert "trailing" not in stripped
        assert _compiles(stripped)
        exec_globals: dict = {}
        exec(stripped, exec_globals)
        assert exec_globals["f"]() == 1

    def test_line_continuation_with_comment(self):
        src = "x = (1 +  # mid\n     2)\n"
        stripped = _assert_round_trips(src)
        exec_globals: dict = {}
        exec(stripped, exec_globals)
        assert exec_globals["x"] == 3

    def test_no_comments(self):
        src = 'def f():\n    return "hi"\n'
        stripped, removed = strip_python_comments(src)
        assert removed == 0
        assert _compiles(stripped)

    def test_empty_file(self):
        stripped, removed = strip_python_comments("")
        assert removed == 0

    def test_only_comments(self):
        stripped, removed = strip_python_comments("# just\n# comments\n")
        assert removed == 2
        assert _compiles(stripped)

    def test_semantics_preserved(self):
        src = textwrap.dedent("""\
            # calculate
            def add(a, b):
                # sum them
                return a + b  # result

            result = add(2, 3)  # should be 5
            """)
        stripped = _assert_round_trips(src)
        stripped_globals: dict = {}
        exec(stripped, stripped_globals)
        orig_globals: dict = {}
        exec(src, orig_globals)
        assert stripped_globals["result"] == orig_globals["result"] == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
