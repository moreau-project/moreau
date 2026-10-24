"""Tests for Python comment stripping without changing executable code."""

import ast
import io
import tokenize

import pytest
from scrub_common import strip_python_comments


@pytest.mark.parametrize(
    "source,removed",
    [
        ("x = 1  # inline\n", 1),
        ("# leading\nx = 1\n", 1),
        ("# header\nx = 1  # a\ny = 2  # b\n# trailer\n", 4),
        (
            '"""Module docstring."""\ndef f():\n    """Function docstring."""\n    return 1 # strip\n',
            1,
        ),
        ('x = "not # a comment"  # real comment\n', 1),
        ('x = f"hash: #{y}"  # strip\n', 1),
        (
            "def f():\n    if True:\n        # nested\n        x = 1 # trailing\n        return x\n",
            2,
        ),
        ("x = (1 +  # mid\n     2)\n", 1),
        ('def f():\n    return "hi"\n', 0),
        ("", 0),
        ("# just\n# comments\n", 2),
        (
            "# calculate\ndef add(a, b):\n    # sum\n    return a + b # result\nresult = add(2, 3) # 5\n",
            4,
        ),
    ],
)
def test_strip_comments_preserves_code(source, removed):
    stripped, count = strip_python_comments(source)
    assert count == removed
    assert ast.dump(ast.parse(stripped)) == ast.dump(ast.parse(source))
    assert all(
        tok.type != tokenize.COMMENT
        for tok in tokenize.generate_tokens(io.StringIO(stripped).readline)
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "#!/usr/bin/env python3\n",
        "# -*- coding: utf-8 -*-\n",
        "#!/usr/bin/env python3\n# -*- coding: latin-1 -*-\n",
    ],
)
def test_preserves_python_directives(prefix):
    source = prefix + "x = 1  # strip\n"
    stripped, removed = strip_python_comments(source)
    assert removed == 1
    assert stripped.startswith(prefix)
    assert "# strip" not in stripped
    assert ast.dump(ast.parse(stripped)) == ast.dump(ast.parse(source))
