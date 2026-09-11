#!/usr/bin/env python3
"""Minimal test runner used when pytest is not installed.

    python3 tests/run_tests.py            # run everything
    python3 tests/run_tests.py test_sf    # one module

Supports the subset of pytest used here: ``@pytest.mark.parametrize``,
``pytest.skip`` and the ``run_file`` fixture from ``conftest.py``.
"""
import importlib
import inspect
import os
import sys
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

try:
    import pytest  # noqa: F401
    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False
    pytest = types.ModuleType("pytest")

    class _Skip(Exception):
        pass

    def _skip(msg=""):
        raise _Skip(msg)

    class _Mark:
        @staticmethod
        def parametrize(names, values):
            def deco(fn):
                fn._params = (names, values)
                return fn
            return deco

    def _fixture(*a, **k):
        def deco(fn):
            fn._fixture = True
            return fn
        return deco if not (a and callable(a[0])) else deco(a[0])

    pytest.skip = _skip
    pytest.mark = _Mark()
    pytest.fixture = _fixture
    pytest.Skip = _Skip
    sys.modules["pytest"] = pytest

if HAVE_PYTEST:
    sys.exit(os.system(f"{sys.executable} -m pytest -q {HERE} " + " ".join(sys.argv[1:])) >> 8)

import conftest  # noqa: E402

fixtures = {}
for name, fn in inspect.getmembers(conftest, inspect.isfunction):
    if getattr(fn, "_fixture", False):
        try:
            fixtures[name] = fn()
        except pytest.Skip as e:
            fixtures[name] = e

modules = sys.argv[1:] or sorted(f[:-3] for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
npass = nfail = nskip = 0
for modname in modules:
    mod = importlib.import_module(modname)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        params = getattr(fn, "_params", None)
        cases = [{}]
        if params:
            names = [n.strip() for n in params[0].split(",")]
            cases = [dict(zip(names, v if isinstance(v, tuple) else (v,))) for v in params[1]]
        for kw in cases:
            args = dict(kw)
            skipped = False
            for p in inspect.signature(fn).parameters:
                if p in fixtures:
                    if isinstance(fixtures[p], Exception):
                        skipped = True
                    args[p] = fixtures[p]
            label = f"{modname}.{name}" + (f"[{kw}]" if kw else "")
            if skipped:
                nskip += 1
                print(f"SKIP {label}")
                continue
            try:
                fn(**args)
                npass += 1
                print(f"ok   {label}")
            except pytest.Skip as e:
                nskip += 1
                print(f"SKIP {label}: {e}")
            except Exception:
                nfail += 1
                print(f"FAIL {label}")
                traceback.print_exc()
print(f"\n{npass} passed, {nfail} failed, {nskip} skipped")
sys.exit(1 if nfail else 0)
