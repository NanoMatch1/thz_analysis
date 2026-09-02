"""The thz-core resolution guard.

`import thz_core.thz_core` needs a top-level directory named `thz_core` whose inner
thz_core/ is the package, so each machine bridges the hyphenated sibling checkout
(`thz-core`) with a link at this repo's root. When that link is instead a stale nested
COPY, everything still imports — from the wrong tree — and the failure surfaces as
whatever symbol happens to be missing. That has cost real work twice.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from dataset_core.adapters import thz_adapter as thz


def test_a_current_thz_core_passes():
    thz._verify_thz_core_resolution()      # must not raise in a healthy checkout


def test_the_installed_thz_core_meets_the_minimum():
    import thz_core.thz_core as core
    found = tuple(int(part) for part in core.__version__.split(".")[:3])
    assert found >= thz._MINIMUM_THZ_CORE_VERSION


def test_a_stale_copy_is_named_and_explained(monkeypatch):
    """The whole point: diagnose the cause, not the symptom."""
    stale = types.ModuleType("thz_core.thz_core")
    stale.__file__ = r"c:\Users\Samuel\matchbook\thz_analysis\thz_core\thz_core\__init__.py"
    stale.__version__ = "0.2.0"
    monkeypatch.setattr(thz, "core", stale)

    with pytest.raises(ImportError) as caught:
        thz._verify_thz_core_resolution()

    message = str(caught.value)
    assert "0.2.0" in message                       # what was found
    assert "stale nested COPY" in message           # the cause, named
    assert "mklink /J thz_core" in message          # the Windows fix
    assert "ln -s ../thz-core" in message           # the POSIX fix
    assert "git -C" in message and "status" in message   # check before deleting


def test_an_unparseable_version_is_treated_as_too_old(monkeypatch):
    odd = types.ModuleType("thz_core.thz_core")
    odd.__file__ = "/somewhere/thz_core/__init__.py"
    odd.__version__ = "not-a-version"
    monkeypatch.setattr(thz, "core", odd)
    with pytest.raises(ImportError):
        thz._verify_thz_core_resolution()


def test_a_missing_version_attribute_is_treated_as_too_old(monkeypatch):
    bare = types.ModuleType("thz_core.thz_core")
    bare.__file__ = "/somewhere/thz_core/__init__.py"
    monkeypatch.setattr(thz, "core", bare)
    with pytest.raises(ImportError):
        thz._verify_thz_core_resolution()
