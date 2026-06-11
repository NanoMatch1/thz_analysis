"""Run the thz_core test suite (nested repo copy) with the project venv.

Ensures pytest is available (installs it into the venv if missing), then runs
pytest on thz_core/tests. Pass extra pytest args on the command line, e.g.:

    .venv/Scripts/python.exe explorations/run_thz_core_tests.py tests/test_window.py -q
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THZ_CORE_REPO = os.path.join(REPO_ROOT, "thz_core")


def ensure_pytest() -> None:
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest not found in venv — installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pytest"]
        )


def main() -> int:
    ensure_pytest()
    extra_args = sys.argv[1:] or ["tests", "-q"]
    os.chdir(THZ_CORE_REPO)
    return subprocess.call([sys.executable, "-m", "pytest", *extra_args])


if __name__ == "__main__":
    sys.exit(main())
