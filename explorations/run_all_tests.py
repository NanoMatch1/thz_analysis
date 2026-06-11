"""Run every test suite in the project and summarise pass counts.

- dataset_core tests: each tests/*.py is a standalone script (exit code 0 = all pass).
- thz_core tests: pytest suite in the nested thz_core repo.

Run:
    .venv/Scripts/python.exe explorations/run_all_tests.py
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")
THZ_CORE_REPO = os.path.join(REPO_ROOT, "thz_core")


def main() -> int:
    failures = []

    print("=== dataset_core test scripts ===")
    for name in sorted(os.listdir(TESTS_DIR)):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        result = subprocess.run(
            [sys.executable, os.path.join(TESTS_DIR, name)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        tail = [line for line in result.stdout.strip().splitlines() if "passed" in line]
        summary = tail[-1] if tail else f"exit {result.returncode}"
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"{status}  {name}: {summary}")
        if result.returncode != 0:
            failures.append(name)
            print(result.stdout[-3000:])
            print(result.stderr[-3000:])

    print("\n=== thz_core pytest suite ===")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        capture_output=True, text=True, cwd=THZ_CORE_REPO,
    )
    print(result.stdout.strip().splitlines()[-1])
    if result.returncode != 0:
        failures.append("thz_core pytest")
        print(result.stdout[-5000:])

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nAll suites passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
