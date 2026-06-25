"""Regression test: load_all_data must skip non-spectrum files (e.g. _state.pkl).

save_state / save_database write pickle files INTO the data directory. Before the fix,
load_all_data force-parsed them through GenericTextLoader, producing binary garbage that
crashed downstream (binary bytes parsed as a timestamp). The loader now returns None for
non-data extensions so those files are skipped.

Run with:
    .venv/Scripts/python.exe tests/test_loader_skips_state_files.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.dataset import DataSet


def _write_acc(path, name, n=64):
    # minimal valid .acc: title/type header then "time_ps amplitude" rows
    lines = [f"%title {name} acc 1", "%type 0"]
    lines += [f"{150.0 + 0.05 * i:.6f} {1.0e-3 * (i % 7):.10e}" for i in range(n)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def test_load_any_returns_none_for_non_data_extensions():
    dataset = DataSet.__new__(DataSet)  # bare instance; load_any only needs the class attr
    for ext in (".pkl", ".pickle", ".png", ".npz", ".json", ".log", ".zip"):
        assert dataset.load_any(f"anything{ext}") is None, ext


def test_acc_extension_is_not_skipped():
    dataset = DataSet.__new__(DataSet)
    assert ".acc" not in DataSet._NON_DATA_EXTENSIONS
    assert ".dat" not in DataSet._NON_DATA_EXTENSIONS


def test_load_all_data_skips_pickle_in_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_acc(os.path.join(tmpdir, "reference_x.acc"), "reference_x")
        _write_acc(os.path.join(tmpdir, "sample_y.acc"), "sample_y")
        # a binary pickle masquerading in the data dir (what save_state writes)
        with open(os.path.join(tmpdir, "series_state.pkl"), "wb") as fh:
            fh.write(b"\x80\x04\x95garbage\xff\xfe\x00binary")

        dataset = DataSet(tmpdir)
        dataset.load_all_data(case_insensitive=True)  # must NOT raise
        loaded = sorted(dataset.data.data_dict.keys())
        assert loaded == ["reference_x.acc", "sample_y.acc"], loaded


_TESTS = [
    test_load_any_returns_none_for_non_data_extensions,
    test_acc_extension_is_not_skipped,
    test_load_all_data_skips_pickle_in_data_dir,
]


def main() -> int:
    passed = failed = 0
    for test in _TESTS:
        try:
            test()
            passed += 1
            print(f"PASS  {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed ({len(_TESTS)} total)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
