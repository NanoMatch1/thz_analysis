"""Unit tests for the filename-similarity reference tiebreaker in GroupingService.

Dependency-free (no pytest required) so it runs under the project venv:
    .venv/Scripts/python.exe tests/test_grouping_tiebreaker.py
It is also pytest-collectable if pytest is available.
"""

import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_core.services.grouping import GroupingService, select_closest_filename


@contextlib.contextmanager
def assert_raises(exception_type, match=None):
    """Minimal stand-in for pytest.raises so the suite needs no pytest."""
    try:
        yield
    except exception_type as error:
        if match is not None and match not in str(error):
            raise AssertionError(
                f"Expected substring '{match}' in error message, got: {error}"
            )
        return
    raise AssertionError(f"Expected {exception_type.__name__} to be raised, none was.")


# --- select_closest_filename (pure function) --------------------------------

def test_select_closest_picks_most_similar_filename():
    """post-adjustment is closer to post-alignment than to pre-alignment."""
    result = select_closest_filename(
        "sample_300k_up_post-adjustment.acc",
        ["air_300k_up_post-alignment.acc", "air_300k_up_pre-alignment.acc"],
        margin=0.05,
    )
    assert result["winner"] == "air_300k_up_post-alignment.acc"
    assert result["decisive"] is True
    assert result["winner_score"] > result["runner_up_score"]


def test_select_closest_reports_genuine_tie_as_not_decisive():
    """A sample with no distinguishing trailing clue cannot be resolved."""
    result = select_closest_filename(
        "sample_300k_up.acc",
        ["air_300k_up_a.acc", "air_300k_up_b.acc"],
        margin=0.05,
    )
    assert result["decisive"] is False
    assert result["gap"] < 0.05


def test_select_closest_respects_margin():
    """The same decisive case becomes non-decisive under a stricter margin."""
    candidates = ["air_300k_up_post-alignment.acc", "air_300k_up_pre-alignment.acc"]
    target = "sample_300k_up_post-adjustment.acc"
    assert select_closest_filename(target, candidates, margin=0.05)["decisive"] is True
    # Real gap here is ~0.096, below the strict 0.10 default.
    assert select_closest_filename(target, candidates, margin=0.10)["decisive"] is False


# --- GroupingService integration -------------------------------------------

def _make_service(filenames, **group_kwargs):
    service = GroupingService(keywords=["type", "temp", "series"])
    service.update(filenames)
    service.simple_grouping(keywords=["type", "temp", "series"], **group_kwargs)
    return service


def test_closest_mode_resolves_ambiguous_air_reference():
    files = [
        "sample_300k_up_post-adjustment.acc",
        "air_300k_up_post-alignment.acc",
        "air_300k_up_pre-alignment.acc",
    ]
    service = _make_service(files, reference_tiebreaker="closest", tiebreak_margin=0.05)
    sample = service.file_items["sample_300k_up_post-adjustment.acc"]
    assert sample.air_reference == "air_300k_up_post-alignment.acc"
    # Score diagnostics are stored on the item (user chose print + store).
    assert sample.__dict__["air_reference_match"]["decisive"] is True


def test_strict_mode_raises_on_ambiguous_reference():
    files = [
        "sample_300k_up_post-adjustment.acc",
        "air_300k_up_post-alignment.acc",
        "air_300k_up_pre-alignment.acc",
    ]
    with assert_raises(ValueError, match="Ambiguous reference pairing"):
        _make_service(files, reference_tiebreaker="strict")


def test_default_margin_raises_on_borderline_example():
    """The chosen 0.10 default is below this example's 0.096 gap -> raises."""
    files = [
        "sample_300k_up_post-adjustment.acc",
        "air_300k_up_post-alignment.acc",
        "air_300k_up_pre-alignment.acc",
    ]
    with assert_raises(ValueError, match="could not decide"):
        _make_service(files, reference_tiebreaker="closest", tiebreak_margin=0.10)


def test_closest_mode_still_raises_on_genuine_tie():
    files = [
        "sample_300k_up.acc",
        "air_300k_up_a.acc",
        "air_300k_up_b.acc",
    ]
    with assert_raises(ValueError, match="could not decide"):
        _make_service(files, reference_tiebreaker="closest", tiebreak_margin=0.05)


def test_single_match_has_no_tiebreaker_diagnostics():
    files = [
        "sample_300k_up.acc",
        "air_300k_up.acc",
    ]
    service = _make_service(files, reference_tiebreaker="closest")
    sample = service.file_items["sample_300k_up.acc"]
    assert sample.air_reference == "air_300k_up.acc"
    assert "air_reference_match" not in sample.__dict__


def test_keyvalue_pairing_unaffected_by_tiebreaker():
    """key=value hard matches resolve without invoking the tiebreaker at all."""
    files = [
        "sample_cu_optp_series=1.acc",
        "ref-substrate_cu_tds_series=1.acc",
    ]
    service = GroupingService(keywords=["type"])
    service.update(files)
    service.simple_grouping(keywords=["type"])
    sample = service.file_items["sample_cu_optp_series=1.acc"]
    assert sample.substrate_reference == "ref-substrate_cu_tds_series=1.acc"
    assert "substrate_reference_match" not in sample.__dict__


def _run_all():
    test_functions = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test_function in test_functions:
        try:
            test_function()
            print(f"PASS  {test_function.__name__}")
        except Exception as error:  # noqa: BLE001 - test harness reports all failures
            failures += 1
            print(f"FAIL  {test_function.__name__}: {error}")
    print(f"\n{len(test_functions) - failures}/{len(test_functions)} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
