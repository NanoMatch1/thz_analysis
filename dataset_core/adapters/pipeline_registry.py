"""Pipeline stage registry + recipe recording + headless replay.

Purpose
-------
Turn an interactive pipeline run into a *recipe* — the ordered list of pipeline calls and
their arguments — so a run can be reopened, reproduced, or modified later without hand-editing
a config dict per dataset.

How it works
------------
`activate_recording()` wraps the core `thz_adapter` stage functions AND the two `DataSet`
setup methods (`load_all_data`, `group_files`) so that **each call self-records** a step
``{"stage", "kwargs", "ts"}`` onto ``dataset.recipe`` (a list). Call sites are unchanged —
``thz.transfer_function(ds, ...)`` still works and now records — because we replace the
attribute on the module/class, and Python looks it up at call time.

The set of recorded stages lives in ONE place (``CORE_STAGE_NAMES`` below) — the single source
of truth for "what is reproducible". Adding a new reproducible stage = add its name here.

`replay_recipe(recipe)` rebuilds a fresh ``DataSet`` from the raw data directory and dispatches
the recorded steps in order — reproducing the analysis headlessly with the *current* code (which
sidesteps pickle-version fragility) and lets you override the config or individual step kwargs.

Design notes
------------
- Recording is re-entrancy guarded: only the TOP-LEVEL call the script makes is recorded, not
  internal sub-calls a stage may make to other wrapped stages.
- ``config`` and the dataset itself are excluded from recorded kwargs — config is stored ONCE at
  the recipe level (single source of truth), not duplicated into every step.
- Wrapping is idempotent; ``activate_recording()`` is safe to call more than once.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Callable

from dataset_core.services.provenance import _jsonable

# ── Single source of truth: which stages are recorded/replayable ───────────────
# thz_adapter stage functions with signature ``(dataset, ...)``.
CORE_STAGE_NAMES: tuple[str, ...] = (
    # shared pre-processing
    "subtract_baseline",
    "taper_and_pad_traces",
    "taper_and_pad_traces_universal",
    "zero_pad",
    "fft_spectrum",
    # transmission
    "window_time_fixed_width",
    "transfer_function",
    "remove_phase_offset",
    "invert_nk",
    # reflection (single + window)
    "build_full_trace_reflection",
    "define_reflection_regions",
    "window_pulses_fixed_width",
    "window_single_pulse_fixed_width",
    "detrend_transfer_phase",
    "phase_correct_kk",
    "selfref_quality",
    "invert_nk_reflection",
    "deembed_air_gap_reflection",
    # shared post-processing
    "compute_instrument_resolution",
    "compute_transfer_uncertainty",
    "apply_instrument_resolution",
    "derive_eps_sigma",
)

# DataSet methods (signature ``(self, ...)`` — self IS the dataset) that set up the run.
DATASET_STAGE_NAMES: tuple[str, ...] = (
    "load_all_data",
    "group_files",
)

# Populated by activate_recording(): stage name -> "thz" | "dataset" (how replay dispatches it).
STAGE_KIND: dict[str, str] = {}

_EXCLUDED_ARG_NAMES = frozenset({"dataset", "self", "config"})


# ── Recording ──────────────────────────────────────────────────────────────────


def _flatten_bound_kwargs(signature: inspect.Signature, bound_arguments: dict) -> dict:
    """Turn bound arguments into a flat, replay-ready kwargs dict.

    A ``**kwargs`` (VAR_KEYWORD) parameter binds its contents nested under the parameter
    name; spread those to the top level so replay can re-pass them by keyword. The dataset
    argument and ``config`` are dropped (config is stored once at the recipe level).
    ``*args`` (VAR_POSITIONAL) is skipped — the recorded stages take the dataset positionally
    and everything else by keyword.
    """
    flat: dict = {}
    for name, value in bound_arguments.items():
        if name in _EXCLUDED_ARG_NAMES:
            continue
        kind = signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            for inner_name, inner_value in (value or {}).items():
                if inner_name not in _EXCLUDED_ARG_NAMES:
                    flat[inner_name] = _jsonable(inner_value)
        elif kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        else:
            flat[name] = _jsonable(value)
    return flat


def _record_step(dataset, stage_name: str, kwargs: dict) -> None:
    """Append one ``{stage, kwargs, ts}`` record to ``dataset.recipe`` (best-effort)."""
    recipe = getattr(dataset, "recipe", None)
    if recipe is None:
        recipe = []
        try:
            dataset.recipe = recipe
        except Exception:
            return  # not a recordable dataset; skip silently
    recipe.append({
        "stage": stage_name,
        "kwargs": kwargs,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


def _make_recording_wrapper(stage_name: str, func: Callable, dataset_position: int) -> Callable:
    """Wrap ``func`` so it records a recipe step for the TOP-LEVEL call, then delegates."""
    signature = inspect.signature(func)

    def wrapper(*args, **kwargs):
        dataset = args[dataset_position] if len(args) > dataset_position else kwargs.get("dataset")
        # Re-entrancy guard: record only the outermost stage, not nested sub-calls.
        already_in_stage = getattr(dataset, "_recipe_in_stage", False) if dataset is not None else True
        if dataset is not None and not already_in_stage and getattr(dataset, "_recipe_recording", True):
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError:
                bound = None
            dataset._recipe_in_stage = True
            try:
                result = func(*args, **kwargs)
            finally:
                dataset._recipe_in_stage = False
            if bound is not None:
                _record_step(dataset, stage_name, _flatten_bound_kwargs(signature, bound.arguments))
            return result
        return func(*args, **kwargs)

    wrapper.__name__ = getattr(func, "__name__", stage_name)
    wrapper.__qualname__ = getattr(func, "__qualname__", stage_name)
    wrapper.__doc__ = func.__doc__
    wrapper._records_recipe = True
    wrapper._recipe_wrapped_func = func
    return wrapper


def activate_recording(extra_thz_stages: tuple[str, ...] = ()) -> None:
    """Wrap the core thz stages + DataSet setup methods so calls self-record into ``dataset.recipe``.

    Idempotent. Call once near the top of a run script (before the pipeline calls) so the
    subsequent ``thz.*`` / ``dataset.*`` calls are recorded.
    """
    import dataset_core.adapters.thz_adapter as thz
    from dataset_core.dataset import DataSet

    for name in tuple(CORE_STAGE_NAMES) + tuple(extra_thz_stages):
        func = getattr(thz, name, None)
        if func is None or getattr(func, "_records_recipe", False):
            if func is not None:
                STAGE_KIND[name] = "thz"
            continue
        setattr(thz, name, _make_recording_wrapper(name, func, dataset_position=0))
        STAGE_KIND[name] = "thz"

    for name in DATASET_STAGE_NAMES:
        method = getattr(DataSet, name, None)
        if method is None or getattr(method, "_records_recipe", False):
            if method is not None:
                STAGE_KIND[name] = "dataset"
            continue
        setattr(DataSet, name, _make_recording_wrapper(name, method, dataset_position=0))
        STAGE_KIND[name] = "dataset"


# ── Replay ───────────────────────────────────────────────────────────────────────


def replay_recipe(
    recipe: dict,
    *,
    override_config: dict | None = None,
    override_steps: dict | None = None,
    verbose: bool = True,
):
    """Rebuild a ``DataSet`` headlessly by re-executing a recorded recipe.

    Parameters
    ----------
    recipe : dict
        A recipe dict as written by ``session_bundle.save_session`` — must have
        ``source_dir``, ``config`` and ``steps`` (list of ``{stage, kwargs}``).
    override_config : dict, optional
        Shallow-merged over ``recipe['config']`` before the run (e.g. a new thickness),
        so you can "reuse the config that generated this, but change X".
    override_steps : dict, optional
        ``{stage_name: {kwarg: value}}`` — per-stage kwarg overrides applied at dispatch.
    verbose : bool
        Print a ``[replay]`` line per dispatched step.

    Returns
    -------
    DataSet
        The reproduced dataset (fully processed to the recorded end state).
    """
    import dataset_core.adapters.thz_adapter as thz
    from dataset_core.dataset import DataSet

    activate_recording()  # ensures STAGE_KIND is populated

    merged_config = dict(recipe.get("config") or {})
    if override_config:
        merged_config.update(override_config)
    override_steps = override_steps or {}

    dataset = DataSet(recipe["source_dir"], config=merged_config)
    # Don't re-record while replaying — the recipe already exists.
    dataset._recipe_recording = False

    for step in recipe.get("steps", []):
        stage = step["stage"]
        kwargs = dict(step.get("kwargs") or {})
        kwargs.update(override_steps.get(stage, {}))
        kind = STAGE_KIND.get(stage)
        if kind is None:
            if verbose:
                print(f"[replay] SKIP unknown stage '{stage}' (not registered).")
            continue
        if verbose:
            print(f"[replay] {stage}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})")
        if kind == "dataset":
            getattr(dataset, stage)(**kwargs)
        else:
            getattr(thz, stage)(dataset, **kwargs)

    dataset._recipe_recording = True
    return dataset
