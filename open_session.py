"""Reopen a saved analysis session bundle (.thzbundle) — two ways.

A bundle is written by any run_me_*.py with config['general']['save_session'] set (or by calling
``session_bundle.save_session(dataset, path)`` directly). It contains:

    recipe.json    editable pipeline spec (source dir + config + ordered steps + git SHA)
    snapshot.pkl   the processed results (instant display, no recompute)
    report.md      human-readable processing report

Usage:
    python open_session.py <path/to/run.thzbundle>            # load snapshot + launch viewer
    python open_session.py <path/to/run.thzbundle> --replay   # recompute from raw data
"""

from __future__ import annotations

import sys

from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import session_bundle, pipeline_registry


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    bundle_dir = sys.argv[1]
    replay = "--replay" in sys.argv[2:]

    if replay:
        # Recompute headlessly from the raw data using the recorded recipe. Edit recipe.json
        # first (or pass override_config to replay_recipe) to re-run with tweaks.
        recipe = session_bundle.read_recipe(bundle_dir)
        print(f"Replaying {len(recipe.get('steps', []))} steps from {bundle_dir} ...")
        dataset = pipeline_registry.replay_recipe(recipe)
    else:
        # Instant reload of the saved results — no recompute.
        dataset = session_bundle.load_session(bundle_dir)

    thz.result_viewer(dataset)


if __name__ == "__main__":
    main()
