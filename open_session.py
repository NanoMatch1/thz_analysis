"""Open a saved analysis from the catalogue and look at it — the everyday viewer.

A saved analysis is a ``.thzbundle`` (recipe.json + snapshot.pkl + report.md). You never need
to remember *where* a bundle lives: the catalogue indexes every bundle under the data root
(``THZ_CATALOG_ROOT`` / ``~/.thz/catalog.toml`` / default ``~/data``). Pick one, and it opens
in the interactive ResultsViewer.

Command line:
    python open_session.py                     # interactive picker: browse/filter/pick, then view
    python open_session.py silicon             # match a bundle by id-prefix or series-name
    python open_session.py <id> --replay       # recompute from raw data instead of loading snapshot
    python open_session.py <id> --fit          # fit (GUI) then re-save the bundle
    python open_session.py <id> --export       # write registry-driven CSVs (+ README, fit summary)
    python open_session.py <id> --extract      # list the quantities available to extract/plot
    python open_session.py <id> --no-viewer    # skip launching the results viewer
    python open_session.py <path/to/run.thzbundle>   # explicit bundle path also works

The two companions, for when the viewer isn't enough:
    extract_session_data.py   pull the arrays out as {quantity: {filename: {...}}} dicts
    display_cookbook.py       worked examples for building your own figures

In a REPL / debugger:
    from open_session import open_from_catalogue
    dataset = open_from_catalogue("49ad3cc4")     # or no argument for the interactive picker
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import session_bundle, display
from dataset_core.adapters.catalog import Catalog, parse_filter_tokens


# ── catalogue selection (browse / filter / pick) ───────────────────────────────


def describe_record(record) -> str:
    """One-line human summary of a catalogue record (used by the picker list)."""
    fit_note = "  fit" if record.fit_models else ""
    flag_note = "  [flags]" if record.flags_raised else ""
    created = (record.created or "")[:10]
    return (
        f"{record.bundle_id[:8]}  {created:10}  {record.measurement_type:12} "
        f"pol={record.polarization or '-':2}  {','.join(record.quantities):17}  "
        f"{record.series_name}{fit_note}{flag_note}"
    )


def print_catalogue(records) -> None:
    """Print a numbered catalogue listing for interactive selection."""
    if not records:
        print("  (no matching analyses)")
        return
    for position, record in enumerate(records, 1):
        print(f"  [{position:2}] {describe_record(record)}")


def select_record_from_catalogue(catalog: Catalog, records=None):
    """Interactive picker: browse, filter, and pick one analysis. Returns a record or None.

    Commands at the ``catalogue>`` prompt:
        <number>          open that row
        <id or name>      open the row matching a bundle-id prefix or series name
        find key=value …  filter (keys: type, pol, sample, since, until, notes, fits, flags, text)
        all               clear the filter (show everything again)
        q                 quit without selecting
    """
    current = list(records) if records is not None else catalog.list_all()
    print(f"Catalogue root: {catalog.root}")

    while True:
        print()
        print_catalogue(current)
        print("\nCommands:  <number>=open  |  find key=value ...  |  all  |  q=quit")
        try:
            command = input("catalogue> ").strip()
        except EOFError:
            return None  # non-interactive stdin: don't hang
        if not command:
            continue

        lowered = command.lower()
        if lowered in ("q", "quit", "exit"):
            return None
        if lowered in ("all", "reset"):
            current = catalog.list_all()
            continue
        if lowered.startswith("find"):
            try:
                filters = parse_filter_tokens(command.split()[1:])
            except ValueError as error:
                print(f"  {error}")
                continue
            current = catalog.find(**filters)
            continue
        if command.isdigit():
            index = int(command) - 1
            if 0 <= index < len(current):
                return current[index]
            print("  Number out of range.")
            continue
        current_ids = {record.bundle_id for record in current}
        matches = [record for record in catalog.match(command) if record.bundle_id in current_ids]
        if len(matches) == 1:
            return matches[0]
        print("  Unrecognised or ambiguous. Enter a number, 'find key=value', 'all', or 'q'.")


# ── resolving what to open (catalogue identifier, explicit path, or picker) ─────


def _looks_like_path(argument: str) -> bool:
    return os.path.isdir(argument) or os.sep in argument or "/" in argument


def _bundle_dir_from_path(argument: str) -> str:
    """Resolve an explicit path to a bundle directory (a bundle, or a data dir holding one)."""
    if os.path.exists(os.path.join(argument, "recipe.json")):
        return argument  # argument is itself the .thzbundle directory
    basename = os.path.basename(os.path.normpath(argument))
    candidate = os.path.join(argument, f"{basename}.thzbundle")
    if os.path.exists(os.path.join(candidate, "recipe.json")):
        return candidate  # argument is a data dir containing <basename>.thzbundle
    raise SystemExit(f"No bundle found at '{argument}'.")


def resolve_bundle_dir(identifier: str | None = None, catalog: Catalog | None = None) -> str | None:
    """Turn a path / identifier / None into a bundle directory, asking the user if ambiguous.

    The catalogue is rebuilt first so bundles saved since the last scan are always pickable.
    Returns None if the user quits the picker.
    """
    if identifier is not None and _looks_like_path(identifier):
        return _bundle_dir_from_path(identifier)

    catalog = catalog or Catalog()
    catalog.rebuild()
    if identifier is None:
        record = select_record_from_catalogue(catalog)
    else:
        matches = catalog.match(identifier)
        if len(matches) == 1:
            record = matches[0]
        elif matches:
            print(f"'{identifier}' matches {len(matches)} analyses — pick one:")
            record = select_record_from_catalogue(catalog, records=matches)
        else:
            print(f"No catalogue match for '{identifier}'. Showing everything:")
            record = select_record_from_catalogue(catalog)
    return catalog.resolve_bundle_dir(record) if record is not None else None


def open_from_catalogue(identifier: str | None = None, catalog: Catalog | None = None):
    """Load a bundle's saved results (no recompute). Picker if no identifier. DataSet or None."""
    bundle_dir = resolve_bundle_dir(identifier, catalog)
    if bundle_dir is None:
        return None
    return session_bundle.load_session(bundle_dir)


# ── command line ───────────────────────────────────────────────────────────────


def print_available_quantities(dataset) -> None:
    """List what can be pulled out with extract_session_data.py or plotted with display."""
    print("Registered quantities with data:", ", ".join(display.available_quantities(dataset)))
    print("Files:", ", ".join(dataset.data.data_dict))
    print("Extract them as dicts with extract_session_data.extract_series(dataset, [...]).")


def main(arguments: list[str] | None = None):
    arguments = sys.argv[1:] if arguments is None else arguments
    flags = {argument for argument in arguments if argument.startswith("--")}
    positionals = [argument for argument in arguments if not argument.startswith("--")]

    bundle_dir = resolve_bundle_dir(positionals[0] if positionals else None)
    if bundle_dir is None:
        print("Nothing selected.")
        return None

    if "--replay" in flags:
        # Recompute headlessly from the raw data using the recorded recipe.
        print(f"Replaying {bundle_dir} from raw data ...")
        dataset = session_bundle.replay_session(bundle_dir)
    else:
        # Instant reload of the saved results — no recompute.
        dataset = session_bundle.load_session(bundle_dir)

    if "--fit" in flags:
        # Fit later without reprocessing: open the fit GUI, then re-save so fits travel with it.
        from dataset_core.adapters import fitting
        fitting.fit_interactive(dataset)
        session_bundle.save_session(dataset, bundle_dir, notes="fits added via open_session --fit")

    if "--export" in flags:
        thz.export_quantities(dataset)

    if "--extract" in flags:
        print_available_quantities(dataset)

    if "--no-viewer" not in flags:
        thz.launch_results_viewer(dataset)
        plt.show()

    return dataset


if __name__ == "__main__":
    main()
