"""Browse the THz analysis catalogue from the command line.

The catalogue indexes every saved ``.thzbundle`` under the configured root (env
``THZ_CATALOG_ROOT`` / ``~/.thz/catalog.toml`` / documented default), so you can find an
analysis by what it *is* rather than by remembering its path.

Examples
--------
    python catalog_browse.py                          # list everything (oldest first)
    python catalog_browse.py --rebuild                # rescan the root and rebuild the index
    python catalog_browse.py --verify                 # report indexed bundles now missing on disk
    python catalog_browse.py --find type=reflection sample=CNT since=2026-06
    python catalog_browse.py --find fits=yes flags=yes
    python catalog_browse.py --open <bundle_id>       # load the saved results + launch the viewer
    python catalog_browse.py --root D:/THz --rebuild  # point at a different root for this call
"""

from __future__ import annotations

import argparse

from dataset_core.adapters.catalog import Catalog, parse_filter_tokens


def _parse_filters(tokens: list[str]) -> dict:
    """Parse ``key=value`` filter tokens, surfacing errors as a CLI exit."""
    try:
        return parse_filter_tokens(tokens)
    except ValueError as error:
        raise SystemExit(str(error))


def _print_table(records) -> None:
    if not records:
        print("No matching analyses.")
        return
    header = f"{'id':8}  {'created':10}  {'type':11}  {'pol':3}  {'quantities':16}  {'series / path'}"
    print(header)
    print("-" * len(header))
    for record in records:
        created = (record.created or "")[:10]
        quantities = ",".join(record.quantities)
        flag_marker = "  [flags]" if record.flags_raised else ""
        label = record.series_name or record.relative_path
        print(
            f"{record.bundle_id[:8]:8}  {created:10}  {record.measurement_type:11}  "
            f"{(record.polarization or '-'):3}  {quantities:16}  {label}{flag_marker}"
        )
    print(f"\n{len(records)} analysis/analyses.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse the THz analysis catalogue.")
    parser.add_argument("--root", default=None, help="Override the catalogue root for this call.")
    parser.add_argument("--rebuild", action="store_true", help="Rescan the root and rebuild the index.")
    parser.add_argument("--verify", action="store_true", help="Report indexed bundles missing on disk.")
    parser.add_argument("--open", dest="open_id", default=None, help="Bundle id to load + view.")
    parser.add_argument("--find", nargs="+", default=None, metavar="key=value", help="Query filters.")
    args = parser.parse_args()

    catalog = Catalog(root=args.root)
    print(f"Catalogue root: {catalog.root}")

    if args.rebuild:
        count = catalog.rebuild()
        print(f"Rebuilt catalogue: {count} bundle(s) indexed.")

    if args.verify:
        missing = catalog.verify()
        if missing:
            print(f"{len(missing)} indexed bundle(s) missing on disk:")
            for relative_path in missing:
                print(f"  - {relative_path}")
        else:
            print("All indexed bundles present on disk.")

    if args.open_id is not None:
        dataset = catalog.open(args.open_id)
        from dataset_core.adapters import thz_adapter as thz

        thz.launch_results_viewer(dataset)
        return

    # Default action: list (optionally filtered). Skip if the run was purely rebuild/verify.
    if args.find is not None or not (args.rebuild or args.verify):
        records = catalog.find(**_parse_filters(args.find)) if args.find else catalog.list_all()
        _print_table(records)


if __name__ == "__main__":
    main()
