"""Reopen a saved analysis and extract quantities from it — now via the catalogue.

A saved analysis is a ``.thzbundle`` (recipe.json + snapshot.pkl + report.md). You no longer
need to remember *where* a bundle lives: the catalogue indexes every bundle under the configured
root (``~/.thz/catalog.toml`` / env ``THZ_CATALOG_ROOT``), and this script lets you pick one
interactively, then hands you a fully-populated ``DataSet`` to extract quantities from.

Command line:
    python open_session.py                     # interactive REPL: browse/filter/pick, then view
    python open_session.py silicon             # match a bundle by id-prefix or series-name
    python open_session.py <id> --replay       # recompute from raw data instead of loading snapshot
    python open_session.py <id> --fit          # fit (GUI) then re-save the bundle
    python open_session.py <id> --export       # write registry-driven CSVs
    python open_session.py <id> --extract      # print the n/k/sigma quantities available
    python open_session.py --no-viewer         # skip launching the results viewer
    python open_session.py <path/to/run.thzbundle>   # still works: explicit bundle path

Interactive use (e.g. in a REPL / debugger):
    from open_session import run_me, extract_quantities
    dataset = run_me()                          # REPL-select a bundle -> DataSet
    data = extract_quantities(dataset, ["n", "k", "sigma"])
"""

from __future__ import annotations

import os
import re
import sys
from math import radians

import numpy as np
import matplotlib.pyplot as plt

from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import session_bundle, pipeline_registry, display
from dataset_core.adapters.catalog import Catalog, parse_filter_tokens


# ── quantity extraction (works on any loaded DataSet, however it was loaded) ────


def extract_sigma(dataset):
    """Extract sigma from the dataset and return a dictionary of results."""
    data_dict = {}
    for filename, data_obj in thz._sample_items(dataset):
        freq = data_obj.processing_dict.get('fft_freq')
        sigma = data_obj.processing_dict.get('sigma')
        if freq is None or sigma is None:
            continue
        data_dict[filename] = {
            "freq": freq,
            "sigma": sigma
        }
    return data_dict

def extract_quantities(dataset, quantities):
    """Extract specified quantities from the dataset and return a dictionary of results."""

    data_dict = {}
    for filename, data_obj in thz._sample_items(dataset):
        print(f"Processing file: {filename}")
        data_dict[filename] = {}
        print("Quantities available: ", list(data_obj.processing_dict.keys()))
        for quantity in quantities:
            value = data_obj.processing_dict.get(quantity)
            if value is not None:
                data_dict[filename][quantity] = value
    return data_dict

def generate_channel(dataX, dataY, depth=0.2):
    '''Generate a channel around the dataY curve for visualization tracing.'''
    upper = dataY + dataY * depth
    lower = dataY - dataY * depth
    return upper, lower

def strip_nan(data):
    """Remove NaN values from the data array."""
    return data[~np.isnan(data)]

def _plot_with_snr_mask(ax, freq_thz, dataY, mask, label=None, color=None, linestyle='solid', marker='o', config={}):
    """Plot dataY vs freq_thz, optionally masking out low-SNR points."""
    guideline = config.get('guideline', False)
    normalise = config.get('normalise', False)
    if mask is None:
        mask = np.ones_like(dataY, dtype=bool)  # If no mask is provided, consider all points as valid
    if normalise:
        max_val = np.max(strip_nan(dataY[mask])) # Normalize to the maximum absolute value
        min_val = np.min(strip_nan(dataY[mask]))
        dataY = (dataY - min_val) / (max_val - min_val)
    # if mask is not None:
    ax.scatter(freq_thz[mask], dataY[mask], label=label, s=10, marker=marker, color=color, alpha=0.8) # high-SNR points
        # ax.scatter(freq_thz[~mask], dataY[~mask], color=color, marker='.', alpha=0.1, s=1, label=None)  # low-SNR points, faint
    # else:
        # ax.scatter(freq_thz, dataY, label=label, color=color, marker=marker, alpha=0.8, s=10)
    if guideline:
        upper, lower = generate_channel(freq_thz[mask], dataY[mask])
        ax.fill_between(freq_thz[mask], lower, upper, color=color, alpha=0.1)  # guideline for reference

def plot_sigma(dataset, title=""):
    fig_sigma, ax = plt.subplots()
    show_snr_mask = True
    cmap = plt.get_cmap('tab10')
    for index, (filename, data_obj) in enumerate(thz._sample_items(dataset)):
        freq = data_obj.processing_dict.get('fft_freq')
        # n = data_obj.processing_dict.get('n')
        # k = data_obj.processing_dict.get('k')
        sigma = data_obj.processing_dict.get('sigma')
        real = sigma.real
        imag = sigma.imag
        if freq is None or real is None or imag is None:
            continue
        mask = data_obj.processing_dict.get('transfer_mask') if show_snr_mask else None

        _plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, real, mask, label="{} (real)".format(filename), color=cmap(index))
        _plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, imag, mask, label="{} (imag)".format(filename), color=cmap(index), marker='x')
    ax.set_xlabel("Frequency (THz)")
    # plt.ylabel("Refractive Index / Extinction Coefficient")
    ax.set_ylabel("Conductivity (S/m)")
    ax.set_title("Derived Conductivity {}".format(title))
    ax.legend()
    return


# ── misalignment series figure (time / FFT / inverted n,k, coloured by angle) ──


def _parse_displacement_mrad(filename: str, displacement_key: str = "displacement_mrad") -> float | None:
    """Signed mrad from a '...159_displace-{minus|plus}-{N}' filename (0 for displace-0). Uses the displacement_key to identify the displacement in the filename.

    Returns None for files outside the 159 misalignment series (e.g. the .dat, the non-159
    zero) so the figure stays in one consistent reference frame.
    """
    if displacement_key not in filename:
        return None
    signed = re.search(rf"{displacement_key}(minus|plus)-(\d+(?:\.\d+)?)", filename)
    if signed:
        magnitude = float(signed.group(2))
        return -magnitude if signed.group(1) == "minus" else magnitude
    if re.search(rf"{displacement_key}-0(?!\d)", filename):
        return 0.0
    return None


def _misalignment_color(signed_mrad: float, max_negative: float, max_positive: float):
    """Black at 0; deepening blue for negative angles, deepening red for positive angles."""
    if signed_mrad == 0:
        return "black"
    if signed_mrad < 0:
        fraction = abs(signed_mrad) / max_negative if max_negative else 1.0
        return plt.cm.Blues(0.2 + 0.6 * fraction)
    fraction = signed_mrad / max_positive if max_positive else 1.0
    return plt.cm.Reds(0.2 + 0.6 * fraction)


def plot_misalignment_series(
    dataset,
    reference_name: str = "reference_gold_0-mrad.acc",
    displacement_key: str = "displacement_mrad",
    mask_band_thz: tuple[float, float] = (0.2, 4.0),
    show: bool = True,
):
    """Three-panel figure of a gold-mirror misalignment series, coloured by displacement angle.

    Panels: (1) time-domain traces, (2) spectral magnitude, (3) inverted n (solid) & k (dashed).
    n,k come from referencing each misaligned trace to the aligned ``reference_name`` gold trace
    (``r = r_reference · Y_sample / Y_ref``) and inverting with the bundle's reflection geometry —
    so the aligned repeats become a "no-misalignment" noise-floor control, and the deviations show
    the spurious material response that pure misalignment injects.
    """
    from thz_core.thz_core.invert import invert_nk_reflection

    geometry = (getattr(dataset, "config", {}) or {}).get("geometry", {}) or {}
    theta_rad = radians(float(geometry.get("theta_external_deg", 0.0)))
    polarization = geometry.get("polarization", "s")
    r_reference = complex(geometry.get("r_reference", -1.0))

    # Collect the series traces, ordered blue -> black -> red by signed angle.
    traces = []
    for filename, data_object in dataset.data.data_dict.items():
        signed_mrad = _parse_displacement_mrad(filename, displacement_key=displacement_key)
        if signed_mrad is not None:
            traces.append((signed_mrad, filename, data_object.processing_dict))
    traces.sort(key=lambda item: item[0])
    if not traces:
        raise SystemExit(f"No '{displacement_key}' misalignment traces found in this dataset.")

    negatives = [m for m, _, _ in traces if m < 0]
    positives = [m for m, _, _ in traces if m > 0]
    max_negative = abs(min(negatives)) if negatives else 1.0
    max_positive = max(positives) if positives else 1.0

    reference_processing = dataset.data.data_dict[reference_name].processing_dict
    reference_frequency_hz = np.asarray(reference_processing["fft_freq"])
    reference_spectrum = np.asarray(reference_processing["fft_spectrum"])
    trusted_band = (
        (reference_frequency_hz >= mask_band_thz[0] * 1e12)
        & (reference_frequency_hz <= mask_band_thz[1] * 1e12)
    )

    figure, (axis_time, axis_fft, axis_n) = plt.subplots(1, 3, figsize=(17, 5.2))
    axis_k = axis_n.twinx()
    legend_handles, legend_labels = [], []

    for signed_mrad, filename, processing in traces:
        color = _misalignment_color(signed_mrad, max_negative, max_positive)
        is_zero = signed_mrad == 0
        line_width = 2.2 if is_zero else 1.4
        z_order = 6 if is_zero else 2

        # (1) time-domain trace
        time_domain = np.asarray(processing["time_domain"])
        axis_time.plot(time_domain[:, 0] * 1e12, time_domain[:, 1],
                       color=color, linewidth=line_width, zorder=z_order)

        # (2) spectral magnitude
        frequency_hz = np.asarray(processing["fft_freq"])
        spectrum = np.asarray(processing["fft_spectrum"])
        axis_fft.semilogy(frequency_hz * thz._HZ_TO_THZ, np.abs(spectrum),
                          color=color, linewidth=line_width, zorder=z_order)

        # legend proxy (one entry per trace)
        repeat_tag = "  (rep)" if re.search(r"repeat|_rep", filename) else ""
        handle, = axis_time.plot([], [], color=color, linewidth=line_width)
        legend_handles.append(handle)
        legend_labels.append(f"{signed_mrad:+.1f} mrad{repeat_tag}")

        # (3) inverted n, k — skip the reference itself (ref-vs-ref is the mirror singularity)
        if filename == reference_name:
            continue
        if frequency_hz.shape != reference_frequency_hz.shape:
            print(f"[misalignment] skipping n,k for {filename}: frequency grid differs from reference.")
            continue
        reflection_coefficient = r_reference * (spectrum / reference_spectrum)
        n, k, _ = invert_nk_reflection(
            reference_frequency_hz, reflection_coefficient, trusted_band,
            theta_rad=theta_rad, polarization=polarization, n_incident=1.0,
        )
        frequency_thz = reference_frequency_hz * thz._HZ_TO_THZ
        axis_n.plot(frequency_thz, n, color=color, linewidth=line_width, zorder=z_order)
        axis_k.plot(frequency_thz, k, color=color, linewidth=line_width, linestyle="--", zorder=z_order)

    axis_time.set_xlabel("Time (ps)")
    axis_time.set_ylabel("Field amplitude (a.u.)")
    axis_time.set_title("1. Time-domain traces")

    axis_fft.set_xlabel("Frequency (THz)")
    axis_fft.set_ylabel("|FFT| (a.u.)")
    axis_fft.set_xlim(0, mask_band_thz[1])
    axis_fft.set_title("2. Spectral magnitude")

    axis_n.set_xlabel("Frequency (THz)")
    axis_n.set_ylabel("n  (solid)")
    axis_k.set_ylabel("k  (dashed)")
    axis_n.set_xlim(*mask_band_thz)
    axis_n.set_title("3. Inverted n & k (vs aligned gold)")

    figure.suptitle(
        f"Gold-mirror misalignment series — {getattr(dataset, 'seriesname', '')} "
        f"({polarization}-pol, {geometry.get('theta_external_deg', '?')}°); "
        "blue = negative mrad, red = positive, black = aligned",
        fontsize=12,
    )
    figure.legend(legend_handles, legend_labels, loc="center right",
                  fontsize=8, title="displacement", framealpha=0.9)
    figure.tight_layout(rect=(0, 0, 0.9, 0.95))
    if show:
        plt.show()
    return figure


# ── catalogue selection (browse / filter / pick) ───────────────────────────────


def describe_record(record) -> str:
    """One-line human summary of a catalogue record (used by the REPL list)."""
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


def _match_by_identifier(records, identifier: str):
    """Resolve a typed identifier to a single record by id-prefix or unique series substring."""
    by_id = [record for record in records if record.bundle_id.startswith(identifier)]
    if len(by_id) == 1:
        return by_id[0]
    by_series = [r for r in records if identifier.lower() in (r.series_name or "").lower()]
    if len(by_series) == 1:
        return by_series[0]
    return None


def select_record_from_catalogue(catalog: Catalog | None = None, records=None):
    """Interactive REPL: browse, filter, and pick one analysis. Returns a record or None.

    Commands at the ``catalogue>`` prompt:
        <number>          open that row
        find key=value …  filter (keys: type, pol, sample, since, until, notes, fits, flags, text)
        all               clear the filter (show everything again)
        q                 quit without selecting
    """
    catalog = catalog or Catalog()
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
        match = _match_by_identifier(current, command)
        if match is not None:
            return match
        print("  Unrecognised. Enter a number, 'find key=value', 'all', or 'q'.")


# ── loading (catalogue identifier, explicit path, or REPL) ─────────────────────


def _looks_like_path(argument: str) -> bool:
    return os.path.isdir(argument) or os.sep in argument or "/" in argument


def _bundle_dir_from_path(argument: str) -> str:
    """Resolve an explicit path to a bundle directory (back-compatible with the old usage)."""
    if os.path.exists(os.path.join(argument, "recipe.json")):
        return argument  # argument is itself the .thzbundle directory
    basename = os.path.basename(os.path.normpath(argument))
    candidate = os.path.join(argument, f"{basename}.thzbundle")
    if os.path.exists(os.path.join(candidate, "recipe.json")):
        return candidate  # argument is a data dir containing <basename>.thzbundle
    raise SystemExit(f"No bundle found at '{argument}'.")


def _find_records(catalog: Catalog, identifier: str):
    """All catalogue records matching an identifier by id-prefix, else by series substring."""
    records = catalog.list_all()
    by_id = [record for record in records if record.bundle_id.startswith(identifier)]
    if by_id:
        return by_id
    return [r for r in records if identifier.lower() in (r.series_name or "").lower()]


def _resolve_bundle_dir(argument: str | None, catalog: Catalog) -> str | None:
    """Turn a CLI argument (path / identifier / None) into a bundle directory to load."""
    if argument is not None and _looks_like_path(argument):
        return _bundle_dir_from_path(argument)

    if argument is not None:
        matches = _find_records(catalog, argument)
        if len(matches) == 1:
            record = matches[0]
        elif matches:
            print(f"'{argument}' matches {len(matches)} analyses — pick one:")
            record = select_record_from_catalogue(catalog, records=matches)
        else:
            print(f"No catalogue match for '{argument}'. Showing everything:")
            record = select_record_from_catalogue(catalog)
    else:
        record = select_record_from_catalogue(catalog)

    return catalog.resolve_bundle_dir(record) if record is not None else None


def open_from_catalogue(identifier: str | None = None, catalog: Catalog | None = None):
    """Load a bundle by catalogue identifier (or REPL if None). Returns a DataSet or None."""
    catalog = catalog or Catalog()
    catalog.rebuild()  # ensure the catalogue is up-to-date
    bundle_dir = _resolve_bundle_dir(identifier, catalog)
    if bundle_dir is None:
        return None
    return session_bundle.load_session(bundle_dir)


def run_me(identifier: str | None = None):
    """Convenience entry point for interactive use: REPL-select (or match) and load a DataSet."""
    return open_from_catalogue(identifier)


# ── command line ───────────────────────────────────────────────────────────────


def main() -> None:
    flags = {argument for argument in sys.argv[1:] if argument.startswith("--")}
    positionals = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    argument = positionals[0] if positionals else None

    catalog = Catalog()
    bundle_dir = _resolve_bundle_dir(argument, catalog)
    if bundle_dir is None:
        print("Nothing selected.")
        return

    if "--replay" in flags:
        # Recompute headlessly from the raw data using the recorded recipe.
        recipe = session_bundle.read_recipe(bundle_dir)
        print(f"Replaying {len(recipe.get('steps', []))} steps from {bundle_dir} ...")
        dataset = pipeline_registry.replay_recipe(recipe)
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
        extract_quantities(dataset, quantities=["n", "k", "sigma"])

    if "--no-viewer" not in flags:
        thz.launch_results_viewer(dataset)
        plt.show()

    return dataset


def show_misalignment_figures(
    dataset,
    displacement_key: str = "_gold_",
    band_thz: tuple[float, float] = (0.0, 4.0),
    nk_band_thz: tuple[float, float] = (0.2, 4.0),
    show: bool = True,
):
    """Group-meeting view: each misalignment-series panel as its OWN figure via ``display``.

    Replaces the single 3-panel ``plot_misalignment_series`` with independent figures built
    through the reusable registry-driven plotter. Add or drop a panel by adding or removing a
    single ``display.plot_quantity`` call — any registered quantity (``transfer_phase``,
    ``sigma_real``, ...) is available the same way.
    """
    # STEP 1 — build the per-item styling ONCE (a pre-plot manipulation that returns a
    # style dict). diverging_series_style parses the signed mrad out of each filename and
    # returns a style_fn that paints blue -> black -> red with "+/-N mrad" labels and an
    # emphasised zero trace. Sharing this one dict across every figure keeps colour/label
    # consistent from the time trace through to n and k.
    style = display.diverging_series_style(dataset, displacement_key=displacement_key)

    # STEP 2 — one call per figure. `style=` controls how each item looks; `plot_config=`
    # controls how the figure looks. Time-domain uses its own helper (line plot, time axis);
    # every frequency-domain quantity goes through plot_quantity (scatter + error bars on the
    # instrument-resolution grid, mask + overlay handled automatically from the registry).
    display.plot_time_domain(
        dataset, style=style,
        plot_config={"title": "Time-domain traces"})

    display.plot_quantity(
        dataset, "fft_mag", style=style,
        plot_config={"title": "Spectral magnitude", "x_range": band_thz})

    display.plot_quantity(
        dataset, "fft_phase", style=style,
        plot_config={"title": "FFT phase (unwrapped)", "x_range": band_thz})

    display.plot_quantity(
        dataset, "n", style=style,
        plot_config={"title": "Refractive index n", "x_range": nk_band_thz})

    display.plot_quantity(
        dataset, "k", style=style,
        plot_config={"title": "Extinction coefficient k", "x_range": nk_band_thz})

    display.plot_quantity(
        dataset, "sigma_real", style=style,
        plot_config={"title": "Conductivity Re σ", "x_range": nk_band_thz})

    display.plot_quantity(
        dataset, "sigma_imag", style=style,
        plot_config={"title": "Conductivity Im σ", "x_range": nk_band_thz})

    if show:
        plt.show()


def display_adapter_cookbook(dataset, displacement_key: str = "_gold_", show: bool = True):
    """Worked examples of the ``display`` adapter — the reference for future data views.

    Every pattern you are likely to want, each as a couple of lines. Read top to bottom;
    copy the block you need. Nothing here is misalignment-specific except the style helper.
    """
    # ── 1. The simplest possible call: one registered quantity -> one figure. ──────────
    # No style, no config: samples only, default colours, scatter + error bars, registry
    # axis label and y-scale. `quantity` is any name in quantity_registry.QUANTITY_REGISTRY.
    display.plot_quantity(dataset, "n")

    # ── 2. List what is actually available to plot for this dataset. ───────────────────
    print("Plottable quantities:", display.available_quantities(dataset))

    # ── 3. Shared computed style + explicit per-file override. ─────────────────────────
    # style resolution priority (low -> high): default < style_fn < per_file(regex) <
    # per_file(exact). So the gradient paints the series, and per_file tweaks individuals.
    # Keys in per_file are regex/substring by default; an EXACT filename wins outright.
    style = display.diverging_series_style(dataset, displacement_key=displacement_key)
    style["per_file"] = {
        "plus-4": {"marker": "s"},                 # every +4 mrad file -> square markers
        # "sample_gold_plus-2-mrad.acc": {"color": "magenta"},  # one exact file recoloured
    }
    display.plot_quantity(dataset, "k", style=style,
                          plot_config={"title": "k — with per-file overrides",
                                       "x_range": (0.2, 4.0)})

    # ── 4. plot_config knobs (how the FIGURE looks; overrides DEFAULT_PLOT_CONFIG). ────
    display.plot_quantity(
        dataset, "transfer_phase", style=style,
        plot_config={
            "title": "Transfer phase — config demo",
            "x_range": (0.2, 3.0),      # axis limits in plot units (THz)
            "show_errorbars": True,     # error bars from the MC *_sigma arrays
            "show_excluded": True,      # also draw the masked-out (low-SNR) points, faint
            "normalise": False,         # True -> min-max each series to [0, 1]
            "guideline": False,         # True -> shade a +/- band around each series
            "legend": True,
        })

    # ── 5. Select WHICH items to plot (files=): regex, list, or a predicate. ───────────
    # None (default) = samples (+ references when the quantity is reference-meaningful,
    # e.g. the FFT spectra). Override to focus a subset:
    display.plot_quantity(dataset, "fft_mag",
                          files="minus",                       # regex/substring on filename
                          plot_config={"title": "fft_mag — negative-mrad files only",
                                       "x_range": (0, 4.0)})
    # files=["a.acc", "b.acc"]              # explicit list
    # files=lambda name: "plus" in name      # arbitrary predicate

    # ── 6. Get the arrays WITHOUT plotting (inspect / manipulate / export yourself). ───
    # Returns {filename: {freq_hz, freq_thz, y, error, mask}} — exactly what would be drawn.
    series = display.get_series(dataset, "n")
    for filename, arrays in series.items():
        finite = arrays["mask"] if arrays["mask"] is not None else slice(None)
        print(f"  {filename}: n over {arrays['freq_thz'][finite].min():.2f}-"
              f"{arrays['freq_thz'][finite].max():.2f} THz")

    # ── 7. Compose into your own multi-panel figure by passing an existing axis. ───────
    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(12, 4.5))
    display.plot_quantity(dataset, "n", style=style, ax=axis_left,
                          plot_config={"title": "n", "x_range": (0.2, 4.0)})
    display.plot_quantity(dataset, "k", style=style, ax=axis_right,
                          plot_config={"title": "k", "x_range": (0.2, 4.0), "legend": False})
    figure.tight_layout()

    if show:
        plt.show()


if __name__ == "__main__":
    # Gold-mirror misalignment series (catalogue id 0f304af1 = 2026-06-30_ref_testing):
    # each output (time / FFT magnitude / FFT phase / n / k) as its own figure, coloured
    # by displacement angle, via the reusable registry-driven display layer.
    dataset = open_from_catalogue(identifier='49ad3cc4')
    show_misalignment_figures(dataset, displacement_key="_gold_", show=True)

    # For the fuller tour of the display adapter (per-file overrides, get_series, config
    # knobs, item selection, composing into your own axes), run the cookbook instead:
    # display_adapter_cookbook(dataset, displacement_key="_gold_", show=True)
