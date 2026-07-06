"""Interactive air-gap de-embedding explorer (sliders for gap POSITION and WIDTH).

Purpose (Samuel, 2026-06-23): manually hunt for the contact-gap parameters that make the
de-embedded CNT optical constants look physical. Two sliders:

  * gap POSITION  d_mean  (um) -- the mean gap thickness. Sets the round-trip phase that is
                                  stripped off:  r_back = x * exp(+i 2 beta(d_mean)).
  * gap WIDTH     sigma_d (um) -- surface-roughness spread of the gap distribution. Undoes
                                  the Debye-Waller magnitude suppression the roughness causes:
                                  r_back /= W,  W = exp(-2 ((omega/c) cos(theta_gap) sigma_d)^2).

For each (position, width) the script re-de-embeds the MEASURED reflection, re-inverts r_back
with AIR incidence at theta_gap = 45 deg, and recomputes the complex conductivity. It live-plots
four panels -- n, k, Re(sigma), Im(sigma) -- one line per sample, so you can eyeball the combo
that fits the expected behaviour. The naive (no de-embed) curve is drawn faintly for reference.

Model notes / honesty:
  * The POSITION correction (linear-phase strip) is the EXACT single-gap inverse (|x| = |r_back|
    when sigma_d = 0; verified to ~1e-15 in explore_air_gap_deembedding.py).
  * The WIDTH correction is the SINGLE-BOUNCE Debye-Waller approximation of the roughness
    magnitude loss -- a first-order un-suppression, NOT the exact inverse of the full numerical
    spot-average (rough_gap_reflection in air_gap_models.py). It is the right tool for *building
    intuition* about how roughness lifts the high-frequency magnitude; treat extracted sigma_d as
    indicative, not a fitted value. 1/W is clipped (max_boost) so high-f noise does not explode.

Silicon calibration (Samuel's plan): a polished Si wafer pressed on the window is optically
SMOOTH, so the gap is a single well-defined thickness, not a distribution -> set WIDTH ~ 0 and use
only POSITION. Si is essentially non-dispersive (n ~ 3.4175, k ~ 0) across the THz band, so you
KNOW the target: dial d_mean until n is flat at ~3.4175 and k ~ 0. That calibrates the gap
extraction (and exposes residual angle / n_sio2 systematics). See the chat note.

Run (interactive GUI):
    PYTHONPATH=. ../.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/air_gap_slider_explorer.py
Run (headless smoke test -> PNG, no display needed):
    PYTHONPATH=. ../.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/air_gap_slider_explorer.py --headless
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# Capture the genuine plt.show BEFORE importing the headless pipeline module below
# (compare_reflection_pathways forces the Agg backend and monkeypatches plt.show to a
# no-op so the pipeline runs without blocking). We restore both for the interactive GUI.
_REAL_PLT_SHOW = plt.show

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import thz_core.thz_core as core



# ── Geometry / physical constants ───────────────────────────────────────────
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
VACUUM_PERMITTIVITY = 8.8541878128e-12
EXTERNAL_ANGLE_DEG = 45.0
REFRACTIVE_INDEX_SIO2 = 1.95

DISPLAY_BAND_THZ = (0.2, 3.0)


def internal_and_gap_angles(
    n_sio2: float = REFRACTIVE_INDEX_SIO2, external_deg: float = EXTERNAL_ANGLE_DEG,
) -> tuple[float, float]:
    """(theta_in_SiO2, theta_in_gap) for an `external_deg` beam; the air gap returns to 45 deg."""
    external = np.deg2rad(external_deg)
    theta_sio2 = np.arcsin(np.sin(external) / n_sio2)
    theta_gap = np.arcsin(n_sio2 * np.sin(theta_sio2) / 1.0)
    return float(theta_sio2), float(theta_gap)


# ── De-embed primitives (pure, unit-testable) ───────────────────────────────


def gap_round_trip_phase(frequency_hz, gap_thickness_m, gap_angle_rad):
    """2*beta = (omega/c) * 2 d cos(theta_gap) -- the gap pulse round-trip phase."""
    omega = 2.0 * np.pi * frequency_hz
    return omega / SPEED_OF_LIGHT_M_PER_S * 2.0 * gap_thickness_m * np.cos(gap_angle_rad)


def deembed_single_gap(reflection_measured, reflection_front):
    """x = (r_meas - r_front)/(1 - r_front r_meas).  For a single gap, x = r_back e^{-i 2beta}."""
    return (reflection_measured - reflection_front) / (
        1.0 - reflection_front * reflection_measured
    )


def debye_waller_magnitude(frequency_hz, roughness_sigma_m, gap_angle_rad):
    """Single-bounce roughness magnitude factor W = exp(-2 ((omega/c) cos(theta_gap) sigma_d)^2).

    This is the high-frequency magnitude suppression a Gaussian spread of gap thicknesses
    imposes on the coherent (specular) reflection. sigma_d = 0 -> W = 1 (no suppression).
    """
    omega = 2.0 * np.pi * frequency_hz
    exponent = (omega / SPEED_OF_LIGHT_M_PER_S) * np.cos(gap_angle_rad) * roughness_sigma_m
    return np.exp(-2.0 * exponent**2)


def deembed_reflection(
    reflection_measured, reflection_front, frequency_hz,
    gap_position_m, roughness_sigma_m, gap_angle_rad, max_boost: float = 1.0e3,
):
    """Recover r_back from r_meas given a trial gap (position, width).

    POSITION strips the round-trip phase (exact single-gap inverse); WIDTH divides out the
    Debye-Waller magnitude suppression (single-bounce approximation). 1/W is clipped at
    `max_boost` so noise in the (suppressed) high-frequency tail cannot blow up.
    """
    x = deembed_single_gap(reflection_measured, reflection_front)
    phase_strip = np.exp(1j * gap_round_trip_phase(frequency_hz, gap_position_m, gap_angle_rad))
    if roughness_sigma_m > 0.0:
        magnitude_boost = 1.0 / debye_waller_magnitude(frequency_hz, roughness_sigma_m, gap_angle_rad)
        magnitude_boost = np.minimum(magnitude_boost, max_boost)
    else:
        magnitude_boost = 1.0
    return x * phase_strip * magnitude_boost


def invert_air_incidence(frequency_hz, reflection_back, mask, gap_angle_rad, polarization="s"):
    """Invert r_back = r_{air->CNT} with air incidence (n=1) at the in-gap angle -> n, k.

    ``polarization`` must match how the data was measured ('s' or 'p'); passing 's' for a p-pol
    measurement re-inverts with the wrong Fresnel branch and collapses n onto the grazing floor.
    """
    n, k, _ = core.invert_nk_reflection(
        frequency_hz, reflection_back, mask, theta_rad=gap_angle_rad, n_incident=1.0,
        polarization=polarization,
    )
    return n, k


def conductivity_from_nk(frequency_hz, n, k, eps_background):
    """Complex optical conductivity, matching thz_core.derive_eps_sigma.

    n_hat = n + i k (reporting convention) -> eps = n_hat^2 ->
    sigma = -i omega eps0 (eps - eps_background).  Re(sigma) = omega eps0 2 n k (eps_inf-free).
    """
    omega = 2.0 * np.pi * frequency_hz
    eps = (n + 1j * k) ** 2
    return -1j * omega * VACUUM_PERMITTIVITY * (eps - eps_background)


def compute_curves(sample, gap_position_m, roughness_sigma_m, gap_angle_rad, eps_background):
    """De-embed + invert + conductivity for one sample at a trial (position, width).

    `sample` is a dict with keys frequency_hz, mask, r_meas, r_front. Returns dict with
    n, k, sigma (complex) -- all masked to NaN outside the trusted band.
    """
    r_back = deembed_reflection(
        sample["r_meas"], sample["r_front"], sample["frequency_hz"],
        gap_position_m, roughness_sigma_m, gap_angle_rad,
    )
    n, k = invert_air_incidence(
        sample["frequency_hz"], r_back, sample["mask"], gap_angle_rad,
        polarization=sample.get("polarization", "s"),
    )
    sigma = conductivity_from_nk(sample["frequency_hz"], n, k, eps_background)
    return dict(n=n, k=k, sigma=sigma)


# ── Measured-data loader (runs the reflection pipeline once) ─────────────────


def _short_sample_name(filename: str) -> str:
    """Compact label, e.g. 'CNT-0-deg' / 'Si-0-deg' (case-insensitive; falls back to filename)."""
    for token in filename.split("_"):
        lowered = token.lower()
        if (
            "cnt" in lowered
            or "silicon" in lowered
            or lowered == "si"
            or lowered.startswith("si-")  # e.g. 'Si-0-deg'
        ):
            return token
    return filename


def samples_from_dataset(dataset) -> list[dict]:
    """Build de-embed sample records from an ALREADY-PROCESSED reflection dataset.

    Dependency-injection counterpart to ``load_measured_samples``: instead of running a
    pipeline internally, it reads the records straight off a dataset that the caller has
    already pushed through ``transfer_function`` -> ``invert_nk_reflection`` (e.g.
    ``run_me_low-level.py``). This is the way to drive the explorer with the LATEST
    pipeline corrections rather than the standalone ``compare_reflection_pathways`` copy.

    Requires each non-reference sample to carry ``reflection_r``, ``r_reference``, ``n``,
    ``k`` (written by ``invert_nk_reflection``) plus ``fft_freq`` and ``transfer_mask``.
    """
    # Polarisation the data was measured in (single source of truth = the dataset config); the
    # air-incidence re-inversion must use the same 's'/'p' branch as the window inversion did,
    # otherwise a p-pol measurement is re-inverted with the s-pol formula and n collapses.
    dataset_config = getattr(dataset, "config", None) or {}
    polarization = str(dataset_config.get("geometry", {}).get("polarization", "s")).lower()

    samples = []
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        processing = data_obj.processing_dict
        if processing.get("reflection_r") is None:
            continue
        # Prefer the window-geometry inversion as the "naive" baseline: if the de-embed
        # step (deembed_air_gap_reflection) has already run, it overwrote 'n'/'k' with the
        # de-embedded values and stashed the originals under 'n_window'/'k_window'. Using
        # those keeps the slider's dotted reference the true un-de-embedded curve (and keeps
        # the live slider de-embedding r_meas from scratch, never double-correcting).
        n_naive = processing.get("n_window", processing.get("n"))
        k_naive = processing.get("k_window", processing.get("k"))
        samples.append(dict(
            name=_short_sample_name(filename),
            frequency_hz=np.asarray(processing["fft_freq"], dtype=float),
            mask=np.asarray(processing["transfer_mask"], dtype=bool),
            r_meas=np.asarray(processing["reflection_r"], dtype=complex),
            r_front=complex(processing["r_reference"]),
            n_naive=np.asarray(n_naive, dtype=float),
            k_naive=np.asarray(k_naive, dtype=float),
            polarization=polarization,
        ))
    if not samples:
        raise RuntimeError(
            "No non-reference samples with a reflection coefficient were found; run "
            "transfer_function + invert_nk_reflection before samples_from_dataset()."
        )
    return samples


def load_measured_samples(eps_infinity: float = 1) -> list[dict]:
    """Run the shared-axis reflection pipeline; return per-sample measured reflection records.

    Importing the pipeline module forces the Agg backend + disables plt.show (headless); the
    GUI launcher restores both afterwards. Each record carries everything the de-embed needs.
    """
    from compare_reflection_pathways import run_shared_axis, PIPELINE_CONFIG

    dataset = run_shared_axis(PIPELINE_CONFIG, eps_infinity=eps_infinity)
    # Polarisation the data was measured in (single source of truth = the dataset config); the
    # air-incidence re-inversion must use the same 's'/'p' branch as the window inversion did.
    dataset_config = getattr(dataset, "config", None) or {}
    polarization = str(dataset_config.get("geometry", {}).get("polarization", "s")).lower()

    samples = []
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        processing = data_obj.processing_dict
        if processing.get("reflection_r") is None:
            continue
        samples.append(dict(
            name=_short_sample_name(filename),
            frequency_hz=np.asarray(processing["fft_freq"], dtype=float),
            mask=np.asarray(processing["transfer_mask"], dtype=bool),
            r_meas=np.asarray(processing["reflection_r"], dtype=complex),
            r_front=complex(processing["r_reference"]),
            n_naive=np.asarray(processing["n"], dtype=float),
            k_naive=np.asarray(processing["k"], dtype=float),
            polarization=polarization,
        ))
    if not samples:
        raise RuntimeError("No non-reference samples with a reflection coefficient were found.")
    return samples


# ── Plotting helpers shared by GUI + headless ───────────────────────────────

_QUANTITY_PANELS = [
    ("n", "refractive index n", lambda curves: np.real(curves["n"])),
    ("k", "extinction k", lambda curves: np.real(curves["k"])),
    ("sigma_real", "Re sigma (S/m)", lambda curves: np.real(curves["sigma"])),
    ("sigma_imag", "Im sigma (S/m)", lambda curves: np.imag(curves["sigma"])),
]


def _band_mask(sample) -> np.ndarray:
    frequency_thz = sample["frequency_hz"] * 1e-12
    return (
        (frequency_thz >= DISPLAY_BAND_THZ[0]) & (frequency_thz <= DISPLAY_BAND_THZ[1])
        & sample["mask"]
    )


def _naive_curves(sample, eps_background) -> dict:
    sigma = conductivity_from_nk(
        sample["frequency_hz"], sample["n_naive"], sample["k_naive"], eps_background,
    )
    return dict(n=sample["n_naive"], k=sample["k_naive"], sigma=sigma)


# ── Interactive GUI ─────────────────────────────────────────────────────────


def launch_gui(
    samples, gap_angle_rad, eps_background=1.0,
    initial_position_um=15.0, initial_width_um=0.0,
    max_position_um=40.0, max_width_um=20.0,
):
    """Open the slider window. Sliders: gap position (um), roughness width (um), eps_inf."""
    from matplotlib.widgets import Slider

    # Restore an interactive backend + the real plt.show (the pipeline import disabled them).
    plt.close("all")
    for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            plt.switch_backend(backend)
            break
        except Exception:
            continue
    plt.show = _REAL_PLT_SHOW

    figure, axes_grid = plt.subplots(2, 2, figsize=(13, 8))
    figure.subplots_adjust(bottom=0.20, hspace=0.32, wspace=0.26)
    axes = axes_grid.ravel()
    colors = [f"C{i}" for i in range(len(samples))]

    # Persistent line handles: live de-embedded line + faint naive reference, per sample/panel.
    live_lines = {}
    for panel_index, (key, ylabel, _getter) in enumerate(_QUANTITY_PANELS):
        axis = axes[panel_index]
        for sample_index, sample in enumerate(samples):
            band = _band_mask(sample)
            frequency_thz = sample["frequency_hz"][band] * 1e-12
            naive = _naive_curves(sample, eps_background)
            axis.plot(
                frequency_thz, _QUANTITY_PANELS[panel_index][2](naive)[band],
                color=colors[sample_index], lw=1.0, ls=":", alpha=0.45,
            )
            (line,) = axis.plot(
                frequency_thz, np.zeros(band.sum()),
                color=colors[sample_index], lw=1.7, label=sample["name"],
            )
            live_lines[(panel_index, sample_index)] = line
        axis.set_title(ylabel)
        axis.set_xlabel("Frequency (THz)")
        axis.grid(alpha=0.3)
        if key == "n":
            axis.axhline(1.0, color="0.5", ls=":", lw=1.0)  # n = 1 reference
        if panel_index == 0:
            axis.legend(fontsize=8, title="solid = de-embedded\ndotted = naive")

    # Sliders.
    position_axis = figure.add_axes([0.12, 0.10, 0.76, 0.03])
    width_axis = figure.add_axes([0.12, 0.06, 0.76, 0.03])
    eps_axis = figure.add_axes([0.12, 0.02, 0.76, 0.03])
    position_slider = Slider(position_axis, "gap position d_mean (um)", 0.0, max_position_um,
                             valinit=initial_position_um, valstep=0.1)
    width_slider = Slider(width_axis, "roughness width sigma_d (um)", 0.0, max_width_um,
                          valinit=initial_width_um, valstep=0.1)
    eps_slider = Slider(eps_axis, "eps_inf (Im sigma only)", 1.0, 12.0,
                        valinit=eps_background, valstep=0.1)

    def redraw(_event=None):
        position_m = position_slider.val * 1e-6
        width_m = width_slider.val * 1e-6
        eps_inf = eps_slider.val
        for sample_index, sample in enumerate(samples):
            band = _band_mask(sample)
            curves = compute_curves(sample, position_m, width_m, gap_angle_rad, eps_inf)
            for panel_index, (_key, _ylabel, getter) in enumerate(_QUANTITY_PANELS):
                live_lines[(panel_index, sample_index)].set_ydata(getter(curves)[band])
        # autoscale every panel so an over-/under-lift is never hidden; anchor n, k at 0.
        for panel_index, (key, _ylabel, _getter) in enumerate(_QUANTITY_PANELS):
            axis = axes[panel_index]
            axis.relim()
            axis.autoscale_view()
            if key in ("n", "k"):
                axis.set_ylim(bottom=0.0)
        figure.canvas.draw_idle()

    position_slider.on_changed(redraw)
    width_slider.on_changed(redraw)
    eps_slider.on_changed(redraw)
    redraw()

    figure.suptitle(
        "Air-gap de-embed explorer -- drag gap position / roughness width to fit n, k, sigma",
        fontsize=12,
    )
    plt.show()


def launch_from_dataset(
    dataset, n_sio2: float = REFRACTIVE_INDEX_SIO2, external_deg: float = EXTERNAL_ANGLE_DEG,
    eps_background: float = 11.7, initial_position_um: float = 0.0, initial_width_um: float = 0.0,
):
    """One-call launcher: open the slider on an already-processed reflection dataset.

    Thin wrapper for pipeline scripts (``run_me_low-level.py``): extracts the sample
    records, resolves the in-gap angle from the geometry, and opens the GUI. The gap
    angle returns to ``external_deg`` (the air gap sees the same external incidence as
    the beam entering the window), so it is geometry-consistent with the inversion.
    """
    samples = samples_from_dataset(dataset)
    _, gap_angle_rad = internal_and_gap_angles(n_sio2=n_sio2, external_deg=external_deg)
    print(
        f"[air-gap explorer] {len(samples)} sample(s): "
        f"{', '.join(s['name'] for s in samples)}; gap angle "
        f"{np.rad2deg(gap_angle_rad):.2f} deg (cos {np.cos(gap_angle_rad):.3f})."
    )
    launch_gui(
        samples, gap_angle_rad, eps_background=eps_background,
        initial_position_um=initial_position_um, initial_width_um=initial_width_um,
    )


# ── Headless smoke test ─────────────────────────────────────────────────────


def render_static(samples, gap_angle_rad, position_um, width_um, eps_background, output_path):
    """Agg render of one parameter set (no display) + return band stats for sanity checks."""
    matplotlib.use("Agg", force=True)
    figure, axes_grid = plt.subplots(2, 2, figsize=(13, 8))
    figure.subplots_adjust(hspace=0.32, wspace=0.26)
    axes = axes_grid.ravel()
    colors = [f"C{i}" for i in range(len(samples))]

    stats = {}
    for sample_index, sample in enumerate(samples):
        band = _band_mask(sample)
        frequency_thz = sample["frequency_hz"][band] * 1e-12
        curves = compute_curves(sample, position_um * 1e-6, width_um * 1e-6, gap_angle_rad, eps_background)
        naive = _naive_curves(sample, eps_background)
        for panel_index, (key, ylabel, getter) in enumerate(_QUANTITY_PANELS):
            axis = axes[panel_index]
            axis.plot(frequency_thz, getter(naive)[band], color=colors[sample_index],
                      lw=1.0, ls=":", alpha=0.45)
            axis.plot(frequency_thz, getter(curves)[band], color=colors[sample_index],
                      lw=1.7, label=sample["name"])
            axis.set_title(ylabel)
            axis.set_xlabel("Frequency (THz)")
            axis.grid(alpha=0.3)
            if key == "n":
                axis.axhline(1.0, color="0.5", ls=":", lw=1.0)
        n_band = np.real(curves["n"])[band]
        stats[sample["name"]] = dict(
            n_mean=float(np.nanmean(n_band)), n_min=float(np.nanmin(n_band)),
            sigma_real_mean=float(np.nanmean(np.real(curves["sigma"])[band])),
        )
    axes[0].legend(fontsize=8)
    figure.suptitle(
        f"Air-gap de-embed (static) d_mean={position_um:.1f} um, sigma_d={width_um:.1f} um, "
        f"eps_inf={eps_background:.1f}", fontsize=12,
    )
    figure.savefig(output_path, dpi=130)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true",
                        help="render a static frame to PNG + print band stats (no GUI).")
    parser.add_argument("--position-um", type=float, default=15.0,
                        help="initial gap position d_mean (um).")
    parser.add_argument("--width-um", type=float, default=0.0,
                        help="initial roughness width sigma_d (um).")
    parser.add_argument("--eps-inf", type=float, default=11.7,
                        help="background permittivity for the conductivity (1.0 = vacuum).")
    args = parser.parse_args()

    _, gap_angle_rad = internal_and_gap_angles()
    print(f"gap angle {np.rad2deg(gap_angle_rad):.2f} deg (cos {np.cos(gap_angle_rad):.3f})")
    samples = load_measured_samples(eps_infinity=args.eps_inf)
    print(f"loaded {len(samples)} sample(s): {', '.join(s['name'] for s in samples)}")

    if args.headless:
        output_path = os.path.join(os.path.dirname(__file__), "air_gap_slider_explorer.png")
        stats = render_static(samples, gap_angle_rad, args.position_um, args.width_um,
                              args.eps_inf, output_path)
        print(f"\nstatic frame: d_mean={args.position_um} um, sigma_d={args.width_um} um, "
              f"eps_inf={args.eps_inf}")
        for name, stat in stats.items():
            print(f"  {name}: n mean {stat['n_mean']:.3f} (min {stat['n_min']:.3f}), "
                  f"Re sigma mean {stat['sigma_real_mean']:.1f} S/m")
        print(f"saved figure: {output_path}")
    else:
        launch_gui(samples, gap_angle_rad, eps_background=args.eps_inf,
                   initial_position_um=args.position_um, initial_width_um=args.width_um)


if __name__ == "__main__":
    main()
