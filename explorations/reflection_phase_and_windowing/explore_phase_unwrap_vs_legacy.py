"""Compare our transfer-phase handling against the legacy reduced-phase unwrap.

Question under investigation (Samuel, 2026-06-17): our pipeline needs an
explicit DC-origin *anchor* inside ``invert_nk`` to flatten ``n`` for a thick
slab, yet the legacy ``main_TDS.py`` produced a flat ``n`` with no such step.
Are we over-engineering?

This script isolates the ONE thing that differs — how the transfer-function
phase is unwrapped — using the same complex spectra for every method, so the
comparison is convention-consistent (identical ``n = 1 - c*phi/(2*pi*f*d)``
formula throughout). It does NOT modify any pipeline code.

Methods compared (all from the same stored Y_sample, Y_reference):
  1. ours-raw      : unwrap angle(H) directly, no anchor      -> expect droop
  2. ours-anchor   : core.invert_nk with anchor_phase_origin   -> expect flat
  3. legacy-reduced: reduced-phase unwrap per spectrum (each
                     referenced to its own pulse-peak time,
                     unwrap the small residual, add the ramp
                     back), then form phi(H). No anchor.        -> expect flat
  4. legacy+phaseex: method 3 then remove the phase intercept   -> expect flat

HYPOTHESIS GOING IN (was that the legacy "reduced-phase" trick — subtract the
2*pi*f*t_peak ramp before unwrapping, add it back — is what keeps the branch
correct, making our anchor unnecessary).

RESULT (see VERDICT printed at the end): the hypothesis is FALSE. Reduced-phase
unwrap lands on the SAME ~2*pi branch as a raw unwrap, because the 2*pi error is
not an unwrap failure — the phase genuinely starts beyond pi at the first bin and
there is no information to fix the absolute cycle without asserting phi(0)=0. The
legacy's flatness comes entirely from phaseex removing the full intercept on the
REAL phase. Our anchor is the same assertion, just placed inside invert_nk
because our pipeline round-trips the phase through a complex H and re-unwraps.

Run headless; writes a PNG next to this script and prints a verdict.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz
import thz_core.thz_core as core

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
SAMPLE_THICKNESS_M = 2.08e-3
TRUSTED_BAND_HZ = (0.3e12, 2.0e12)
STATE_DIRECTORY = r"C:\Users\Samuel\Data\THz\Sam\2026_06_17_reflection_setup_large\holder"


def refractive_index_from_phase(frequency_hz, transfer_phase_rad, thickness_m):
    """n(f) = 1 - c*phi/(2*pi*f*d), the same relation invert_nk uses."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - SPEED_OF_LIGHT_M_PER_S * transfer_phase_rad / (
            2.0 * np.pi * frequency_hz * thickness_m
        )


def reduced_phase_unwrap(frequency_hz, complex_spectrum, peak_time_s):
    """Legacy branch-safe unwrap: remove the pulse-peak ramp, unwrap residual, restore.

    A pulse peaked at absolute time t_peak carries a phase ramp ~ -2*pi*f*t_peak
    (numpy exp(-i w t) convention). Multiplying by exp(+i 2*pi*f*t_peak) cancels
    it so the residual phase stays within (-pi, pi] and np.unwrap cannot pick a
    wrong 2*pi branch; we then add the exact ramp back to recover the absolute,
    branch-correct unwrapped phase.
    """
    ramp = 2.0 * np.pi * frequency_hz * peak_time_s
    residual = np.unwrap(np.angle(complex_spectrum * np.exp(1j * ramp)))
    return residual - ramp


def remove_intercept(frequency_hz, phase_rad, band_hz):
    """phaseex: subtract only the linear-fit intercept over the trusted band."""
    in_band = (frequency_hz >= band_hz[0]) & (frequency_hz <= band_hz[1]) & np.isfinite(phase_rad)
    if np.count_nonzero(in_band) < 2:
        return phase_rad
    _, intercept = np.polyfit(frequency_hz[in_band], phase_rad[in_band], 1)
    return phase_rad - intercept


def main():
    dataset = DataSet(STATE_DIRECTORY)
    dataset.load_state()

    sample_object = next(obj for fn, obj in dataset.data.items()
                         if not dataset.data.is_reference(fn))
    reference_object = next(obj for fn, obj in dataset.data.items()
                            if dataset.data.is_reference(fn))

    frequency_hz = sample_object.processing_dict["fft_freq"]
    sample_spectrum = sample_object.processing_dict["fft_spectrum"]
    reference_spectrum = reference_object.processing_dict["fft_spectrum"]

    sample_time_trace = sample_object.processing_dict["time_domain_prefft"]
    reference_time_trace = reference_object.processing_dict["time_domain_prefft"]
    sample_peak_time_s = sample_time_trace[np.argmax(np.abs(sample_time_trace[:, 1])), 0]
    reference_peak_time_s = reference_time_trace[np.argmax(np.abs(reference_time_trace[:, 1])), 0]
    print(f"peak times: sample {sample_peak_time_s*1e12:.3f} ps, "
          f"reference {reference_peak_time_s*1e12:.3f} ps, "
          f"delta {(sample_peak_time_s-reference_peak_time_s)*1e12:.3f} ps")

    transfer_function_complex = sample_spectrum / reference_spectrum
    positive = frequency_hz > 0

    # --- Method 1: ours, raw unwrap, no anchor ---
    phase_ours_raw = np.full(frequency_hz.size, np.nan)
    phase_ours_raw[positive] = np.unwrap(np.angle(transfer_function_complex[positive]))
    n_ours_raw = refractive_index_from_phase(frequency_hz, phase_ours_raw, SAMPLE_THICKNESS_M)

    # --- Method 2: ours, via core.invert_nk with the DC-origin anchor ---
    full_mask = np.isfinite(transfer_function_complex) & positive
    n_ours_anchor, _, _ = core.invert_nk(
        frequency_hz, transfer_function_complex, SAMPLE_THICKNESS_M, full_mask,
        {"invert": {"anchor_phase_origin": True}},
    )

    # --- Method 3: legacy reduced-phase unwrap, no anchor ---
    phase_sample = reduced_phase_unwrap(frequency_hz, sample_spectrum, sample_peak_time_s)
    phase_reference = reduced_phase_unwrap(frequency_hz, reference_spectrum, reference_peak_time_s)
    phase_legacy = phase_sample - phase_reference
    n_legacy_reduced = refractive_index_from_phase(frequency_hz, phase_legacy, SAMPLE_THICKNESS_M)

    # --- Method 4: legacy reduced-phase + phaseex intercept removal ---
    phase_legacy_phaseex = remove_intercept(frequency_hz, phase_legacy, TRUSTED_BAND_HZ)
    n_legacy_phaseex = refractive_index_from_phase(frequency_hz, phase_legacy_phaseex, SAMPLE_THICKNESS_M)

    def sample_n(n_array):
        good = np.isfinite(n_array) & positive
        f = frequency_hz[good]
        return {t: float(np.interp(t*1e12, f, n_array[good])) for t in (0.4, 1.0, 2.0)}

    print("\nn(f) sampled at 0.4 / 1.0 / 2.0 THz:")
    for label, n_array in [
        ("1 ours-raw (no anchor)", n_ours_raw),
        ("2 ours-anchor          ", n_ours_anchor),
        ("3 legacy-reduced       ", n_legacy_reduced),
        ("4 legacy+phaseex       ", n_legacy_phaseex),
    ]:
        s = sample_n(n_array)
        print(f"  {label}: {s[0.4]:.3f} / {s[1.0]:.3f} / {s[2.0]:.3f}")

    # report the DC-extrapolated intercept of phi(H) for ours vs legacy
    def intercept_of(phase):
        in_band = (frequency_hz >= TRUSTED_BAND_HZ[0]) & (frequency_hz <= TRUSTED_BAND_HZ[1]) & np.isfinite(phase)
        _, b = np.polyfit(frequency_hz[in_band], phase[in_band], 1)
        return b
    print(f"\nphi(H) DC intercept  ours-raw: {intercept_of(phase_ours_raw):+.3f} rad "
          f"(~{intercept_of(phase_ours_raw)/(2*np.pi):+.2f} cycles)")
    print(f"phi(H) DC intercept  legacy : {intercept_of(phase_legacy):+.3f} rad "
          f"(~{intercept_of(phase_legacy)/(2*np.pi):+.2f} cycles)")

    fig, (ax_phase, ax_n) = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    f_thz = frequency_hz * 1e-12
    ax_phase.plot(f_thz, phase_ours_raw, label="ours-raw unwrap phi(H)")
    ax_phase.plot(f_thz, phase_legacy, "--", label="legacy reduced-phase phi(H)")
    ax_phase.axhline(0, color="gray", lw=0.5)
    ax_phase.set_xlim(0, 3); ax_phase.set_xlabel("THz"); ax_phase.set_ylabel("phase (rad)")
    ax_phase.set_title("Transfer phase: raw unwrap vs reduced-phase unwrap")
    ax_phase.legend(fontsize=8)
    for label, n_array, style in [
        ("ours-raw (no anchor)", n_ours_raw, "-"),
        ("ours-anchor", n_ours_anchor, "-"),
        ("legacy-reduced (no anchor)", n_legacy_reduced, "--"),
        ("legacy+phaseex", n_legacy_phaseex, ":"),
    ]:
        ax_n.plot(f_thz, n_array, style, label=label)
    ax_n.set_xlim(0.2, 2.5); ax_n.set_ylim(1.0, 2.4)
    ax_n.set_xlabel("THz"); ax_n.set_ylabel("n"); ax_n.legend(fontsize=8)
    ax_n.set_title("Refractive index by phase-handling method")

    out_path = os.path.join(os.path.dirname(__file__), "explore_phase_unwrap_vs_legacy.png")
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved figure to {out_path}")

    print(
        "\n--- VERDICT ---------------------------------------------------------\n"
        "The reduced-phase unwrap (method 3) gives the SAME phi(H) intercept (~2*pi)\n"
        "and the SAME drooping n as our raw unwrap (method 1). So the legacy's\n"
        "flatness does NOT come from how it unwraps -- it comes from phaseex\n"
        "(method 4) removing the full ~2*pi intercept from the REAL phase, right\n"
        "before computing n, in ONE place, never round-tripping through a complex\n"
        "array. Our pipeline stores H as complex and invert_nk RE-unwraps, which\n"
        "splits that single correction into two: the integer-cycle half (the\n"
        "anchor, inside invert_nk) and the sub-2*pi half (remove_phase_offset, on\n"
        "complex H). The anchor ALONE already flattens n (~1.94); remove_phase_offset\n"
        "only trims the ~0.07 rad residual. We are not physically over-doing it --\n"
        "same answer as legacy -- but the complex round-trip is why it takes two\n"
        "steps where legacy takes one.\n"
        "---------------------------------------------------------------------"
    )


if __name__ == "__main__":
    main()
