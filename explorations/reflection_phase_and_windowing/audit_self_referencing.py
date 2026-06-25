"""AUDIT 2: is the front-pulse self-referencing behaving as expected?

Two parts.

PART A — analytic correctness (documented, then checked numerically below):
  Self-ref forms  H = (Y2_s/Y1_s)/(Y2_r/Y1_r) = W_s/W_r.  Model every spectrum as
      Y1 = D1·S·r_front ,  Y2 = D2·S·K·r_back
  where D1,D2 = the per-pulse detection responses (the cone effect — front and back focus
  different cone-sections so D1≠D2), S = source spectrum (varies acquisition-to-acquisition
  = drift), K = window propagation/transmission, r_front/r_back the interface reflections.
  Then W = Y2/Y1 = (D2/D1)·K·r_back/r_front, and
      H = W_s/W_r = r_back,sample / r_back,reference.
  EVERYTHING common to the two acquisitions cancels — including the cone factor D2/D1 and the
  drift S. So in the ideal flat-interface model self-ref is EXACT and cone-robust. The cone
  effect only fails to cancel if D2 differs between sample and reference (it does not for a flat
  specular interface) — or if you reference using the front pulse ALONE (no measured back).

PART B — measure the cone transfer on REAL data using the bare-window model:
  For the SiO2 reference, W_r is also predicted analytically by window_transfer_model(n_SiO2).
  The ratio  cone(f) = W_r_measured / W_r_model  isolates the per-pulse detection mismatch
  (the cone) PLUS any error in the assumed n_SiO2. Plug a transmission-measured n_SiO2(f) to
  remove the model error and leave the pure cone transfer. If cone(f) ≈ 1, the front pulse is a
  faithful reference and self-ref is clean; if not, its size tells you how much the front/back
  cone mismatch matters (and, crucially, it STILL cancels in H because H divides by the measured
  W_r, not the model — so this is a diagnostic of robustness, not a correction you must apply).

Run:  PYTHONIOENCODING=utf-8 PYTHONPATH=. ../.venv/Scripts/python.exe explorations/reflection_phase_and_windowing/audit_self_referencing.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None  # headless

import thz_core.thz_core as core

# Reuse the validated shared-axis pipeline front-end.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "air_gap_cnt_reflection"))
from compare_reflection_pathways import run_shared_axis, PIPELINE_CONFIG

THICKNESS_M = 2.08e-3
THETA_EXT_DEG = 45.0
N_SIO2 = 1.95           # swap for transmission-measured n_SiO2(f) to remove model error
BAND_THZ = (0.2, 3.0)


def _measured_W(reflection_obj):
    """W = Y2/Y1 from the stored segment spectra (absolute-time factor cancels in the ratio)."""
    y1 = reflection_obj.first_reflection.processing_dict["fft_spectrum"]
    y2 = reflection_obj.second_reflection.processing_dict["fft_spectrum"]
    return y2 / y1


def _segment_mask(obj):
    return obj.second_reflection.processing_dict.get("transfer_mask")


def main():
    dataset = run_shared_axis(PIPELINE_CONFIG, eps_infinity=1.0)

    # Locate the reference object and the sample objects.
    reference_obj = None
    sample_objs = {}
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            reference_obj = data_obj
        elif data_obj.second_reflection.processing_dict.get("fft_spectrum") is not None:
            sample_objs[filename] = data_obj
    if reference_obj is None:
        raise RuntimeError("No reference object found — cannot audit self-referencing.")

    freq = reference_obj.first_reflection.processing_dict["fft_freq"]
    f_thz = freq * 1e-12
    band = (f_thz >= BAND_THZ[0]) & (f_thz <= BAND_THZ[1])

    # PART B: cone transfer = measured W_r / model W_r.
    W_r_measured = _measured_W(reference_obj)
    W_r_model = core.window_transfer_model(
        freq, complex(N_SIO2), THICKNESS_M, np.deg2rad(THETA_EXT_DEG),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cone = W_r_measured / W_r_model

    ref_mask = _segment_mask(reference_obj)
    trusted = band & (ref_mask if ref_mask is not None else True) & np.isfinite(cone)

    print("=== AUDIT 2: self-referencing ===")
    print(f"window model: n_SiO2={N_SIO2}, d={THICKNESS_M*1e3:.2f} mm, theta_ext={THETA_EXT_DEG} deg")
    print(f"|W_r| measured  band-median = {np.nanmedian(np.abs(W_r_measured[trusted])):.3f}")
    print(f"|W_r| model     band-median = {np.nanmedian(np.abs(W_r_model[trusted])):.3f}")
    cone_mag = np.abs(cone[trusted])
    cone_phase = np.angle(cone[trusted])
    print("\nCONE / DETECTION TRANSFER (measured W_r / model W_r), over "
          f"{BAND_THZ[0]}-{BAND_THZ[1]} THz trusted band:")
    print(f"  |cone|  median {np.nanmedian(cone_mag):.3f}, "
          f"range {np.nanmin(cone_mag):.3f}-{np.nanmax(cone_mag):.3f} "
          f"(=1 would mean the front pulse is a perfect reference)")
    print(f"  arg(cone) spans {np.rad2deg(np.nanmax(cone_phase) - np.nanmin(cone_phase)):.1f} deg "
          f"(a linear ramp here = n_SiO2/thickness mismatch, not cone)")
    print("  NOTE: this includes assumed-n_SiO2 model error; plug transmission n_SiO2(f) to isolate the cone.")

    # Is the magnitude gap consistent with real SiO2 absorption (model uses k=0)?
    # The back pulse traverses the window twice: round-trip path L_rt = 2 d / cos(theta_internal).
    theta_internal = np.arcsin(np.sin(np.deg2rad(THETA_EXT_DEG)) / N_SIO2)
    round_trip_path_cm = (2.0 * THICKNESS_M / np.cos(theta_internal)) * 1e2
    median_cone_mag = float(np.nanmedian(cone_mag))
    implied_amplitude_alpha = -np.log(median_cone_mag) / round_trip_path_cm
    print(f"\nIs the |W_r| gap just SiO2 absorption (model has k=0)?")
    print(f"  round-trip path in window = {round_trip_path_cm*10:.2f} mm")
    print(f"  implied amplitude absorption from median |cone|={median_cone_mag:.3f}: "
          f"alpha_amp ~ {implied_amplitude_alpha:.1f} /cm  (power alpha ~ {2*implied_amplitude_alpha:.1f} /cm)")
    print("  -> compare with fused-silica THz absorption (~a few to ~10 /cm power, 1-2.5 THz). If")
    print("     consistent, the front pulse is a sound reference and |W_r|<model is REAL absorption,")
    print("     which CANCELS in H (sample sees the same window). Plug transmission n_SiO2(f) to verify.")

    # The cone STILL cancels in H (H divides by the measured W_r). Show that the self-ref H is
    # exactly W_s/W_r_measured and is therefore independent of the model.
    print("\nSelf-ref H per sample uses the MEASURED W_r (cone already included -> cancels):")
    for filename, sample_obj in sample_objs.items():
        H_stored = sample_obj.second_reflection.processing_dict.get("transfer_H")
        if H_stored is None:
            continue
        W_s = _measured_W(sample_obj)
        H_reconstructed = W_s / W_r_measured
        agree = np.nanmax(np.abs((H_stored - H_reconstructed)[trusted]))
        short = filename.split("_")[1] if "_" in filename else filename
        print(f"  {short:12s}: H_stored == W_s/W_r_measured to {agree:.1e} "
              f"(confirms the pipeline's self-ref identity)")

    # Figures.
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    figure.suptitle("Audit 2 — self-referencing: measured vs model window W_r, and the cone transfer", fontsize=12)

    ax = axes[0][0]
    ax.plot(f_thz[band], np.abs(W_r_measured[band]), label="|W_r| measured")
    ax.plot(f_thz[band], np.abs(W_r_model[band]), "--", label="|W_r| model (n_SiO2)")
    ax.set_title("Reference window transfer magnitude")
    ax.set_xlabel("THz"); ax.set_ylabel("|W_r|"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(f_thz[band], np.unwrap(np.angle(W_r_measured[band])), label="arg W_r measured")
    ax.plot(f_thz[band], np.unwrap(np.angle(W_r_model[band])), "--", label="arg W_r model")
    ax.set_title("Reference window transfer phase")
    ax.set_xlabel("THz"); ax.set_ylabel("phase (rad)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(f_thz[band], np.abs(cone[band]))
    ax.axhline(1.0, color="0.5", ls=":")
    ax.set_title("|cone| = |W_r measured / W_r model|  (=1 -> front is a faithful reference)")
    ax.set_xlabel("THz"); ax.set_ylabel("|cone|"); ax.grid(alpha=0.3); ax.set_ylim(0, 2)

    ax = axes[1][1]
    ax.plot(f_thz[band], np.unwrap(np.angle(cone[band])))
    ax.set_title("arg(cone)  (linear ramp = n_SiO2/thickness error; curvature = real cone effect)")
    ax.set_xlabel("THz"); ax.set_ylabel("phase (rad)"); ax.grid(alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "audit_self_referencing.png")
    figure.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
