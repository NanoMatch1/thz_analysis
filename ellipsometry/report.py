"""Text and figure output for an ellipsometry run.

Every method prints a ``[method_name]`` outcome line, following the house convention, so a run
reads as a log of what happened rather than a wall of numbers.
"""

from __future__ import annotations

import numpy as np

__all__ = ["format_run_report", "plot_run"]


def _format_quality_flags(flags):
    lines = []
    for name, value in flags.items():
        if isinstance(value, bool):
            lines.append(f"      {name:<36} {'yes' if value else 'NO'}")
        else:
            lines.append(f"      {name:<36} {value:.5g}")
    return lines


def format_run_report(outcome):
    """Human-readable summary of a RunOutcome."""
    result = outcome.result
    inversion = result.inversion
    band = inversion.frequencies_hz
    lines = ["[run_ellipsometry] summary"]

    lines.append(f"   sample      : {len(outcome.sample_series)} emitter angles "
                 f"({outcome.sample_series.angles_deg.min():.1f} to "
                 f"{outcome.sample_series.angles_deg.max():.1f} deg), "
                 f"{int(outcome.sample_series.scan_counts.sum())} scans total")
    if outcome.reference_series is not None:
        lines.append(f"   reference   : {len(outcome.reference_series)} angles, "
                     f"'{result.calibration.reference_name}'")
    lines.append(f"   band        : {band.min()/1e12:.2f} to {band.max()/1e12:.2f} THz, "
                 f"{band.size} points")
    lines.append(f"   geometry    : {np.rad2deg(inversion.incidence_angle_rad):.2f} deg "
                 f"incidence, blur "
                 f"{np.rad2deg(inversion.angular_spread_rad):.2f} deg")

    if result.calibration is not None:
        lines.append(f"   calibration : d_p/d_s = {result.calibration.ratio:.4f}, "
                     f"emitter offset {result.calibration.emitter_offset_deg:.3f} deg")
    if result.incidence_angle_fit is not None:
        lines.append(f"   angle check : fitted {result.incidence_angle_fit.angle_deg:.3f} deg "
                     f"vs nominal {result.incidence_angle_fit.nominal_angle_deg:.3f} deg "
                     f"({result.incidence_angle_fit.discrepancy_deg:+.3f})")

    lines.append("   quality flags:")
    lines.extend(_format_quality_flags(result.quality_flags))

    lines.append(f"   result      : n = {np.median(inversion.refractive_index):.4f} "
                 f"(median), k = {np.median(inversion.extinction):.5f} (median)")
    conductivity = np.median(inversion.conductivity_real_si) / 100.0
    lines.append(f"                 sigma_1 = {conductivity:.3f} S/cm (median)")

    for warning in outcome.warnings or []:
        lines.append(f"   WARNING     : {warning}")

    if outcome.report is not None:
        lines.append("")
        lines.append(str(outcome.report))
    return "\n".join(lines)


def plot_run(outcome, show=True, save_path=None):
    """Four-panel figure: tan(Psi)/Delta, n and k, the harmonic residual, the channel ratio."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = outcome.result
    inversion = result.inversion
    band_thz = inversion.frequencies_hz / 1e12
    all_thz = outcome.frequencies_hz / 1e12

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    axes[0, 0].plot(band_thz, inversion.tan_psi, label=r"$\tan\Psi$")
    twin = axes[0, 0].twinx()
    twin.plot(band_thz, np.rad2deg(inversion.delta_rad), color="tab:orange",
              label=r"$\Delta$")
    axes[0, 0].set_ylabel(r"$\tan\Psi$")
    twin.set_ylabel(r"$\Delta$ [deg]")
    axes[0, 0].set_xlabel("frequency [THz]")
    axes[0, 0].set_title("ellipsometric parameters")

    axes[0, 1].plot(band_thz, inversion.refractive_index, label="n")
    axes[0, 1].plot(band_thz, inversion.extinction, label="k")
    axes[0, 1].set_xlabel("frequency [THz]")
    axes[0, 1].set_title("optical constants")
    axes[0, 1].legend()

    axes[1, 0].semilogy(all_thz, np.maximum(result.sample_fit.residual_per_frequency, 1e-12),
                        label="sample")
    if result.reference_fit is not None:
        axes[1, 0].semilogy(
            all_thz, np.maximum(result.reference_fit.residual_per_frequency, 1e-12),
            label="reference")
    axes[1, 0].set_xlabel("frequency [THz]")
    axes[1, 0].set_title("harmonic residual (quality flag)")
    axes[1, 0].legend()

    if result.calibration is not None:
        axes[1, 1].plot(all_thz, np.abs(result.calibration.ratio_per_frequency),
                        label="|d_p/d_s|")
        axes[1, 1].axhline(abs(result.calibration.ratio), color="k", linestyle="--",
                           label="applied")
        axes[1, 1].set_title("channel ratio (should be flat)")
        axes[1, 1].set_xlabel("frequency [THz]")
        axes[1, 1].legend()
    else:
        axes[1, 1].set_axis_off()

    if save_path:
        figure.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure
