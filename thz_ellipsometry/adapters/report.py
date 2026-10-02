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

    sample = outcome.sample_series
    lines.append(f"   sample      : {len(sample)} acquisitions at "
                 f"{len(np.unique(sample.angles_deg % 360.0))} polarisation angles, "
                 f"{int(sample.scan_counts.sum())} scans total"
                 + ("" if sample.elapsed_seconds is None
                    else f", {np.ptp(sample.elapsed_seconds) / 60.0:.1f} min span"))
    for (role, probe), series in sorted(outcome.series.items(), key=lambda item: str(item[0])):
        if (role, probe) == ("sample", outcome.primary_probe_deg):
            continue
        lines.append(f"   {role:<12}: {len(series)} acquisitions, probe {probe} deg")
    lines.append(f"   band        : {band.min()/1e12:.2f} to {band.max()/1e12:.2f} THz, "
                 f"{band.size} points")
    lines.append(f"   geometry    : {np.rad2deg(inversion.incidence_angle_rad):.2f} deg "
                 f"incidence, blur "
                 f"{np.rad2deg(inversion.angular_spread_rad):.2f} deg")

    if result.calibration is not None:
        lines.append(f"   calibration : {outcome.calibration_source}: d_p/d_s = "
                     f"{result.calibration.ratio:.4f} ({result.calibration.reference_name})")
    lines.append(f"   angle       : {np.rad2deg(inversion.incidence_angle_rad):.3f} deg "
                 f"({result.incidence_angle_source})")
    if result.tilt_fit is not None:
        lines.append(f"   tilt fit    : {result.tilt_fit.tilt_deg:+.3f} +/- "
                     f"{result.tilt_fit.tilt_standard_error_deg:.3f} deg "
                     f"({result.tilt_fit.model_name}: "
                     + ", ".join(f"{name}={value:.4g}" for name, value in
                                 result.tilt_fit.model_parameters.items()) + ")")
    if result.incidence_angle_fit is not None:
        lines.append(f"   angle check : fitted {result.incidence_angle_fit.angle_deg:.3f} deg "
                     f"vs nominal {result.incidence_angle_fit.nominal_angle_deg:.3f} deg "
                     f"({result.incidence_angle_fit.discrepancy_deg:+.3f})")

    lines.append("   quality flags:")
    lines.extend(_format_quality_flags(result.quality_flags))

    error_text = ""
    if result.index_standard_error is not None:
        error_text = (f" +/- {np.median(result.index_standard_error):.4f} "
                      "(median noise bar, same for n and k)")
    lines.append(f"   result      : n = {np.median(inversion.refractive_index):.4f} "
                 f"(median), k = {np.median(inversion.extinction):.5f} (median){error_text}")
    systematic = result.index_angle_systematic
    if systematic is not None:
        lines.append(f"   systematic  : +/- {np.median(systematic):.4f} in |N| for the stated "
                     f"+/- {result.incidence_angle_uncertainty_deg:g} deg angle uncertainty "
                     "(not in the noise bar)")
    conductivity = np.median(inversion.conductivity_real_si) / 100.0
    lines.append(f"                 sigma_1 = {conductivity:.3f} S/cm (median)")

    for warning in outcome.warnings or []:
        lines.append(f"   WARNING     : {warning}")

    findings = getattr(outcome, "findings", None) or []
    lines.append("")
    lines.append(f"[check_assumptions] {len(findings)} finding(s) from the registered "
                 "ellipsometry diagnostics")
    for finding in findings:
        lines.append(finding.format(indent="   "))

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

    for values, name in ((inversion.refractive_index, "n"), (inversion.extinction, "k")):
        line, = axes[0, 1].plot(band_thz, values, label=name)
        if result.index_standard_error is not None:
            axes[0, 1].fill_between(band_thz, values - result.index_standard_error,
                                    values + result.index_standard_error,
                                    color=line.get_color(), alpha=0.25, linewidth=0)
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
