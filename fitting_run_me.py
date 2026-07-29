"""Fit the conductivity of a processed dataset — the pipeline-integrated way.

This replaces the old per-sample FitGUI loop (which read a saved *database* and opened one window
per file). Now the fitting is a single call on a session bundle:

  1. a run_me_*.py wrote a `.thzbundle` (with the Monte-Carlo error bars from
     `propagate_uncertainty`);
  2. `session_bundle.load_session` restores it instantly (no recompute);
  3. `fitting.fit_interactive` opens ONE multi-file fit GUI (prev/next navigation) showing those
     error bars, and writes each `FitResult` back into `processing_dict`;
  4. we re-save the bundle so the fits travel with it and reappear as the viewer's Drude overlay.

You can also do all of this without a script: `python open_session.py <bundle> --fit`.
For a headless / automatic Drude fit instead, use `conductivity_fitting.fit_conductivity`.
"""

from __future__ import annotations

from dataset_core.adapters import session_bundle
from dataset_core.adapters import fitting
from dataset_core.adapters import results_viewer

import matplotlib.pyplot as plt
import numpy as np

from dataset_core.adapters import thz_adapter as thz

def drude_fit(freq, plasma_freq, scattering_rate):
    """Drude model for conductivity."""
    return (plasma_freq ** 2) / (1j * 2 * np.pi * freq + scattering_rate)

def drude_fit_dc(freq, sigma_dc, scattering_rate):
    """Drude model for conductivity using DC conductivity.

    freq : frequency in Hz
    scattering_rate : 1/tau in Hz
    """
    return sigma_dc / (1 - 1j * 2 * np.pi * freq / scattering_rate)

def generate_channel(dataX, dataY, depth=0.2):
    '''Generate a channel around the dataY curve for visualization tracing.'''
    upper = dataY + dataY * depth
    lower = dataY - dataY * depth
    return upper, lower

def strip_nan(data):
    """Remove NaN values from the data array."""
    return data[~np.isnan(data)]

def _plot_with_snr_mask(ax, freq_thz, dataY, mask=None, label=None, color=None, linestyle='solid', marker='o', config={}):
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
    ax.scatter(freq_thz[mask], dataY[mask], label=label, s=30, marker=marker, color=color, alpha=0.8) # high-SNR points
        # ax.scatter(freq_thz[~mask], dataY[~mask], color=color, marker='.', alpha=0.1, s=1, label=None)  # low-SNR points, faint
    # else:
        # ax.scatter(freq_thz, dataY, label=label, color=color, marker=marker, alpha=0.8, s=10)
    if guideline:
        upper, lower = generate_channel(freq_thz[mask], dataY[mask])
        ax.fill_between(freq_thz[mask], lower, upper, color=color, alpha=0.1)  # guideline for reference
    


def fit_bundle(bundle_dir: str, *, quantity: str = "sigma",
               model: str = "drude_conductivity", fit_band_thz=(0.35, 1.6),
               save: bool = True):
    """Load a bundle, fit its conductivity interactively, re-save, and show the result."""
    dataset = session_bundle.load_session(bundle_dir)
    fitting.fit_interactive(dataset, quantity=quantity, model=model, fit_band_thz=fit_band_thz)
    if save:
        session_bundle.save_session(dataset, bundle_dir, notes="fits added via fitting_run_me")
    results_viewer.launch_results_viewer(dataset)
    return dataset

def get_fit_dict(bundle_dirs: list, *, quantity: str = "sigma", model: str = "drude_conductivity"):
    """Load a bundle and plot the fitted conductivity results."""
    fit_dict = {}
    for bundle_dir in bundle_dirs:
        print("opening bundle:", bundle_dir)
        dataset = session_bundle.load_session(bundle_dir)
        # fitting.plot_fit_results(dataset, quantity=quantity, model=model)
        for filename, data_obj in dataset.data.items():
            fit_results = data_obj.processing_dict.get('fit_result')
            errors = data_obj.processing_dict.get('propagated_error')
            print(f"file: {filename}, fit_results: {fit_results}, errors: {errors}")
            if fit_results is not None:
                fit_dict[filename] = fit_results
                print("adding fit result for:", filename)
    
    return fit_dict
            
            
            # breakpoint()

def _drude_sigma_band(fit_freq_hz, sigma_dc, tau, covariance, param_uncertainties):
    """Propagate fit parameter uncertainties through the Drude model.

    Returns (band_real, band_imag) — 1-sigma half-widths at each frequency.
    Uses the full covariance matrix if available, otherwise independent uncertainties.
    """
    omega = 2.0 * np.pi * fit_freq_hz
    denom = 1.0 - 1j * omega * tau

    # Jacobians: d(sigma)/d(sigma_dc) and d(sigma)/d(tau)
    j_sigma_dc = 1.0 / denom
    j_tau = sigma_dc * 1j * omega / denom ** 2

    # Split into real/imag Jacobians (chain rule: d(Re[sigma])/dp = Re[d(sigma)/dp])
    jr_dc, jr_tau = j_sigma_dc.real, j_tau.real
    ji_dc, ji_tau = j_sigma_dc.imag, j_tau.imag

    if (covariance is not None
            and covariance.shape == (2, 2)
            and np.all(np.isfinite(covariance))):
        C = covariance
        var_real = jr_dc**2 * C[0, 0] + 2*jr_dc*jr_tau * C[0, 1] + jr_tau**2 * C[1, 1]
        var_imag = ji_dc**2 * C[0, 0] + 2*ji_dc*ji_tau * C[0, 1] + ji_tau**2 * C[1, 1]
    else:
        d_sigma_dc = param_uncertainties.get('sigma_dc', 0.0)
        d_tau = param_uncertainties.get('tau', 0.0)
        var_real = (jr_dc * d_sigma_dc)**2 + (jr_tau * d_tau)**2
        var_imag = (ji_dc * d_sigma_dc)**2 + (ji_tau * d_tau)**2

    return np.sqrt(np.abs(var_real)), np.sqrt(np.abs(var_imag))


def normalise_data(dataY, mask):
    """Normalise the data to the maximum of the real part for better comparison."""
    max_val = np.max(strip_nan(dataY[mask]))
    min_val = np.min(strip_nan(dataY[mask]))
    return (dataY - min_val) / (max_val - min_val)

def plot_fit_results(fit_dict: dict, config={}) -> None:
    '''Plot the fitted conductivity results for each result in the fit_dict."""'''
    fig, ax = plt.subplots()

    report = {}

    fit_range = config.get('fit_range', (0, 10))  # Default fit range in THz
    fit_freq = np.linspace(fit_range[0], fit_range[1], 1000)  # Frequency range for plotting

    for index, (filename, fit_data) in enumerate(fit_dict.items()):
        label_base = filename
        # fit_freq = fit_data.fit_freq
        # breakpoint()
        fit_params = fit_data.param_values
        sigma_dc = fit_params.get('sigma_dc', None)
        tau = fit_params.get('tau', None)
        r2 = fit_data.r_squared

        data_real = fit_data.fit_data.real
        data_imag = fit_data.fit_data.imag
        freq = fit_data.fit_freq
        fit_curve = fit_data.fit_curve
        curve_real = fit_curve.real
        curve_imag = fit_curve.imag

        d_sigma_dc = fit_data.param_uncertainties.get('sigma_dc', float('nan'))
        d_tau = fit_data.param_uncertainties.get('tau', float('nan'))

        report[filename] = {
            'sigma_dc': sigma_dc,
            'd_sigma_dc': d_sigma_dc,
            'tau_s': tau,
            'd_tau_s': d_tau,
            'tau_fs': tau * 1e15 if tau is not None else None,
            'd_tau_fs': d_tau * 1e15,
            'crossover_thz': 1.0 / (2.0 * np.pi * tau) * thz._HZ_TO_THZ if tau else None,
            'r_squared': r2,
        }

        plot_color = plt.get_cmap("tab10")(index)

        if config.get('normalise', False):
            # Normalise the data to the maximum of the real part for better comparison
            curve_real = normalise_data(curve_real, np.isfinite(curve_real))
            curve_imag = normalise_data(curve_imag, np.isfinite(curve_imag))

            data_real = normalise_data(data_real, np.isfinite(data_real))
            data_imag = normalise_data(data_imag, np.isfinite(data_imag))

        if sigma_dc is not None and tau is not None:
            # Generate the Drude fit curve using the fitted parameters
            # fit_freq is in THz — convert to Hz for the SI-unit Drude function
            fit_freq_hz = fit_freq * 1e12
            drude_fit_curve = drude_fit_dc(fit_freq_hz, sigma_dc, 1/tau)
            drude_real = drude_fit_curve.real
            drude_imag = drude_fit_curve.imag
            band_real, band_imag = _drude_sigma_band(
                fit_freq_hz, sigma_dc, tau,
                fit_data.covariance, fit_data.param_uncertainties
            )


        # breakpoint()

        if "n-type_trans" in filename.lower():
            # Plot as alpha 0.3 for transmission data, black for real and red for imaginary
            ax.scatter(freq * thz._HZ_TO_THZ, data_real, label=r"Transmission ($\sigma_\mathrm{real}$)", color='tab:blue', alpha=0.3)
            ax.scatter(freq * thz._HZ_TO_THZ, data_imag, label=r"Transmission ($\sigma_\mathrm{imag}$)", color='tab:green', alpha=0.3)

            ax.plot(fit_freq, drude_real, label=r"Transmission ($\sigma_\mathrm{real}$ Drude)", color='tab:blue', linestyle='solid', alpha=0.3)
            ax.fill_between(fit_freq, drude_real - band_real, drude_real + band_real, color='tab:blue', alpha=0.08)
            ax.plot(fit_freq, drude_imag, label=r"Transmission ($\sigma_\mathrm{imag}$ Drude)", color='tab:green', linestyle='dashed', alpha=0.3)
            # ax.fill_between(fit_freq, drude_imag - band_imag, drude_imag + band_imag, color='tab:green', alpha=0.08)
            crossover_thz = 1.0 / (2.0 * np.pi * tau) * thz._HZ_TO_THZ
            ax.axvline(crossover_thz, color='tab:green', linestyle='dotted', alpha=0.5)

        elif "n-type_1" in filename.lower():
            # Plot as alpha 0.3 for n-type data, blue for real and orange for imaginary
            # ax.scatter(freq * thz._HZ_TO_THZ, data_real, label=r"Reflection ($\sigma_\mathrm{real}$)", color='black', alpha=1)
            ax.scatter(freq * thz._HZ_TO_THZ, data_imag, label=r"Reflection ($\sigma_\mathrm{imag}$)", color='tab:red', alpha=1)

            ax.plot(fit_freq, drude_real, label=r"Reflection ($\sigma_\mathrm{real}$ Drude)", color='black', linestyle='solid', alpha=1, lw=1.5)
            ax.fill_between(fit_freq, drude_real - band_real, drude_real + band_real, color='black', alpha=0.08)
            ax.plot(fit_freq, drude_imag, label=r"Reflection ($\sigma_\mathrm{imag}$ Drude)", color='red', linestyle='dashed', alpha=1, lw=1.5)
            # ax.fill_between(fit_freq, drude_imag - band_imag, drude_imag + band_imag, color='red', alpha=0.08)

            crossover_thz = 1.0 / (2.0 * np.pi * tau) * thz._HZ_TO_THZ
            ax.axvline(crossover_thz, color='tab:red', linestyle='dotted', alpha=0.5)
                       #label=f"{filename}: $f_c$ = {crossover_thz:.2f} THz")
            # _plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, data_real, color=plot_color, label="{} (data sigma real)".format(filename))
        # ax.plot(freq * thz._HZ_TO_THZ, curve_real, label="{} (fit sig_r)".format(filename), color=plot_color, linestyle='solid')

        # _plot_with_snr_mask(ax, freq * thz._HZ_TO_THZ, data_imag, color=plot_color, label="{} (data sigma imag)".format(filename))
        # ax.plot(freq * thz._HZ_TO_THZ, curve_imag, label="{} (fit sig_i)".format(filename), color=plot_color, linestyle='dashed')

        ax.set_xlim(fit_range)
        ax.set_xlabel("Frequency (THz)")
        # plt.ylabel("Refractive Index / Extinction Coefficient")
        ax.set_ylabel("Conductivity (S/m)")
        # ax.set_title("Derived Conductivity")
        ax.legend()

    return report

def fit_all(bundle_dirs: list, *, quantity: str = "sigma", model: str = "drude_conductivity", fit_band_thz=(0.35, 1.6), save: bool = True):
    """Fit all bundles in the list and plot the results."""
    fit_dict = {}
    for bundle_dir in bundle_dirs:
        dataset = fit_bundle(bundle_dir, quantity=quantity, model=model, fit_band_thz=fit_band_thz, save=save)
        for filename, data_obj in dataset.data.items():
            fit_results = data_obj.processing_dict.get('fit_result')
            if fit_results is not None:
                fit_dict[filename] = fit_results
    plot_fit_results(fit_dict)
    plt.show()

if __name__ == "__main__":
    # Point this at a bundle written by any run_me_*.py (config['general']['save_session']).
    n_type = r"C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon_trans\ntype\ntype.thzbundle"
    # si_type = r"C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon_trans\insulating"
    refl = r"C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon\export\export.thzbundle"
    refl_2 = r'C:\Users\Samuel\Data\THz\calibration\silicon\silicon_p-pol_2\export\export.thzbundle'
    # refl_2 = r'C:\Users\Samuel\Data\THz\Sam\2026-07-10_sio2-E_testing\export\export.thzbundle'
    # fit_bundle(bundle_dir)
    dataset = fit_bundle(refl_2, quantity="sigma", model="drude_conductivity", fit_band_thz=(0.5, 3), save=True)

    config = {
        'normalise': False,
        'fit_range': (0, 3)
    }

# Group processing
    bundle_dirs = [n_type, refl, refl_2]
    # fit_all(bundle_dirs, quantity="sigma", model="drude_conductivity", fit_band_thz=(0.35, 1.6), save=True)

    n_type = session_bundle.load_session(n_type)
    # si_type = session_bundle.load_session()
    refl_data = session_bundle.load_session(refl)
    refl_2_data = session_bundle.load_session(refl_2)


    # for filename, data_obj in n_type.data.items():
    #     if 'reference' in filename.lower():
    #         continue  # skip the reference file
    #     fit_results = data_obj.processing_dict.get('fit_result')
    #     errors = data_obj.processing_dict.get('propagated_error')
    #     # print(f"file: {filename}, fit_results: {fit_results}, errors: {errors}")
    #     breakpoint()

    fit_dict = get_fit_dict(bundle_dirs)

    # breakpoint()
    report = plot_fit_results(fit_dict, config)
    plt.savefig(r'C:\Users\Samuel\Documents\Professional\Presentations\More-N\figures\conductivity_fit_results_error.png', dpi=600)
    print(report)
    plt.show()