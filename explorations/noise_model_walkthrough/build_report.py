"""Build the noise-model report: inject measured numbers and figures into the template.

Keeps the prose and the data in one artefact without letting them drift apart — every
number in the report is recomputed here from the same files the figures came from, so a
re-run after changing the estimator updates the text as well as the plots.

    python explorations/noise_model_walkthrough/build_report.py
"""
from __future__ import annotations

import base64
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root, for thz_core
sys.path.insert(0, HERE)                                    # the walkthrough module

# Importing the walkthrough re-runs it, which regenerates the figures from the current
# estimator before they are embedded. Slower, but the report can never ship figures and
# numbers that came from different versions of the code.
from noise_model_walkthrough import (  # noqa: E402
    FEATURED,
    OUT_DIR,
    SURVEY,
    drift_corrected_scatter,
    fit_noise_parameters,
    load_acquisition,
    spectral_noise_moments,
    tail_median_floor,
)

TEMPLATE = os.path.join(HERE, "report_template.html")
OUTPUT = os.path.join(HERE, "noise_report.html")
TEST_COUNT = 31


def data_uri(name: str) -> str:
    with open(os.path.join(OUT_DIR, name), "rb") as handle:
        return "data:image/png;base64," + base64.b64encode(handle.read()).decode("ascii")


def analyse(filepath: str) -> dict:
    entry = load_acquisition(filepath)
    drift = drift_corrected_scatter(entry["waveforms"], entry["dt"])
    parameters, _ = fit_noise_parameters(entry["waveforms"], entry["dt"], drift=drift)
    peak = float(np.max(np.abs(drift.mean_waveform)))
    fractions = parameters.term_contributions(drift.mean_waveform, entry["dt"])
    return dict(entry=entry, drift=drift, parameters=parameters, peak=peak,
                fractions=fractions)


print("Recomputing report numbers...")
featured = analyse(FEATURED)
entry, drift, parameters = featured["entry"], featured["drift"], featured["parameters"]
waveforms = entry["waveforms"]
n_scans, n_samples = waveforms.shape
dt = entry["dt"]

window = np.hanning(n_samples)
mean_spectrum = np.fft.rfft(drift.mean_waveform * window, n=n_samples)
moments = spectral_noise_moments(drift.sigma_t, window=window, n_fft=n_samples,
                                 spectrum=mean_spectrum, n_averaged=n_scans)
floor = tail_median_floor(mean_spectrum)
measured = float(np.median(moments.sigma_magnitude))

print("Analysing the survey set...")
survey_rows = []
alpha_fractions, beta_values = [], []
TAG = {"additive": ("tag-add", "additive"),
       "multiplicative": ("tag-mult", "multiplicative"),
       "jitter": ("tag-jitter", "jitter")}
for label, filepath in SURVEY.items():
    if not os.path.exists(filepath):
        continue
    result = analyse(filepath)
    alpha_percent = result["parameters"].sigma_alpha / result["peak"] * 100
    beta_percent = result["parameters"].sigma_beta * 100
    alpha_fractions.append(alpha_percent)
    beta_values.append(beta_percent)
    dominant = max(result["fractions"], key=result["fractions"].get)
    tag_class, tag_text = TAG[dominant]
    survey_rows.append(
        f'        <tr><td>{label}</td>'
        f'<td class="mono">{result["entry"]["waveforms"].shape[0]}</td>'
        f'<td class="mono">{alpha_percent:.3f}%</td>'
        f'<td class="mono">{beta_percent:.3f}%</td>'
        f'<td class="mono">{result["parameters"].sigma_tau * 1e15:.2f} fs</td>'
        f'<td class="mono">{result["drift"].drift_inflation:.2f}&times;</td>'
        f'<td><span class="tag {tag_class}">{tag_text}</span></td></tr>'
    )

time_ps = entry["time_seconds"] * 1e12
replacements = {
    "FILE": os.path.basename(FEATURED),
    "N_SCANS": str(n_scans),
    "MINUTES": f"{entry['elapsed_minutes'][-1]:.0f}",
    "DT": f"{dt * 1e15:.0f}",
    "SPAN": f"{time_ps[-1] - time_ps[0]:.1f}",
    "ALPHA": f"{parameters.sigma_alpha:.2e}",
    "ALPHA_PCT": f"{parameters.sigma_alpha / featured['peak'] * 100:.3f}",
    "BETA": f"{parameters.sigma_beta * 100:.2f}",
    "TAU": f"{parameters.sigma_tau * 1e15:.2f}",
    "SHARE_ADD": f"{featured['fractions']['additive'] * 100:.0f}",
    "SHARE_MULT": f"{featured['fractions']['multiplicative'] * 100:.0f}",
    "SHARE_JIT": f"{featured['fractions']['jitter'] * 100:.0f}",
    "PROFILE_ERR": f"{parameters.profile_error:.2f}",
    "PROFILE_EXPECTED": f"{1 / np.sqrt(2 * (n_scans - 1)):.2f}",
    "DRIFT_AMP": f"{np.ptp(drift.amplitudes) * 100:.1f}",
    "DRIFT_DELAY": f"{np.ptp(drift.delays) * 1e15:.0f}",
    "INFLATION": f"{drift.drift_inflation:.2f}",
    "FLOOR_RATIO": f"{floor / measured:.0f}",
    "ALPHA_MIN": f"{min(alpha_fractions):.3f}",
    "ALPHA_MAX": f"{max(alpha_fractions):.3f}",
    "BETA_MIN": f"{min(beta_values):.2f}",
    "BETA_MAX": f"{max(beta_values):.2f}",
    "SURVEY_ROWS": "\n".join(survey_rows),
    "N_TESTS": str(TEST_COUNT),
    "DATE": datetime.date.today().isoformat(),
}

figures = {
    "FIG01": "01_scans", "FIG02": "02_sigma_structure", "FIG03": "03_decomposition",
    "FIG04": "04_drift", "FIG05": "05_drift_removed", "FIG06": "06_spectrum",
    "FIG07": "07_averaging", "FIG08": "08_budget",
}
for key, stem in figures.items():
    replacements[f"{key}_LIGHT"] = data_uri(f"{stem}_light.png")
    replacements[f"{key}_DARK"] = data_uri(f"{stem}_dark.png")

with open(TEMPLATE) as handle:
    html = handle.read()
for key, value in replacements.items():
    html = html.replace("{{" + key + "}}", value)

remaining = [token for token in ("{{",) if token in html]
if remaining:
    start = html.index("{{")
    raise SystemExit(f"unfilled placeholder near: {html[start:start + 60]!r}")

with open(OUTPUT, "w") as handle:
    handle.write(html)

size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
print(f"Wrote {OUTPUT} ({size_mb:.2f} MB)")
for key in ("ALPHA_PCT", "BETA", "TAU", "SHARE_JIT", "INFLATION", "FLOOR_RATIO",
            "PROFILE_ERR"):
    print(f"  {key:<14} {replacements[key]}")
