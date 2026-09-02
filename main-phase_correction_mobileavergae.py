import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

# path = Path(r"E:\CNT-5")
# sample = path / "sample-S.dat"
# reference = path / "gold.dat"

# path = Path(r"C:\Users\IMDEA B26-01\Desktop\CNT-tape\CNT-3")
# sample = path / "sample-270.dat"
# reference = path / "gold-end.dat"

# path = Path(r"E:\CNT-2")
# sample = path / "sample-s.dat"
# reference = path / "silver.dat"

# path = Path(r"C:\Users\IMDEA B26-01\Desktop\CNT-tape\CNT-6")

path = Path(r"C:\Users\Samuel\Data\Denis\CNT-6")
path = Path(r"C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon\export")
# sample = path / "gold2.dat"
# reference = path / "Si2-90.dat"
reference = path / "reference_gold_A-100_S-pol.dat"
sample = path / "sample_silicon-2-N-type_A-100_S-pol.dat"
# sample = path / "sample_silicon-1-SI_A-100_S-pol.dat"

# ============================================================
# TIPO DE REFERENCIA
# ============================================================

reference_type = "gold"       # "gold", "metal", "silver" o "silicon"

n_reference_silicon = 3.4
k_reference_silicon = 0.0

# ============================================================
# RANGO DE FRECUENCIA
# ============================================================

condsign = 1

f_min = 0
f_max = 8.0

f_kk_min = 0.88
f_kk_max = 7.4

f_fit_min = 0
f_fit_max = 5.0

# ============================================================
# GEOMETRÍA
# ============================================================

theta_deg = 45.0
theta = np.deg2rad(theta_deg)

# ============================================================
# DESPLAZAMIENTO TEMPORAL MANUAL DE LA MUESTRA
# ============================================================

t_delay_ps = 0.0

# ============================================================
# CORRECCIÓN DE FASE
# ============================================================

phase_correction_mode = "afm"   # "none", "manual" o "afm"
manual_phase_delay_ps = 0.0

# ============================================================
# SUAVIZADO Y AJUSTE DE n,k
# ============================================================
#
# nk_for_optics_mode:
# "raw"    -> usa n_corr, k_corr directamente
# "smooth" -> usa n_corr, k_corr suavizados
# "fit"    -> usa el ajuste polinómico de n_corr, k_corr
#
# nk_fit_degree:
# 1 -> recta
# 2 -> polinomio cuadrático
# 3 -> cúbico
# etc.
# ============================================================

apply_nk_moving_average = True
nk_moving_average_points = 3

nk_for_optics_mode = "fit"       # "raw", "smooth" o "fit"

nk_fit_input = "smooth"          # "raw" o "smooth"
nk_fit_degree = 1                # 1 recta, 2 cuadrático, 3 cúbico...
nk_fit_only_inside_window = False

# ============================================================
# SUAVIZADO EXTRA DE SIGMA SOLO PARA PLOTEAR
# ============================================================

apply_sigma_plot_moving_average = True
sigma_plot_moving_average_points = 3

# ============================================================
# CONSTANTES
# ============================================================

c_um_ps = 299.792458
eps0 = 8.8541878128e-12
epsilon_inf = 0.0

# ============================================================
# FUNCIONES
# ============================================================

def linea_cero(ax):
    if ax.get_yscale() != "log":
        ax.axhline(
            0,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            zorder=0
        )


def moving_average_centered(y, window_points):
    y = np.asarray(y, dtype=float)

    if window_points < 2:
        return y.copy()

    if window_points % 2 == 0:
        window_points += 1

    pad = window_points // 2
    kernel = np.ones(window_points, dtype=float)

    finite = np.isfinite(y)
    y_clean = np.where(finite, y, 0.0)
    weights = finite.astype(float)

    y_pad = np.pad(y_clean, pad_width=pad, mode="edge")
    w_pad = np.pad(weights, pad_width=pad, mode="edge")

    numerator = np.convolve(y_pad, kernel, mode="valid")
    denominator = np.convolve(w_pad, kernel, mode="valid")

    with np.errstate(divide="ignore", invalid="ignore"):
        y_smooth = numerator / denominator

    y_smooth[denominator == 0] = np.nan

    return y_smooth


def fresnel_s_air_to_material(n_complex, theta_i):
    n0 = 1.0 + 0.0j

    with np.errstate(divide="ignore", invalid="ignore"):
        sin_theta_t = n0 * np.sin(theta_i) / n_complex
        cos_theta_t = np.sqrt(1.0 - sin_theta_t**2)

    if np.real(cos_theta_t) < 0:
        cos_theta_t = -cos_theta_t

    r_s_phys = (
        n0 * np.cos(theta_i) - n_complex * cos_theta_t
    ) / (
        n0 * np.cos(theta_i) + n_complex * cos_theta_t
    )

    return r_s_phys


def reference_calibration_factor_s_pol(reference_kind):
    ref = reference_kind.lower().strip()

    if ref in ["gold", "metal", "silver"]:
        factor = 1.0 + 0.0j
        r_ref_phys = -1.0 + 0.0j
        label = "Metal reference"

    elif ref == "silicon":
        n_ref_complex = n_reference_silicon + 1j * k_reference_silicon
        r_ref_phys = fresnel_s_air_to_material(n_ref_complex, theta)
        factor = -r_ref_phys
        label = (
            f"Silicon reference, "
            f"n={n_reference_silicon:g}, "
            f"k={k_reference_silicon:g}, "
            f"-r_s={np.real(factor):.5g}"
        )

    else:
        raise ValueError(
            "reference_type debe ser 'gold', 'metal', 'silver' o 'silicon'."
        )

    return factor, r_ref_phys, label


def refractive_index_s_pol_45(r_complex):
    with np.errstate(divide="ignore", invalid="ignore"):
        N_complex = np.sqrt(
            1.0 + (4.0 * r_complex * np.cos(theta)**2) / (1.0 - r_complex)**2
        )

    N_complex = np.where(np.real(N_complex) < 0, -N_complex, N_complex)

    return N_complex


def polynomial_fit(x, y, fit_mask, degree):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = fit_mask & np.isfinite(x) & np.isfinite(y)

    if degree < 0:
        raise ValueError("degree debe ser >= 0.")

    if np.sum(valid) < degree + 1:
        raise ValueError(
            f"No hay suficientes puntos válidos para ajuste polinómico de grado {degree}."
        )

    coeffs = np.polyfit(x[valid], y[valid], degree)
    poly = np.poly1d(coeffs)
    y_fit = poly(x)

    return coeffs, poly, y_fit


def poly_label(name, coeffs):
    degree = len(coeffs) - 1

    if degree == 0:
        return rf"{name} fit:" + "\n" + rf"$y = {coeffs[0]:.5g}$"

    terms = []

    for i, a in enumerate(coeffs):
        power = degree - i

        if not np.isfinite(a):
            continue

        sign = "+" if a >= 0 else "-"
        abs_a = abs(a)

        if power == 0:
            term = rf"{sign} {abs_a:.4g}"
        elif power == 1:
            term = rf"{sign} {abs_a:.4g} f"
        else:
            term = rf"{sign} {abs_a:.4g} f^{power}"

        terms.append(term)

    expression = " ".join(terms)

    if expression.startswith("+ "):
        expression = expression[2:]

    return rf"{name} fit:" + "\n" + rf"$y = {expression}$"


def compute_delta_m(omega_kk, phi_kk, r_abs_kk):
    delta_m = np.full_like(omega_kk, np.nan, dtype=float)

    ln_r = np.log(np.clip(r_abs_kk, 1e-30, None))

    for i, w in enumerate(omega_kk):
        denom = omega_kk**2 - w**2

        integrand = np.full_like(omega_kk, np.nan, dtype=float)
        valid = np.abs(denom) > 0

        integrand[valid] = omega_kk[valid] * phi_kk[valid] / denom[valid]

        valid_int = np.isfinite(integrand) & np.isfinite(omega_kk)

        if np.sum(valid_int) >= 2:
            pv_integral = np.trapezoid(integrand[valid_int], omega_kk[valid_int])
            delta_m[i] = (2.0 / np.pi) * pv_integral - ln_r[i]

    return delta_m


def optical_constants_from_N(N_complex, freq_THz, eps_inf):
    omega_SI = 2.0 * np.pi * freq_THz * 1e12

    epsilon_complex = N_complex**2
    sigma_complex = -1j * omega_SI * eps0 * (epsilon_complex - eps_inf)

    eps_real = np.real(epsilon_complex)
    eps_imag = np.imag(epsilon_complex)

    sigma_real = condsign * np.real(sigma_complex)
    sigma_imag = -np.imag(sigma_complex)

    eps_real[~np.isfinite(eps_real)] = np.nan
    eps_imag[~np.isfinite(eps_imag)] = np.nan
    sigma_real[~np.isfinite(sigma_real)] = np.nan
    sigma_imag[~np.isfinite(sigma_imag)] = np.nan

    return eps_real, eps_imag, sigma_real, sigma_imag


def make_complex_N_from_nk(n_array, k_array):
    return np.asarray(n_array, dtype=float) + 1j * np.asarray(k_array, dtype=float)


def save_valid_three_columns(output_path, data):
    valid = (
        np.isfinite(data[:, 0]) &
        np.isfinite(data[:, 1]) &
        np.isfinite(data[:, 2])
    )

    np.savetxt(
        output_path,
        data[valid],
        fmt="%.10e"
    )


# ============================================================
# CARGA DE DATOS
# ============================================================

t, E_sam_raw = np.loadtxt(sample, comments="%", unpack=True)
t_ref, E_ref_raw = np.loadtxt(reference, comments="%", unpack=True)

if len(t) != len(t_ref) or not np.allclose(t, t_ref):
    raise ValueError("Los ejes temporales de sample y reference no coinciden.")

# ============================================================
# DESPLAZAMIENTO TEMPORAL MANUAL DE LA MUESTRA
# ============================================================

E_sam = np.interp(t - t_delay_ps, t, E_sam_raw, left=0.0, right=0.0)
E_ref = E_ref_raw.copy()

# ============================================================
# FFT
# ============================================================

dt = np.mean(np.diff(t))
freq_full = np.fft.rfftfreq(len(t), dt)
omega_full = 2.0 * np.pi * freq_full

F_sam_full = np.fft.rfft(E_sam - E_sam.mean())
F_ref_full = np.fft.rfft(E_ref - E_ref.mean())

# ============================================================
# COEFICIENTE COMPLEJO DE REFLEXIÓN
# ============================================================

reference_factor, r_reference_phys, reference_label = reference_calibration_factor_s_pol(
    reference_type
)

with np.errstate(divide="ignore", invalid="ignore"):
    transfer_ratio_full = F_sam_full / F_ref_full
    r_full = transfer_ratio_full * reference_factor

print("\nReferencia:")
print(f"sample = {sample}")
print(f"reference = {reference}")
print(f"reference_type = {reference_type}")
print(f"reference_label = {reference_label}")
print(
    "r_reference_phys = "
    f"{np.real(r_reference_phys):.8g} "
    f"{np.imag(r_reference_phys):+.8g}j"
)
print(
    "reference_factor = -r_reference_phys = "
    f"{np.real(reference_factor):.8g} "
    f"{np.imag(reference_factor):+.8g}j"
)
print(f"|reference_factor| = {abs(reference_factor):.8g}")

r_abs_full = np.abs(r_full)

phi_wrapped_full = np.arctan2(np.imag(r_full), np.real(r_full))
phi_wrapped_full[~np.isfinite(phi_wrapped_full)] = 0.0

if freq_full[0] == 0:
    phi_wrapped_full[0] = 0.0

phi_full = np.unwrap(phi_wrapped_full)

R_full = r_abs_full**2

r_abs_full[~np.isfinite(r_abs_full)] = np.nan
R_full[~np.isfinite(R_full)] = np.nan

# ============================================================
# ÍNDICE DE REFRACCIÓN COMPLEJO SIN CORRECCIÓN
# ============================================================

N_full = refractive_index_s_pol_45(r_full)

n_full = np.real(N_full)
k_full = np.imag(N_full)

n_full[~np.isfinite(n_full)] = np.nan
k_full[~np.isfinite(k_full)] = np.nan

# ============================================================
# FILTRO EN FRECUENCIA
# ============================================================

mask = (freq_full >= f_min) & (freq_full <= f_max)

freq = freq_full[mask]
omega = omega_full[mask]

F_sam = F_sam_full[mask]
F_ref = F_ref_full[mask]

fft_sam = np.abs(F_sam)
fft_ref = np.abs(F_ref)

r = r_full[mask]
r_abs = r_abs_full[mask]
phi = phi_full[mask]
R = R_full[mask]

N = N_full[mask]
n = n_full[mask]
k = k_full[mask]

# ============================================================
# ANALYTICAL FITTING METHOD
# ============================================================

kk_mask = (
    (freq >= f_kk_min) &
    (freq <= f_kk_max) &
    np.isfinite(freq) &
    np.isfinite(omega) &
    np.isfinite(phi) &
    np.isfinite(r_abs) &
    (r_abs > 0)
)

freq_kk = freq[kk_mask]
omega_kk = omega[kk_mask]
phi_kk = phi[kk_mask]
r_abs_kk = r_abs[kk_mask]

delay_afm_ps = np.nan
l_um = np.nan
A_afm = np.nan
C_afm = np.nan
delta_m = np.array([])
delta_m_fit = np.array([])
X_afm = np.array([])

if len(freq_kk) >= 5:
    omega_end = omega_kk[-1]

    delta_m = compute_delta_m(omega_kk, phi_kk, r_abs_kk)

    with np.errstate(divide="ignore", invalid="ignore"):
        X_afm = omega_kk * np.log(
            np.abs((omega_end - omega_kk) / (omega_end + omega_kk))
        )

    afm_fit_mask = (
        np.isfinite(delta_m) &
        np.isfinite(X_afm) &
        (freq_kk > f_kk_min) &
        (freq_kk < f_kk_max)
    )

    if np.sum(afm_fit_mask) >= 2:
        A_afm, C_afm = np.polyfit(
            X_afm[afm_fit_mask],
            delta_m[afm_fit_mask],
            1
        )

        delta_m_fit = np.where(
            np.isfinite(X_afm),
            A_afm * X_afm + C_afm,
            np.nan
        )

        l_um = A_afm * np.pi * c_um_ps * np.cos(theta) / 2.0
        delay_afm_ps = 2.0 * l_um / (c_um_ps * np.cos(theta))

print("\nAnalytical Fitting Method:")
print(f"f_kk_min = {f_kk_min} THz")
print(f"f_kk_max = {f_kk_max} THz")
print(f"A = {A_afm:.6g}")
print(f"C = {C_afm:.6g}")
print(f"l = {l_um:.6g} µm")
print(f"delay AFM = {delay_afm_ps:.8g} ps = {delay_afm_ps * 1000:.6g} fs")

# ============================================================
# SELECCIÓN DE CORRECCIÓN DE FASE
# ============================================================

if phase_correction_mode == "none":
    selected_delay_ps = 0.0
    correction_label = "No phase correction"

elif phase_correction_mode == "manual":
    selected_delay_ps = manual_phase_delay_ps
    correction_label = f"Manual delay = {selected_delay_ps:.8f} ps"

elif phase_correction_mode == "afm":
    if not np.isfinite(delay_afm_ps):
        raise ValueError("AFM no pudo calcular un delay válido.")
    selected_delay_ps = delay_afm_ps
    correction_label = f"AFM delay = {selected_delay_ps:.8f} ps"

else:
    raise ValueError(
        "phase_correction_mode debe ser: 'none', 'manual' o 'afm'."
    )

print("\nCorrección de fase seleccionada:")
print(f"mode = {phase_correction_mode}")
print(correction_label)
print(f"selected_delay = {selected_delay_ps:.8f} ps = {selected_delay_ps * 1000:.4f} fs")

# ============================================================
# FASE CORREGIDA Y r CORREGIDO
# ============================================================

phi_corr_full = phi_full - omega_full * selected_delay_ps
r_corr_full = r_abs_full * np.exp(1j * phi_corr_full)

N_corr_full = refractive_index_s_pol_45(r_corr_full)

n_corr_full = np.real(N_corr_full)
k_corr_full = np.imag(N_corr_full)

n_corr_full[~np.isfinite(n_corr_full)] = np.nan
k_corr_full[~np.isfinite(k_corr_full)] = np.nan

phi_corr = phi_corr_full[mask]
r_corr = r_corr_full[mask]
N_corr = N_corr_full[mask]
n_corr = n_corr_full[mask]
k_corr = k_corr_full[mask]

# ============================================================
# SUAVIZADO DE n,k
# ============================================================

if apply_nk_moving_average:
    n_corr_smooth = moving_average_centered(n_corr, nk_moving_average_points)
    k_corr_smooth = moving_average_centered(k_corr, nk_moving_average_points)
else:
    n_corr_smooth = n_corr.copy()
    k_corr_smooth = k_corr.copy()

N_corr_smooth = make_complex_N_from_nk(n_corr_smooth, k_corr_smooth)

# ============================================================
# AJUSTE POLINÓMICO DE n,k
# ============================================================

fit_mask = (freq >= f_fit_min) & (freq <= f_fit_max)

if nk_fit_input.lower().strip() == "raw":
    n_for_fit = n_corr.copy()
    k_for_fit = k_corr.copy()
    nk_fit_input_label = "raw corrected n,k"

elif nk_fit_input.lower().strip() == "smooth":
    n_for_fit = n_corr_smooth.copy()
    k_for_fit = k_corr_smooth.copy()
    nk_fit_input_label = f"smoothed corrected n,k MA{nk_moving_average_points}"

else:
    raise ValueError("nk_fit_input debe ser 'raw' o 'smooth'.")

n_fit_coeffs, n_fit_poly, n_fit_curve = polynomial_fit(
    freq,
    n_for_fit,
    fit_mask,
    nk_fit_degree
)

k_fit_coeffs, k_fit_poly, k_fit_curve = polynomial_fit(
    freq,
    k_for_fit,
    fit_mask,
    nk_fit_degree
)

if nk_fit_only_inside_window:
    outside_fit_window = ~fit_mask
    n_fit_curve[outside_fit_window] = np.nan
    k_fit_curve[outside_fit_window] = np.nan

N_corr_fit = make_complex_N_from_nk(n_fit_curve, k_fit_curve)

print("\nSuavizado y ajuste de n,k:")
print(f"apply_nk_moving_average = {apply_nk_moving_average}")
print(f"nk_moving_average_points = {nk_moving_average_points}")
print(f"nk_fit_input = {nk_fit_input} ({nk_fit_input_label})")
print(f"nk_fit_degree = {nk_fit_degree}")
print(f"nk_fit_only_inside_window = {nk_fit_only_inside_window}")
print(f"f_fit_min = {f_fit_min} THz")
print(f"f_fit_max = {f_fit_max} THz")
print("n fit coefficients, high-to-low order:")
print(n_fit_coeffs)
print("k fit coefficients, high-to-low order:")
print(k_fit_coeffs)

# ============================================================
# SELECCIÓN DE n,k PARA EPSILON Y SIGMA
# ============================================================

mode = nk_for_optics_mode.lower().strip()

if mode == "raw":
    N_for_optics = N_corr
    n_for_optics = n_corr
    k_for_optics = k_corr
    optics_source_label = "corrected raw n,k"

elif mode == "smooth":
    N_for_optics = N_corr_smooth
    n_for_optics = n_corr_smooth
    k_for_optics = k_corr_smooth
    optics_source_label = f"corrected smoothed n,k MA{nk_moving_average_points}"

elif mode == "fit":
    N_for_optics = N_corr_fit
    n_for_optics = n_fit_curve
    k_for_optics = k_fit_curve
    optics_source_label = f"polynomial fit n,k degree {nk_fit_degree}"

else:
    raise ValueError("nk_for_optics_mode debe ser 'raw', 'smooth' o 'fit'.")

print("\nFuente de n,k usada para epsilon y sigma:")
print(f"nk_for_optics_mode = {nk_for_optics_mode}")
print(f"optics_source_label = {optics_source_label}")

# ============================================================
# PERMITIVIDAD Y CONDUCTIVIDAD
# ============================================================

eps_real_meas, eps_imag_meas, sigma_real_meas, sigma_imag_meas = optical_constants_from_N(
    N,
    freq,
    epsilon_inf
)

eps_real_corr_raw, eps_imag_corr_raw, sigma_real_corr_raw, sigma_imag_corr_raw = optical_constants_from_N(
    N_corr,
    freq,
    epsilon_inf
)

eps_real_corr_smooth, eps_imag_corr_smooth, sigma_real_corr_smooth, sigma_imag_corr_smooth = optical_constants_from_N(
    N_corr_smooth,
    freq,
    epsilon_inf
)

eps_real_corr_fit, eps_imag_corr_fit, sigma_real_corr_fit, sigma_imag_corr_fit = optical_constants_from_N(
    N_corr_fit,
    freq,
    epsilon_inf
)

eps_real_selected, eps_imag_selected, sigma_real_selected, sigma_imag_selected = optical_constants_from_N(
    N_for_optics,
    freq,
    epsilon_inf
)

if apply_sigma_plot_moving_average:
    sigma_real_selected_plot_smooth = moving_average_centered(
        sigma_real_selected,
        sigma_plot_moving_average_points
    )
    sigma_imag_selected_plot_smooth = moving_average_centered(
        sigma_imag_selected,
        sigma_plot_moving_average_points
    )
else:
    sigma_real_selected_plot_smooth = sigma_real_selected.copy()
    sigma_imag_selected_plot_smooth = sigma_imag_selected.copy()

# ============================================================
# GUARDAR ARCHIVOS
# ============================================================

sigma_raw_output = path / f"sigma_{sample.stem}_raw_from_nk_raw.dat"
sigma_smooth_output = path / f"sigma_{sample.stem}_from_nk_smooth_ma{nk_moving_average_points}.dat"
sigma_fit_output = path / f"sigma_{sample.stem}_from_nk_poly{nk_fit_degree}.dat"
sigma_main_output = path / f"sigma_{sample.stem}.dat"

nk_selected_output = path / f"nk_used_for_sigma_{sample.stem}.dat"
eps_selected_output = path / f"epsilon_{sample.stem}.dat"

sigma_raw_data = np.column_stack([
    freq,
    sigma_real_corr_raw,
    sigma_imag_corr_raw
])

sigma_smooth_data = np.column_stack([
    freq,
    sigma_real_corr_smooth,
    sigma_imag_corr_smooth
])

sigma_fit_data = np.column_stack([
    freq,
    sigma_real_corr_fit,
    sigma_imag_corr_fit
])

sigma_selected_data = np.column_stack([
    freq,
    sigma_real_selected,
    sigma_imag_selected
])

nk_selected_data = np.column_stack([
    freq,
    n_for_optics,
    k_for_optics
])

eps_selected_data = np.column_stack([
    freq,
    eps_real_selected,
    eps_imag_selected
])

save_valid_three_columns(sigma_raw_output, sigma_raw_data)
save_valid_three_columns(sigma_smooth_output, sigma_smooth_data)
save_valid_three_columns(sigma_fit_output, sigma_fit_data)
save_valid_three_columns(sigma_main_output, sigma_selected_data)
save_valid_three_columns(nk_selected_output, nk_selected_data)
save_valid_three_columns(eps_selected_output, eps_selected_data)

print("\nArchivos guardados:")
print(f"Sigma raw n,k:        {sigma_raw_output}")
print(f"Sigma smooth n,k:     {sigma_smooth_output}")
print(f"Sigma polynomial n,k: {sigma_fit_output}")
print(f"Sigma principal:      {sigma_main_output}")
print(f"n,k usados:           {nk_selected_output}")
print(f"epsilon principal:    {eps_selected_output}")

# ============================================================
# AJUSTES PARA GRAFICAR |r| Y FASE
# ============================================================

r_fit_coeffs, r_fit_poly, r_fit_line = polynomial_fit(
    freq,
    r_abs,
    fit_mask,
    1
)

phi_fit_coeffs, phi_fit_poly, phi_fit_line = polynomial_fit(
    freq,
    phi_corr,
    fit_mask,
    1
)

# ============================================================
# GRÁFICAS: TIEMPO Y FFT
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(t, E_ref, label=reference_label)
ax[0].plot(t, E_sam, label=sample.stem)
linea_cero(ax[0])
ax[0].set_xlabel("Time (ps)")
ax[0].set_ylabel("E. Field (arb. units)")
ax[0].set_title("Time traces")
ax[0].legend()

ax[1].semilogy(freq, fft_ref, label=reference_label)
ax[1].semilogy(freq, fft_sam, label=sample.stem)
ax[1].set_xlabel("Frequency (THz)")
ax[1].set_ylabel("FFT amplitude (arb. units)")
ax[1].set_title("FFT")
ax[1].set_ylim(1e-2, 10)
ax[1].legend()

plt.tight_layout()

# ============================================================
# FIGURA ANALYTICAL FITTING METHOD
# ============================================================

if len(freq_kk) >= 5 and len(delta_m) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(freq_kk, delta_m, label=r"$\Delta_m$ data")

    if len(delta_m_fit) == len(freq_kk):
        ax.plot(freq_kk, delta_m_fit, "k--", label="Analytical fit")

    ax.axvspan(f_kk_min, f_kk_max, alpha=0.15)
    linea_cero(ax)

    ax.text(
        0.03,
        0.06,
        rf"$l = {l_um:.4g}\ \mu m$" + "\n" +
        rf"$\Delta t_{{AFM}} = {delay_afm_ps * 1000:.4g}\ fs$",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85)
    )

    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(r"$\Delta_m$")
    ax.set_title("Analytical Fitting Method")
    ax.legend()

    plt.tight_layout()

# ============================================================
# |r| Y FASE
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(freq, r_abs, label=sample.stem)
ax[0].plot(
    freq,
    r_fit_line,
    "k--",
    label=rf"Linear fit {f_fit_min}-{f_fit_max} THz"
)
ax[0].axvspan(f_fit_min, f_fit_max, alpha=0.15)
linea_cero(ax[0])
ax[0].text(
    0.03,
    0.06,
    poly_label(r"$|\tilde r|$", r_fit_coeffs),
    transform=ax[0].transAxes,
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.85)
)
ax[0].set_xlabel("Frequency (THz)")
ax[0].set_ylabel(r"$|\tilde r|$")
ax[0].set_title("Reflection coefficient amplitude")
ax[0].set_xlim(f_min, f_max)
ax[0].set_ylim(0, 1.2)
ax[0].legend()

ax[1].plot(freq, phi, alpha=0.45, label="Measured phase")
ax[1].plot(freq, phi_corr, label="Corrected phase")
ax[1].plot(
    freq,
    phi_fit_line,
    "k--",
    label=rf"Linear fit {f_fit_min}-{f_fit_max} THz"
)
ax[1].axvspan(f_fit_min, f_fit_max, alpha=0.15)
linea_cero(ax[1])
ax[1].text(
    0.03,
    0.06,
    poly_label(r"$\varphi_{\mathrm{corr}}$", phi_fit_coeffs),
    transform=ax[1].transAxes,
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.85)
)
ax[1].set_xlabel("Frequency (THz)")
ax[1].set_ylabel(r"$\varphi$ (rad)")
ax[1].set_title("Reflection coefficient phase")
ax[1].set_xlim(f_min, f_max)
ax[1].set_ylim(-1, 1)
ax[1].legend()

plt.tight_layout()

# ============================================================
# n Y k
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(freq, n, alpha=0.25, label="n measured")
ax[0].plot(freq, n_corr, alpha=0.45, label="n corrected raw")
ax[0].plot(
    freq,
    n_corr_smooth,
    linewidth=2.0,
    label=rf"n corrected MA{nk_moving_average_points}"
)
ax[0].plot(
    freq,
    n_fit_curve,
    "k--",
    linewidth=2.0,
    label=rf"n poly degree {nk_fit_degree}"
)
ax[0].plot(
    freq,
    n_for_optics,
    linewidth=3.0,
    alpha=0.65,
    label="n used for optics"
)
ax[0].axvspan(f_fit_min, f_fit_max, alpha=0.15)
linea_cero(ax[0])
ax[0].text(
    0.03,
    0.06,
    poly_label(r"$n$", n_fit_coeffs),
    transform=ax[0].transAxes,
    fontsize=8,
    bbox=dict(facecolor="white", alpha=0.85)
)
ax[0].set_xlabel("Frequency (THz)")
ax[0].set_ylabel("n")
ax[0].set_title("Refractive index")
ax[0].set_xlim(f_min, f_max)
ax[0].legend()

ax[1].plot(freq, k, alpha=0.25, label="k measured")
ax[1].plot(freq, k_corr, alpha=0.45, label="k corrected raw")
ax[1].plot(
    freq,
    k_corr_smooth,
    linewidth=2.0,
    label=rf"k corrected MA{nk_moving_average_points}"
)
ax[1].plot(
    freq,
    k_fit_curve,
    "k--",
    linewidth=2.0,
    label=rf"k poly degree {nk_fit_degree}"
)
ax[1].plot(
    freq,
    k_for_optics,
    linewidth=3.0,
    alpha=0.65,
    label="k used for optics"
)
ax[1].axvspan(f_fit_min, f_fit_max, alpha=0.15)
linea_cero(ax[1])
ax[1].text(
    0.03,
    0.06,
    poly_label(r"$k$", k_fit_coeffs),
    transform=ax[1].transAxes,
    fontsize=8,
    bbox=dict(facecolor="white", alpha=0.85)
)
ax[1].set_xlabel("Frequency (THz)")
ax[1].set_ylabel("k")
ax[1].set_title("Extinction coefficient")
ax[1].set_xlim(f_min, f_max)
ax[1].legend()

plt.tight_layout()

# ============================================================
# n y k superpuestos
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(freq, n_corr, alpha=0.30, label="n corrected raw")
ax.plot(freq, k_corr, alpha=0.30, label="k corrected raw")

ax.plot(freq, n_corr_smooth, alpha=0.75, label=rf"n MA{nk_moving_average_points}")
ax.plot(freq, k_corr_smooth, alpha=0.75, label=rf"k MA{nk_moving_average_points}")

ax.plot(freq, n_fit_curve, "k--", linewidth=2.0, label=rf"n poly degree {nk_fit_degree}")
ax.plot(freq, k_fit_curve, "k:", linewidth=2.0, label=rf"k poly degree {nk_fit_degree}")

ax.plot(freq, n_for_optics, linewidth=3.0, alpha=0.65, label="n used for optics")
ax.plot(freq, k_for_optics, linewidth=3.0, alpha=0.65, label="k used for optics")

ax.axvspan(f_fit_min, f_fit_max, alpha=0.15)
linea_cero(ax)

ax.text(
    0.03,
    0.06,
    f"Optics source:\n{optics_source_label}",
    transform=ax.transAxes,
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.85)
)

ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("n, k")
ax.set_title("Refractive index and extinction coefficient")
ax.set_xlim(f_min, f_max)
ax.legend()

plt.tight_layout()

# ============================================================
# PERMITIVIDAD REAL E IMAGINARIA
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(freq, eps_real_corr_raw, alpha=0.30, label=r"$\epsilon_1$ from raw n,k")
ax[0].plot(freq, eps_real_corr_smooth, alpha=0.50, label=r"$\epsilon_1$ from smooth n,k")
ax[0].plot(freq, eps_real_corr_fit, "k--", alpha=0.80, label=r"$\epsilon_1$ from poly n,k")
ax[0].plot(
    freq,
    eps_real_selected,
    linewidth=3.0,
    alpha=0.65,
    label=r"$\epsilon_1$ selected"
)
ax[0].axvspan(f_fit_min, f_fit_max, alpha=0.15)
linea_cero(ax[0])
ax[0].set_xlabel("Frequency (THz)")
ax[0].set_ylabel(r"$\epsilon_1$")
ax[0].set_title("Real permittivity")
ax[0].set_xlim(f_min, f_max)
ax[0].legend()

ax[1].plot(freq, eps_imag_corr_raw, alpha=0.30, label=r"$\epsilon_2$ from raw n,k")
ax[1].plot(freq, eps_imag_corr_smooth, alpha=0.50, label=r"$\epsilon_2$ from smooth n,k")
ax[1].plot(freq, eps_imag_corr_fit, "k--", alpha=0.80, label=r"$\epsilon_2$ from poly n,k")
ax[1].plot(
    freq,
    eps_imag_selected,
    linewidth=3.0,
    alpha=0.65,
    label=r"$\epsilon_2$ selected"
)
ax[1].axvspan(f_fit_min, f_fit_max, alpha=0.15)
linea_cero(ax[1])
ax[1].set_xlabel("Frequency (THz)")
ax[1].set_ylabel(r"$\epsilon_2$")
ax[1].set_title("Imaginary permittivity")
ax[1].set_xlim(f_min, f_max)
ax[1].legend()

plt.tight_layout()

# ============================================================
# CONDUCTIVIDAD REAL E IMAGINARIA
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(9, 5))

ax.plot(
    freq,
    sigma_real_corr_raw,
    alpha=0.25,
    label=r"$\sigma_1$ from raw n,k"
)

ax.plot(
    freq,
    sigma_imag_corr_raw,
    alpha=0.25,
    label=r"$\sigma_2$ from raw n,k"
)

ax.plot(
    freq,
    sigma_real_selected,
    linewidth=2.5,
    label=r"$\sigma_1$ selected"
)

ax.plot(
    freq,
    sigma_imag_selected,
    linewidth=2.5,
    label=r"$\sigma_2$ selected"
)

if apply_sigma_plot_moving_average:
    ax.plot(
        freq,
        sigma_real_selected_plot_smooth,
        "k--",
        linewidth=2.0,
        alpha=0.75,
        label=rf"$\sigma_1$ selected MA{sigma_plot_moving_average_points}"
    )

    ax.plot(
        freq,
        sigma_imag_selected_plot_smooth,
        "k:",
        linewidth=2.0,
        alpha=0.75,
        label=rf"$\sigma_2$ selected MA{sigma_plot_moving_average_points}"
    )

linea_cero(ax)
ax.axvspan(f_fit_min, f_fit_max, alpha=0.15)
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel(r"Conductivity (S/m)")
ax.set_title(f"Complex conductivity from {optics_source_label}")
ax.set_xlim(f_min, f_max)
ax.legend()

plt.tight_layout()

# ============================================================
# CONDUCTIVIDAD: SOLO PARTE REAL
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.plot(
    freq,
    sigma_real_corr_raw,
    alpha=0.30,
    label=r"$\sigma_1$ from raw n,k"
)

ax.plot(
    freq,
    sigma_real_corr_smooth,
    alpha=0.50,
    label=r"$\sigma_1$ from smooth n,k"
)

ax.plot(
    freq,
    sigma_real_corr_fit,
    "k--",
    linewidth=2.0,
    label=r"$\sigma_1$ from poly n,k"
)

ax.plot(
    freq,
    sigma_real_selected,
    linewidth=3.0,
    alpha=0.70,
    label=r"$\sigma_1$ selected"
)

linea_cero(ax)
ax.axvspan(f_fit_min, f_fit_max, alpha=0.15)
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel(r"$\sigma_1$ (S/m)")
ax.set_title("Real conductivity comparison")
ax.set_xlim(f_min, f_max)
ax.legend()

plt.tight_layout()

plt.show()