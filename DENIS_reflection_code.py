# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:03:15 2026

@author: Denís Paredes
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.signal.windows import hann, hamming, flattop, boxcar, kaiser, tukey


# =========================
# Parámetros
# =========================
filepath = Path(r'C:\Users\Samuel\Data\THz\CNTs\CNT-5')
reference = 'gold.dat'
sample = 'sample-P.dat'

n_base = 10

f_min = 0 # THz
f_max =  7  # THz

theta_deg = 45  # incidencia en grados

# Valor inicial de t_delay
# Positivo desplaza la muestra hacia tiempos más largos
t_delay =  0.01# ps

# Error máximo permitido en la corrección estimada
t_delay_error = 0.00000000001  # ps

# Número máximo de iteraciones
max_iter = 0

# Rango para comprobar la pendiente de fase
f_fit_min = 1  # THz
f_fit_max = 5# THz


# =========================
# Center padding
# =========================
usar_centerpad = True

# Aumenta la longitud temporal después de centrar el pico.
# 1.0 = misma longitud original
# 2.0 = doble longitud, añadiendo ceros a ambos lados
centerpad_desired_length_factor = 1

# Quitar offset antes de centrar el pico
centerpad_quitar_offset = True


# =========================
# Windowing
# =========================
usar_windowing = True

# Opciones:
# "boxcar"  -> sin ventana
# "hann"
# "hamming"
# "flattop"
# "kaiser"
# "tukey"
window_type = "hann"

# Parámetros específicos
kaiser_beta = 140
tukey_alpha = 0.25

# Normalización de ventana
normalizar_window = True


# =========================
# Leer archivos
# =========================
def leer_pulso(filename):
    data = np.genfromtxt(
        filename,
        usecols=(0, 1),
        invalid_raise=False,
        autostrip=True
    )

    data = np.atleast_2d(data)
    data = data[np.isfinite(data).all(axis=1)]

    t = data[:, 0]
    E = data[:, 1]

    idx = np.argsort(t)
    return t[idx], E[idx]


# =========================
# Center padding
# =========================
def centerpad_signal(
    t,
    E,
    n_base=10,
    desired_length_factor=1.0,
    quitar_offset=True
):
    """
    Centra el pico THz en la mitad del array usando padding con ceros.

    Esta función está adaptada del código centerpad basado en DataFrames,
    pero aquí trabaja directamente con arrays numpy.

    Parámetros
    ----------
    t : array
        Tiempo en ps.
    E : array
        Campo THz.
    n_base : int
        Número de puntos iniciales usados para calcular el offset.
    desired_length_factor : float
        Factor para aumentar la longitud final.
        1.0 mantiene la longitud original.
        2.0 duplica la longitud añadiendo ceros a ambos lados.
    quitar_offset : bool
        Si True, resta el promedio de los primeros n_base puntos.

    Devuelve
    --------
    t_padded, E_padded
    """

    t = np.asarray(t, dtype=float)
    E = np.asarray(E, dtype=float)

    if len(t) != len(E):
        raise ValueError("t y E deben tener la misma longitud.")

    if len(t) < 3:
        raise ValueError("La señal es demasiado corta para aplicar centerpad.")

    idx = np.argsort(t)
    t = t[idx]
    E = E[idx]

    if quitar_offset:
        n_offset = min(n_base, len(E))
        E = E - np.average(E[:n_offset])

    # Índice del pico usando valor absoluto
    peak_index = np.argmax(np.abs(E))

    # Número de ceros necesarios para llevar el pico al centro del array
    num_zeros_to_add = len(E) // 2 - peak_index

    if num_zeros_to_add < 0:
        num_zeros_to_add = abs(num_zeros_to_add)

        # Añade ceros al final y elimina puntos del principio
        E_padded = np.pad(
            E,
            (0, num_zeros_to_add),
            mode="constant"
        )

        t_padded = np.pad(
            t,
            (0, num_zeros_to_add),
            mode="reflect",
            reflect_type="odd"
        )

        E_padded = E_padded[num_zeros_to_add:]
        t_padded = t_padded[num_zeros_to_add:]

    elif num_zeros_to_add > 0:
        num_zeros_to_add = abs(num_zeros_to_add)

        # Añade ceros al principio y elimina puntos del final
        E_padded = np.pad(
            E,
            (num_zeros_to_add, 0),
            mode="constant"
        )

        t_padded = np.pad(
            t,
            (num_zeros_to_add, 0),
            mode="reflect",
            reflect_type="odd"
        )

        E_padded = E_padded[:-num_zeros_to_add]
        t_padded = t_padded[:-num_zeros_to_add]

    else:
        E_padded = E.copy()
        t_padded = t.copy()

    # Aumentar longitud total si desired_length_factor > 1
    current_length = len(E_padded)
    desired_length = int(round(desired_length_factor * current_length))

    if desired_length > current_length:
        total_zeros = desired_length - current_length
        zeros_left = total_zeros // 2
        zeros_right = total_zeros - zeros_left

        E_padded = np.pad(
            E_padded,
            (zeros_left, zeros_right),
            mode="constant"
        )

        t_padded = np.pad(
            t_padded,
            (zeros_left, zeros_right),
            mode="reflect",
            reflect_type="odd"
        )

    return t_padded, E_padded


# =========================
# Windowing
# =========================
def crear_window(
    N,
    tipo="flattop",
    kaiser_beta=1.5,
    tukey_alpha=1.25,
    normalizar=True
):
    """
    Crea una ventana temporal para aplicar antes de la FFT.
    """

    tipo = tipo.lower()

    if tipo == "boxcar":
        w = boxcar(N)

    elif tipo == "hann":
        w = hann(N, sym=False)

    elif tipo == "hamming":
        w = hamming(N, sym=False)

    elif tipo == "flattop":
        w = flattop(N, sym=False)

    elif tipo == "kaiser":
        w = kaiser(N, beta=kaiser_beta, sym=False)

    elif tipo == "tukey":
        w = tukey(N, alpha=tukey_alpha, sym=False)

    else:
        raise ValueError(
            f"window_type no reconocido: {tipo}. "
            "Usa 'boxcar', 'hann', 'hamming', 'flattop', 'kaiser' o 'tukey'."
        )

    if normalizar:
        mean_w = np.mean(w)
        if mean_w != 0:
            w = w / mean_w

    return w


def aplicar_windowing(E, w):
    """
    Aplica una ventana temporal a una señal 1D.
    """
    return E * w


# =========================
# Cálculo principal con delay
# =========================
def calcular_con_delay(t_delay_actual):
    """
    Aplica un t_delay, calcula FFT, r, R, fase y ajuste lineal.
    Devuelve todos los resultados necesarios.
    """

    # Desplazar muestra artificialmente
    E_sam_shifted = np.interp(
        t_common,
        t_common + t_delay_actual,
        E_sam_common_original,
        left=base_sam,
        right=base_sam
    )

    # Quitar offset antes de la FFT
    E_ref_fft_input = E_ref_common - np.mean(E_ref_common[:n_base])
    E_sam_fft_input = E_sam_shifted - np.mean(E_sam_shifted[:n_base])

    N_points = len(t_common)
    dt_fft = np.median(np.diff(t_common))

    # =========================
    # Aplicar windowing
    # =========================
    if usar_windowing:
        w_fft = crear_window(
            N_points,
            tipo=window_type,
            kaiser_beta=kaiser_beta,
            tukey_alpha=tukey_alpha,
            normalizar=normalizar_window
        )

        E_ref_fft_input = aplicar_windowing(E_ref_fft_input, w_fft)
        E_sam_fft_input = aplicar_windowing(E_sam_fft_input, w_fft)

    else:
        w_fft = boxcar(N_points)

    # FFT
    FFT_ref = np.fft.rfft(E_ref_fft_input)
    FFT_sam = np.fft.rfft(E_sam_fft_input)

    # Si dt está en ps, f queda en THz
    f = np.fft.rfftfreq(N_points, d=dt_fft)

    amp_ref = np.abs(FFT_ref)
    amp_sam = np.abs(FFT_sam)

    # Reflectividad
    threshold = 1e-4 * np.max(amp_ref)
    valid = amp_ref > threshold

    r_sample = np.full_like(FFT_ref, np.nan + 1j*np.nan, dtype=complex)

    # Referencia oro: r_oro ≈ -1
    r_sample[valid] = -FFT_sam[valid] / FFT_ref[valid]

    R = np.abs(r_sample) ** 2

    # Fase
    phase_r = np.full_like(f, np.nan, dtype=float)
    phase_r[valid] = np.unwrap(np.angle(r_sample[valid]))

    # Diagnóstico simple de fase
    mask_phase = (
        valid
        & np.isfinite(phase_r)
        & (f >= f_fit_min)
        & (f <= f_fit_max)
    )

    if np.sum(mask_phase) > 2:
        coef = np.polyfit(f[mask_phase], phase_r[mask_phase], 1)

        slope_phase = coef[0]   # rad / THz
        offset_phase = coef[1]

        t_delay_correction = slope_phase / (2 * np.pi)
        t_delay_suggested = t_delay_actual + t_delay_correction

    else:
        slope_phase = np.nan
        offset_phase = np.nan
        t_delay_correction = np.nan
        t_delay_suggested = t_delay_actual

    return {
        "t_delay": t_delay_actual,
        "t_delay_correction": t_delay_correction,
        "t_delay_suggested": t_delay_suggested,
        "slope_phase": slope_phase,
        "offset_phase": offset_phase,

        "E_sam_common": E_sam_shifted,

        "E_ref_fft_input": E_ref_fft_input,
        "E_sam_fft_input": E_sam_fft_input,
        "window_fft": w_fft,

        "FFT_ref": FFT_ref,
        "FFT_sam": FFT_sam,
        "f": f,
        "amp_ref": amp_ref,
        "amp_sam": amp_sam,
        "valid": valid,
        "r_sample": r_sample,
        "R": R,
        "phase_r": phase_r,
        "mask_phase": mask_phase
    }


# =========================
# Leer pulsos
# =========================
t_ref, E_ref = leer_pulso(filepath / reference)
t_sam, E_sam = leer_pulso(filepath / sample)


# =========================
# Aplicar centerpad
# =========================
if usar_centerpad:
    t_ref, E_ref = centerpad_signal(
        t_ref,
        E_ref,
        n_base=n_base,
        desired_length_factor=centerpad_desired_length_factor,
        quitar_offset=centerpad_quitar_offset
    )

    t_sam, E_sam = centerpad_signal(
        t_sam,
        E_sam,
        n_base=n_base,
        desired_length_factor=centerpad_desired_length_factor,
        quitar_offset=centerpad_quitar_offset
    )


# =========================
# Igualar ventanas temporales
# =========================
dt = min(
    np.median(np.diff(t_ref)),
    np.median(np.diff(t_sam))
)

t_common = np.arange(
    min(t_ref.min(), t_sam.min()),
    max(t_ref.max(), t_sam.max()) + dt,
    dt
)

base_ref = np.mean(E_ref[:n_base])
base_sam = np.mean(E_sam[:n_base])

E_ref_common = np.interp(
    t_common,
    t_ref,
    E_ref,
    left=base_ref,
    right=base_ref
)

E_sam_common_original = np.interp(
    t_common,
    t_sam,
    E_sam,
    left=base_sam,
    right=base_sam
)


# =========================
# Corrección iterativa de t_delay
# =========================
print("===== Corrección iterativa de fase =====")

t_delay_actual = t_delay

for i in range(max_iter):

    res = calcular_con_delay(t_delay_actual)

    t_delay_correction = res["t_delay_correction"]
    t_delay_suggested = res["t_delay_suggested"]
    slope_phase = res["slope_phase"]

    print(f"Iteración {i + 1}")
    print(f"  t_delay actual       = {t_delay_actual:.12f} ps")
    print(f"  pendiente fase       = {slope_phase:.12f} rad/THz")
    print(f"  corrección estimada  = {t_delay_correction:.12f} ps")
    print(f"  t_delay sugerido     = {t_delay_suggested:.12f} ps")

    if not np.isfinite(t_delay_correction):
        print("  No se pudo calcular la corrección. Se detiene la iteración.")
        break

    if abs(t_delay_correction) <= t_delay_error:
        print("  Criterio de convergencia alcanzado.")
        break

    t_delay_actual = t_delay_suggested

else:
    print("  Aviso: se alcanzó max_iter sin cumplir el criterio exacto.")


# Recalcular una última vez con el t_delay final
res = calcular_con_delay(t_delay_actual)

t_delay_final = res["t_delay"]
t_delay_correction = res["t_delay_correction"]
t_delay_suggested = res["t_delay_suggested"]

print("===== Resultado final =====")
print(f"t_delay inicial = {t_delay:.12f} ps")
print(f"t_delay final   = {t_delay_final:.12f} ps")
print(f"última corrección estimada = {t_delay_correction:.12f} ps")


# =========================
# Recuperar variables finales
# =========================
E_sam_common = res["E_sam_common"]

E_ref_fft_input = res["E_ref_fft_input"]
E_sam_fft_input = res["E_sam_fft_input"]
window_fft = res["window_fft"]

FFT_ref = res["FFT_ref"]
FFT_sam = res["FFT_sam"]
f = res["f"]
amp_ref = res["amp_ref"]
amp_sam = res["amp_sam"]
valid = res["valid"]
r_sample = res["r_sample"]
R = res["R"]
phase_r = res["phase_r"]
mask_phase = res["mask_phase"]
slope_phase = res["slope_phase"]
offset_phase = res["offset_phase"]


# =========================
# n y k
# Polarización S a 45 grados
# =========================
theta = np.deg2rad(theta_deg)

cos_t = np.cos(theta)
sin_t = np.sin(theta)

N_sample = np.full_like(r_sample, np.nan + 1j*np.nan, dtype=complex)

valid_nk = (
    valid
    & np.isfinite(r_sample.real)
    & np.isfinite(r_sample.imag)
    & (np.abs(1 + r_sample) > 1e-12)
)

q = cos_t * (1 - r_sample[valid_nk]) / (1 + r_sample[valid_nk])

N_calc = np.sqrt(q**2 + sin_t**2)

# Elegir rama con parte real positiva
N_calc[np.real(N_calc) < 0] *= -1

N_sample[valid_nk] = N_calc

n = np.real(N_sample)
k = np.abs(np.imag(N_sample))


# =========================
# Épsilon y conductividad
# =========================
eps0 = 8.8541878128e-12  # F/m

# Frecuencia angular en rad/s
omega = 2 * np.pi * f * 1e12  # f está en THz

# Usamos N = n + i k, con k positivo
N_opt = n + 1j * k

# Permitividad compleja relativa
epsilon_complex = N_opt ** 2

epsilon_real = np.real(epsilon_complex)
epsilon_imag = np.imag(epsilon_complex)

# Conductividad compleja en S/m
# sigma = -i omega eps0 (epsilon - 1)
sigma_complex = -1j * omega * eps0 * (epsilon_complex - 1)

sigma_real = np.real(sigma_complex)
sigma_imag = np.imag(sigma_complex)

# Opcional: convertir a S/cm
sigma_real_Scm = sigma_real / 100
sigma_imag_Scm = sigma_imag / 100


# =========================
# GUARDAR RESULTADOS EN TXT
# =========================

output_folder = filepath / "resultados_TDS"
output_folder.mkdir(exist_ok=True)

sample_name = Path(sample).stem

try:
    t_delay_usado = t_delay_final
except NameError:
    t_delay_usado = t_delay

mask_freq = (f >= f_min) & (f <= f_max)

try:
    mask_guardar = mask_freq & valid
except NameError:
    mask_guardar = mask_freq


# ============================================================
# 1) Archivo sigma_sample: conductividad real e imaginaria S/m
# ============================================================

output_sigma = output_folder / f"sigma_{sample_name}.txt"

datos_sigma = np.column_stack([
    f[mask_guardar],
    sigma_real[mask_guardar],
    sigma_imag[mask_guardar]
])

header_sigma = f"""
Conductividad THz-TDS reflexión

"""

np.savetxt(
    output_sigma,
    datos_sigma,
    header=header_sigma,
    comments="# ",
    fmt="%.10e",
    delimiter="\t"
)


# ==================================================================
# 2) Archivo óptico: amplitud FFT, n, k, epsilon1 y epsilon2 vs freq
# ==================================================================

output_optico = output_folder / f"optico_{sample_name}.txt"

datos_optico = np.column_stack([
    f[mask_guardar],
    amp_sam[mask_guardar],
    n[mask_guardar],
    k[mask_guardar],
    epsilon_real[mask_guardar],
    epsilon_imag[mask_guardar]
])

header_optico = f"""
Propiedades ópticas THz-TDS reflexión

Archivo referencia: {reference}
Archivo muestra: {sample}

Rango de frecuencia guardado:
f_min = {f_min} THz
f_max = {f_max} THz

Parámetros:
n_base = {n_base}
theta_deg = {theta_deg}
t_delay_usado = {t_delay_usado} ps
f_fit_min = {f_fit_min} THz
f_fit_max = {f_fit_max} THz
slope_phase = {slope_phase} rad/THz
ultima_correccion_t_delay = {t_delay_correction} ps

Center padding:
usar_centerpad = {usar_centerpad}
centerpad_desired_length_factor = {centerpad_desired_length_factor}
centerpad_quitar_offset = {centerpad_quitar_offset}

Windowing:
usar_windowing = {usar_windowing}
window_type = {window_type}
normalizar_window = {normalizar_window}
kaiser_beta = {kaiser_beta}
tukey_alpha = {tukey_alpha}

Columnas:
frecuencia_THz
FFT_sample_amp
n
k
epsilon1_real
epsilon2_imag
"""

np.savetxt(
    output_optico,
    datos_optico,
    header=header_optico,
    comments="# ",
    fmt="%.10e",
    delimiter="\t"
)


print("Archivos TXT guardados en:")
print(output_sigma)
print(output_optico)

# ============================================================
# 2) Archivo sigmareal_sample: frecuencia y sigma real sin header
# ============================================================

output_sigma_real = output_folder / f"sigmareal_{sample_name}.txt"

datos_sigma_real = np.column_stack([
    f[mask_guardar],
    sigma_real[mask_guardar]
])

np.savetxt(
    output_sigma_real,
    datos_sigma_real,
    fmt="%.10e",
    delimiter="\t"
)

# =========================
# PLOTS
# =========================

# Pulsos igualados sin windowing
plt.figure(dpi=150)
plt.plot(t_common, E_ref_common, label='Referencia')
plt.plot(t_common, E_sam_common, label=f'Muestra, t_delay final = {t_delay_final:.6f} ps')
plt.xlabel('Tiempo (ps)')
plt.ylabel('Campo (V)')
plt.title('Pulsos igualados')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Ventana temporal usada para la FFT
plt.figure(dpi=150)
plt.plot(t_common, window_fft, label=f'Window: {window_type}')
plt.xlabel('Tiempo (ps)')
plt.ylabel('Amplitud de ventana')
plt.title('Ventana temporal aplicada antes de la FFT')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Pulsos usados realmente para FFT
plt.figure(dpi=150)
plt.plot(t_common, E_ref_fft_input, label='Referencia windowed')
plt.plot(t_common, E_sam_fft_input, label='Muestra windowed')
plt.xlabel('Tiempo (ps)')
plt.ylabel('Campo windowed')
plt.title(f'Pulsos después de windowing: {window_type}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# FFT normalizadas
plt.figure(dpi=150)
plt.plot(f, amp_ref , label='Referencia')
plt.plot(f, amp_sam , label='Muestra')
plt.xlabel('Frecuencia (THz)')
plt.ylabel('FFT ')
plt.title('FFT de los pulsos')
plt.xlim(f_min, f_max)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# FFT normalizadas
plt.figure(dpi=150)
plt.plot(f, amp_ref / np.max(amp_ref), label='Referencia')
plt.plot(f, amp_sam / np.max(amp_sam), label='Muestra')
plt.xlabel('Frecuencia (THz)')
plt.ylabel('FFT normalizada')
plt.title('FFT de los pulsos')
plt.xlim(f_min, f_max)
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Reflectividad
plt.figure(dpi=150)
plt.plot(f, R, label='R = |r|²')
plt.xlabel('Frecuencia (THz)')
plt.ylabel('Reflectividad')
plt.title('Reflectividad de la muestra')
plt.xlim(f_min, f_max)
plt.ylim(0, 1.2)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Fase del coeficiente de reflexión
fig, ax = plt.subplots(dpi=150)

ax.plot(f, phase_r, label='Fase de r')

if np.isfinite(slope_phase):
    ax.plot(
        f[mask_phase],
        slope_phase * f[mask_phase] + offset_phase,
        '--',
        label='Ajuste lineal final'
    )

ax.set_xlabel('Frecuencia (THz)')
ax.set_ylabel('Fase (rad)')
ax.set_title('Fase del coeficiente de reflexión')

ax.set_xlim(f_min, f_max)

# Autoescala Y usando solo el rango visible en X
mask = (
    (f >= f_min) &
    (f <= f_max) &
    np.isfinite(phase_r)
)

y = phase_r[mask]

if len(y) > 0:
    ax.set_ylim(np.nanmin(y), np.nanmax(y))

ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# n
plt.figure(dpi=150)
plt.plot(f, n, label='n')
plt.xlabel('Frecuencia (THz)')
plt.ylabel('n')
plt.title('Índice de refracción real')
plt.xlim(f_min, f_max)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# k
plt.figure(dpi=150)
plt.plot(f, k, label='k')
plt.xlabel('Frecuencia (THz)')
plt.ylabel('k')
plt.title('Coeficiente de extinción')
plt.xlim(f_min, f_max)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Épsilon real e imaginaria
plt.figure(dpi=150)
plt.plot(f, epsilon_real, label=r'$\epsilon_1$')
plt.plot(f, epsilon_imag, label=r'$\epsilon_2$')
plt.xlabel('Frecuencia (THz)')
plt.ylabel(r'$\epsilon$')
plt.title('Permitividad compleja')
plt.xlim(f_min, f_max)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Conductividad real e imaginaria en S/m
fig, ax = plt.subplots(dpi=150)

ax.plot(f, sigma_real, label=r'$\sigma_1$')
ax.plot(f, sigma_imag, label=r'$\sigma_2$')

ax.set_xlabel('Frecuencia (THz)')
ax.set_ylabel('Conductividad (S/m)')
ax.set_title('Conductividad compleja')

ax.set_xlim(f_min, f_max)

# Autoescalar eje Y usando solo la región visible en X
mask = (f >= f_min) & (f <= f_max)
y = np.concatenate([sigma_real[mask], sigma_imag[mask]])

ax.set_ylim(np.nanmin(y), np.nanmax(y))

ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()