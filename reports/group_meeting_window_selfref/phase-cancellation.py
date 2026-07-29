import numpy as np
import matplotlib.pyplot as plt

"""Script to simulate the phase cancellation of a self-referenced group delay measurement, using a windowed THz pulse."""


def gaussian_cosine_pulse(time_ps, arrival_ps, amplitude, center_frequency_thz, envelope_sigma_ps):
    """A single quasi-few-cycle THz-like pulse: Gaussian envelope times a cosine carrier."""
    relative_time_ps = time_ps - arrival_ps
    envelope = np.exp(-(relative_time_ps ** 2) / (2.0 * envelope_sigma_ps ** 2))
    carrier = np.cos(2.0 * np.pi * center_frequency_thz * relative_time_ps)
    return amplitude * envelope * carrier



dataX = np.linspace(0, 7, 1000) # frequency in THz

phase_reference_front = dataX * 1
phase_reference_back = dataX * 1.5

phase_sample_front = dataX * 1.2
phase_sample_back = dataX * 1.8

pulse_shape = gaussian_cosine_pulse(dataX, arrival_ps=3, amplitude=1.0, center_frequency_thz=0.5, envelope_sigma_ps=0.5)
# reference_back = gaussian_cosine_pulse(dataX, arrival_ps=10, amplitude=1.0, center_frequency_thz=0.5, envelope_sigma_ps=0.5)   

prf = np.copy(pulse_shape) * 0.7


pulse_rf = np.insert(pulse_shape, 0, [0]*500)
ref_pulse = np.insert(pulse_rf, 0, pulse_shape)

plt.plot(ref_pulse, label="Reference Front")
plt.show()
breakpoint()

plt.plot(pulse_shape, label="Reference Front")
# plt.plot(reference_back, label="Reference Back")
plt.legend()

plt.show()