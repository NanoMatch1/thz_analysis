from matplotlib import pyplot as plt
import numpy as np

import thztools as thz

class Spectrum:
    def __init__(self, wave, dt):
        self.wave = wave
        self.dt = dt
        self.n = len(wave)

        self.frequencies = np.fft.rfftfreq(self.n, d=self.dt)
        self.ew = np.fft.rfft(self.wave)
        self.amplitude = np.abs(self.ew)
        self.phase = np.angle(self.ew)

    def plot_spectrum(self):
        fig, ax = plt.subplots(2, 1)
        phase_unwrapped = np.unwrap(self.phase)
        ax[0].plot(self.frequencies, phase_unwrapped, label='Unwrapped Phase')
        ax[0].plot(self.frequencies, self.phase, label='Original Phase', linestyle='--', alpha=0.5)
        ax[1].plot(self.frequencies, self.amplitude, label='Amplitude')
        ax[0].set_xlabel("Frequency (THz)")
        ax[0].set_ylabel("Phase (rad)")
        ax[1].set_xlabel("Frequency (THz)")
        ax[1].set_ylabel("Amplitude (arb. units)")

        ax[0].set_title("THz Spectrum")

        ax[0].legend()
        ax[1].legend()
        plt.show()

class THzWave:
    def __init__(self, n, dt):
        self.n = n
        self.dt = dt
        # self.generate_waveform()

    def generate_waveform(self):
        self.time = thz.timebase(self.n, dt=self.dt)
        self.wave = thz.wave(self.n, dt=self.dt)

    def plot_waveform(self):
        plt.plot(self.time, self.wave)
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (arb. units)")
        plt.title("THz Waveform")
        plt.show()

    def perform_fft(self):
        self.spectrum = Spectrum(self.wave, self.dt)
    
    def plot_spectrum(self):
        if not hasattr(self, 'spectrum'):
            print("Performing FFT...")
            self.spectrum = Spectrum(self.wave, self.dt)

        self.spectrum.plot_spectrum()

    def plot_all(self):
        '''Plots both waveform and spectrum and phase unwrapping in a gridspec layout'''
        fig = plt.figure(constrained_layout=True, figsize=(10, 6))
        gs = fig.add_gridspec(2, 2)
        ax_waveform = fig.add_subplot(gs[:, 0])
        ax_spectrum = fig.add_subplot(gs[0, 1])
        ax_phase = fig.add_subplot(gs[1, 1])

        # Plot waveform
        ax_waveform.plot(self.time, self.wave)
        ax_waveform.set_xlabel("Time (ps)")
        ax_waveform.set_ylabel("Amplitude (arb. units)")
        ax_waveform.set_title("THz Waveform")

        # plot spectrum
        if not hasattr(self, 'spectrum'):
            print("Performing FFT...")
            self.spectrum = Spectrum(self.wave, self.dt)

        # plot phase with unwrapping
        phase_unwrapped = np.unwrap(self.spectrum.phase)
        ax_phase.plot(self.spectrum.frequencies, phase_unwrapped, label='Unwrapped Phase')
        ax_phase.plot(self.spectrum.frequencies, self.spectrum.phase, label='Original Phase', linestyle='--', alpha=0.5)
        ax_phase.set_xlabel("Frequency (THz)")
        ax_phase.set_ylabel("Phase (rad)")
        ax_phase.set_title("THz Phase")
        ax_phase.legend()

        # plot amplitude
        ax_spectrum.plot(self.spectrum.frequencies, self.spectrum.amplitude, label='Amplitude')
        ax_spectrum.set_xlabel("Frequency (THz)")
        ax_spectrum.set_ylabel("Amplitude (arb. units)")
        ax_spectrum.set_title("THz Spectrum")
        ax_spectrum.legend()
        plt.show()


def frfun(omega, _a, _eta):
    '''Rescale the waveform by _a and delay by _eta in frequency domain'''
    return _a * np.exp(-1j * omega * _eta)


def generate_thz_waveform(n, dt):
    '''Generate a sample THz waveform'''

    n = 256  # Number of samples
    dt = 0.05  # Sampling time [ps]

    t = thz.timebase(n, dt=dt)
    mu = thz.wave(n, dt=dt)

    return t, mu

def test_1():
    '''Test THz waveform generation and FRF application'''
    a = 0.5 # Scale factor
    eta = 2.0 # Delay [ps]
    n = 256
    dt = 0.05  # Sampling time [ps]

    t, mu = generate_thz_waveform(n, dt)

    Ew = np.fft.rfft(mu)

    amplitude = np.abs(Ew)
    phase = np.angle(Ew)
    frequencies = np.fft.rfftfreq(n, d=dt)
    # wave = THzWave(n, dt)
    # wave.generate_waveform()
    # wave.plot_all()

    psi = thz.apply_frf(frfun, mu, dt=dt, args=(a, eta))


n = 256  # Number of samples
m = 50  # Number of simulated waveforms
dt = 0.05  # Sampling time [ps]

sigma_alpha = 1e-4  # Additive noise amplitude [signal units]
sigma_beta = 1e-2  # Multiplicative noise amplitude [dimensionless]
sigma_tau = 1e-3  # Time base noise amplitude [ps]

t = thz.timebase(n, dt=dt)
mu = thz.wave(n, dt=dt)

rng = np.random.default_rng(0)
a = 1.0 + 1e-2 * rng.standard_normal(m - 1)
eta = 1e-3 * rng.standard_normal(m - 1)

z = thz.scaleshift(
    np.repeat(np.atleast_2d(mu), m, axis=0),
    dt=dt,
    a=np.insert(a, 0, 1.0),
    eta=np.insert(eta, 0, 0.0)
).T

noise_model = thz.NoiseModel(
    sigma_alpha=sigma_alpha,
    sigma_beta=sigma_beta,
    sigma_tau=sigma_tau,
    dt=dt
)
x = z + noise_model.noise_sim(z, axis=0, seed=12345)

noise_res = thz.noisefit(
    x,
    sigma_alpha0=sigma_alpha,
    sigma_beta0=sigma_beta,
    sigma_tau0=sigma_tau,
    dt=dt
)
print(f"{noise_res.noise_model.sigma_alpha=}")
print(f"{noise_res.noise_model.sigma_beta=}")
print(f"{noise_res.noise_model.sigma_tau=}")

x_corrected = thz.scaleshift(
    x, a=1 / noise_res.a, eta=-noise_res.eta, axis=0
)
plt.plot(t, np.std(x_corrected, axis=1), "-", label="Data")
plt.plot(t, noise_res.noise_model.noise_amp(noise_res.mu), "--", label="Fit")
plt.legend()
plt.xlabel("t (ps)")
plt.ylabel(r"$\sigma(t)$")
plt.show()