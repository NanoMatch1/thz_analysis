from matplotlib import pyplot as plt
import numpy as np

import thztools as thz

def frfun(omega, _a, _eta):
    '''Rescale the waveform by _a and delay by _eta in frequency domain'''
    return _a * np.exp(-1j * omega * _eta)




n = 256  # Number of samples
dt = 0.05  # Sampling time [ps]
a = 0.5 # Scale factor
eta = 2.0 # Delay [ps]


t = thz.timebase(n, dt=dt)
mu = thz.wave(n, dt=dt)

# plt.plot(t, mu, label='Original Waveform')

Ew = np.fft.rfft(mu)

amplitude = np.abs(Ew)
phase = np.angle(Ew)

frequencies = np.fft.rfftfreq(n, d=dt)

fig, ax = plt.subplots(2,1)
phase_unwrapped = np.unwrap(phase)
ax[0].plot(frequencies, phase_unwrapped, label ='Unwrapped Phase')
ax[0].plot(frequencies, phase, label='Original Phase', linestyle='--', alpha=0.5)
ax[1].plot(frequencies, amplitude, label='Amplitude')
ax[0].set_xlabel("Frequency (THz)")
ax[0].set_ylabel("Phase (rad)")
ax[1].set_xlabel("Frequency (THz)")
ax[1].set_ylabel("Amplitude (arb. units)")

ax[0].legend()
ax[1].legend()
plt.show()

psi = thz.apply_frf(frfun, mu, dt=dt, args=(a, eta))

_, ax = plt.subplots()

ax.plot(t, mu, label=r'$\mu$')
ax.plot(t, psi, label=r'$\psi$')

ax.legend()

ax.set_xlabel('Time (ps)')
ax.set_ylabel(r'Amplitude (units of $\mu_{p})$')

ax.set_xticks(np.arange(0, 11, 5))
ax.set_xlim(0, 10)

plt.show()