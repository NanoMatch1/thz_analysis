import matplotlib.pyplot as plt
import numpy as np

def inspect_phase(dataset, **kwargs) -> dict:
    data_dict = {}
    title = kwargs.get("title", "Phase Spectrum")
    fig, ax = plt.subplots(2, 1)
    for filename, data in dataset.data.items():
        # print(data.processing_dict.keys())
        spectrum = data.data
        ax[0].plot(spectrum[:, 0], spectrum[:, 1], label=filename)
        ax[1].plot(spectrum[:, 0], spectrum[:, 2], label=filename)
        ax[1].plot(spectrum[:, 0], np.unwrap(spectrum[:, 2]), label=filename+" unwrapped", linestyle="--")

        data_dict[filename] = {
            "frequency": spectrum[:, 0],
            "amplitude": spectrum[:, 1],
            "phase": spectrum[:, 2],
            "unwrapped_phase": np.unwrap(spectrum[:, 2])
        }

    ax[0].set_title("Amplitude Spectrum")
    ax[1].set_title(title)
    ax[0].legend()
    ax[1].legend()
    plt.show()
    
    return data_dict


def phase_offset(phase_data, offset=0, **kwargs) -> dict:
    '''Subtracts a constant offset from the phase spectrum. Used for correcting for phase offsets due to timing or alignment errors.'''

    show_graph = kwargs.get("show_graph", False)

    dataX = phase_data[:, 0]
    data_amp = phase_data[:, 1]
    data_phase = phase_data[:, 2]

    offset_phase = np.unwrap(data_phase) - offset
    re_wrapped_phase = np.mod(offset_phase + np.pi, 2*np.pi) - np.pi

    if show_graph:
        plt.plot(dataX, data_phase, label="Original Phase"
                 )
        plt.plot(dataX, offset_phase, label=f"Offset Phase (offset={offset})", linestyle="--")

        plt.legend()
        plt.title(f"Phase Spectrum with Offset Correction (offset={offset})")
        plt.show()

    return re_wrapped_phase


